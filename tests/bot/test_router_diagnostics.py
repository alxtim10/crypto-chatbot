from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest

from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.commands import (
    AnalyzeIntent,
    CompareIntent,
    PineIntent,
    RefusalIntent,
    resolve_followup_intent_with_trace,
)
from crypto_chatbot.bot.handlers import BotService
from crypto_chatbot.llm.gemini import GeminiNarrator, RawGeminiOutput
from crypto_chatbot.llm.guard import validate_conversational_answer
from crypto_chatbot.llm.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CONVERSATIONAL_QA_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
)
from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    FuturesTickerBatch,
    FuturesTickerSnapshot,
    OhlcvBatch,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    InputRange,
    MarketStructure,
    Regime,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.market import (
    MarketType,
    ProviderMetadata,
    QualityFlag,
    QualityStatus,
    Timeframe,
)
from crypto_chatbot.storage.models import LlmInteractionRow
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    LlmInteractionRecord,
    PurgeResult,
    StoredMessage,
    UserPreferences,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "router_casual_id.yaml"
FROZEN_FIXTURE_SHA256 = "45ae3cd648b3d210b2d927e9befea2958ef9299ec1e5573ebe2d8282d61e7ccc"
ANCHOR = datetime(2026, 11, 27, 8, 0, tzinfo=UTC)

EXPECTED_INTENTS = frozenset(
    {
        "ANALYZE",
        "FOLLOWUP",
        "EDU_QA",
        "MARKET_GENERAL",
        "SMALLTALK",
        "HELP",
        "OUT_OF_SCOPE",
        "UNKNOWN",
    }
)
LEGIT_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "DOGE", "SUI", "PEPE", "VVV", "ADA", "GAS", "MEME", "OMG"}
)


@dataclass(frozen=True)
class RouterUtterance:
    utterance_id: str
    text: str
    expected_intent: str
    has_history: bool
    expected_symbol: str | None
    tags: tuple[str, ...]


def _parse_yaml_scalar(raw: str) -> Any:
    val = raw.strip()
    if val == "null":
        return None
    if val == "true":
        return True
    if val == "false":
        return False
    if val.startswith('"') and val.endswith('"'):
        return json.loads(val)
    if val.startswith("[") and val.endswith("]"):
        return tuple(json.loads(val))
    return val


def load_router_fixture(path: Path = FIXTURE_PATH) -> tuple[RouterUtterance, ...]:
    """Parse `router_casual_id.yaml` using stdlib only (no PyYAML dependency)."""
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_utterances = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "utterances:":
            in_utterances = True
            continue
        if not in_utterances:
            continue
        if stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {}
            key, _, rest = stripped[2:].partition(":")
            current[key.strip()] = _parse_yaml_scalar(rest)
        elif current is not None:
            key, _, rest = stripped.partition(":")
            current[key.strip()] = _parse_yaml_scalar(rest)

    if current is not None:
        items.append(current)

    return tuple(
        RouterUtterance(
            utterance_id=str(item["id"]),
            text=str(item["text"]),
            expected_intent=str(item["expected_intent"]),
            has_history=bool(item["has_history"]),
            expected_symbol=item["expected_symbol"],
            tags=tuple(item.get("tags", ())),
        )
        for item in items
    )


class _RecordingMemoryRepo:
    def __init__(self) -> None:
        self.prefs: dict[int, UserPreferences] = {}
        self.msgs: dict[int, list[StoredMessage]] = {}
        self.interactions: list[LlmInteractionRecord] = []

    def ensure_user(self, telegram_user_id: int) -> None:
        del telegram_user_id

    def get_preferences(self, telegram_user_id: int) -> UserPreferences:
        return self.prefs.get(telegram_user_id, UserPreferences())

    def save_preferences(self, telegram_user_id: int, prefs: UserPreferences) -> None:
        self.prefs[telegram_user_id] = prefs

    def append_message(
        self,
        telegram_user_id: int,
        *,
        role: Literal["user", "assistant"],
        content: str,
        snapshot_id: str | None = None,
        max_messages: int = 20,
        created_at: datetime | None = None,
    ) -> StoredMessage:
        msg = StoredMessage(
            role=role,
            content=content,
            snapshot_id=snapshot_id,
            created_at=created_at or ANCHOR,
        )
        bucket = self.msgs.setdefault(telegram_user_id, [])
        bucket.append(msg)
        if len(bucket) > max_messages:
            del bucket[: len(bucket) - max_messages]
        return msg

    def get_recent_messages(
        self,
        telegram_user_id: int,
        *,
        limit: int = 20,
    ) -> tuple[StoredMessage, ...]:
        return tuple(self.msgs.get(telegram_user_id, [])[-limit:])

    def record_llm_interaction(self, record: LlmInteractionRecord) -> None:
        self.interactions.append(record)

    def forget_user_data(self, telegram_user_id: int) -> ForgetResult:
        msgs = len(self.msgs.pop(telegram_user_id, []))
        before = len(self.interactions)
        self.interactions = [r for r in self.interactions if r.telegram_user_id != telegram_user_id]
        return ForgetResult(
            deleted_messages=msgs,
            deleted_conversations=1 if msgs else 0,
            deleted_preferences=0,
            deleted_interactions=before - len(self.interactions),
        )

    def purge_expired(
        self,
        *,
        now: datetime,
        message_retention_days: int = 30,
        interaction_retention_days: int = 30,
        snapshot_retention_days: int = 90,
    ) -> PurgeResult:
        del now, message_retention_days, interaction_retention_days, snapshot_retention_days
        return PurgeResult(
            deleted_messages=0,
            deleted_conversations=0,
            deleted_interactions=0,
            deleted_snapshots=0,
        )


class _DiagnosticAnalysisService:
    """Offline stub mimicking exchange: valid coins READY, bogus tokens INSUFFICIENT_DATA."""

    async def analyze(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        include_mtf: bool = True,
        include_derivatives: bool = True,
        persist: bool = True,
    ) -> AnalysisSnapshot:
        del include_mtf, include_derivatives, persist
        mapping = resolve_default_perp_symbol(symbol)
        if mapping.base not in LEGIT_CRYPTO_BASES:
            return AnalysisSnapshot(
                snapshot_id=f"snap-err-{mapping.base}",
                provider="ccxt",
                venue="binance",
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                market_type=MarketType.PERPETUAL,
                timeframe=timeframe,
                as_of=ANCHOR,
                calc_version="1.0.0",
                quality_status=QualityStatus.INSUFFICIENT_DATA,
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                limitations=("Simbol tidak ditemukan di bursa.",),
            )
        return AnalysisSnapshot(
            snapshot_id=f"snap-{mapping.base}",
            provider="ccxt",
            venue="binance",
            canonical_symbol=mapping.canonical_symbol,
            provider_symbol=mapping.provider_symbol,
            market_type=MarketType.PERPETUAL,
            timeframe=timeframe,
            as_of=ANCHOR,
            input_range=InputRange(
                first_open_time=ANCHOR - timedelta(hours=100),
                last_open_time=ANCHOR - timeframe.duration,
                candle_count=50,
                sha256="d" * 64,
            ),
            calc_version="1.0.0",
            technical=TechnicalMetrics(last_close=100.0),
            regime=Regime.TRENDING_UP,
            market_structure=MarketStructure.BULLISH,
            facts=(
                Fact(
                    fact_id="F1",
                    label="Harga penutupan terakhir",
                    value=100.0,
                    unit="USDT",
                    display_format="price",
                    source="ccxt:binance",
                    as_of=ANCHOR,
                ),
            ),
            quality_status=QualityStatus.READY,
        )

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        return FuturesTickerBatch(
            venue="binance",
            received_time=ANCHOR,
            tickers=tuple(
                FuturesTickerSnapshot(
                    venue="binance",
                    canonical_symbol=f"{base}/USDT:USDT",
                    provider_symbol=f"BINANCE:{base}USDT.P",
                    last_price=100.0,
                    change_24h_pct=2.0,
                    quote_volume_24h_usd=10_000_000.0,
                    observed_time=ANCHOR,
                )
                for base in sorted(LEGIT_CRYPTO_BASES)
            ),
        )

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        del limit
        mapping = resolve_default_perp_symbol(symbol)
        return OhlcvBatch(
            metadata=ProviderMetadata(
                provider="ccxt",
                venue="binance",
                market_type=MarketType.PERPETUAL,
                base_unit=mapping.base,
                quote_currency="USDT",
            ),
            canonical_symbol=mapping.canonical_symbol,
            provider_symbol=mapping.provider_symbol,
            timeframe=timeframe,
            received_time=ANCHOR,
            candles=(),
            quality_flags=(),
        )


class _ScriptedTransport:
    def __init__(self, texts: list[str] | None = None, *, fail: bool = False) -> None:
        self._texts = list(texts or [])
        self._fail = fail

    async def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: str,
        response_json_schema: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
    ) -> RawGeminiOutput:
        del (
            model,
            system_instruction,
            contents,
            response_json_schema,
            max_output_tokens,
            temperature,
        )
        if self._fail:
            raise TimeoutError("simulated timeout")
        payload = self._texts.pop(0) if self._texts else "{}"
        return RawGeminiOutput(
            text=payload,
            model_version="gemini-2.5-flash",
            prompt_tokens=42,
            output_tokens=18,
        )


class _HealthyFakeGeminiTransport:
    """Fake healthy Gemini Layer-2 classifier, Q&A responder, and snapshot narrator."""

    _COIN_ANALYZE_RE = re.compile(
        r"\b(ADA|VVV|BTC|ETH|SOL|DOGE|SUI|PEPE)\b",
        re.IGNORECASE,
    )
    _TF_RE = re.compile(r"\b(15m|1h|4h|1d|scalp|intraday|swing)\b", re.IGNORECASE)
    _POLICY_RE = re.compile(
        r"(?:\bbelikan\b|\bpasang\s+order\b|\bberapa\s+persen\s+modal\b|\bpasti\s+profit\b)",
        re.IGNORECASE,
    )
    _FOLLOWUP_TF_RE = re.compile(
        r"^(?:kalau\s+(?:buat\s+)?|coba\s+|yang\s+)?(15m|1h|4h|1d|scalp)(?:\s+gimana)?\??$",
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_tags(raw: str) -> str:
        return re.sub(r"</?untrusted_data>", "", raw).strip()

    async def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: str,
        response_json_schema: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
    ) -> RawGeminiOutput:
        del model, response_json_schema, max_output_tokens, temperature
        if system_instruction == INTENT_CLASSIFIER_SYSTEM_PROMPT:
            payload_obj = json.loads(contents)
            user_msg = self._strip_tags(str(payload_obj.get("user_message", "")))
            recent_syms = list(payload_obj.get("recent_symbols", []))
            if self._POLICY_RE.search(user_msg):
                resp: dict[str, Any] = {"intent": "unsupported", "symbols": [], "timeframe": None}
            elif self._FOLLOWUP_TF_RE.match(user_msg) and recent_syms:
                tf_m = self._FOLLOWUP_TF_RE.match(user_msg)
                resp = {
                    "intent": "analyze",
                    "symbols": [],
                    "timeframe": tf_m.group(1).lower() if tf_m else None,
                }
            elif self._COIN_ANALYZE_RE.search(user_msg) and (
                "analisa" in user_msg.lower()
                or "cek" in user_msg.lower()
                or "gimana" in user_msg.lower()
                or "usdt" in user_msg.lower()
            ):
                coins = [m.upper() for m in self._COIN_ANALYZE_RE.findall(user_msg)]
                tf_m = self._TF_RE.search(user_msg)
                resp = {
                    "intent": "analyze",
                    "symbols": coins[:1],
                    "timeframe": tf_m.group(1).lower() if tf_m else None,
                }
            else:
                resp = {"intent": "chat_qa", "symbols": [], "timeframe": None}
            text = json.dumps(resp, ensure_ascii=False)
        elif system_instruction == CONVERSATIONAL_QA_SYSTEM_PROMPT:
            q_match = re.search(
                r"Pertanyaan pengguna:\s*<untrusted_data>(.*?)</untrusted_data>",
                contents,
                re.DOTALL,
            )
            q_text = (q_match.group(1) if q_match else contents).strip().lower()
            if "rsi 70" in q_text or "rsi di bawah 30" in q_text:
                answer = (
                    "RSI 70 menandakan momentum beli historis tinggi, bukan jaminan harga "
                    "pasti turun. Tetap kombinasikan dengan struktur harga dan manajemen risiko."
                )
            elif "all-in" in q_text:
                answer = (
                    "Hindari all-in dalam satu posisi karena tidak ada strategi trading yang "
                    "bebas risiko. Batasi risiko per transaksi agar modal tetap terjaga."
                )
            elif "ema" in q_text or "macd" in q_text or "leverage" in q_text:
                answer = (
                    "Tidak ada indikator yang pasti 100% akurat di semua kondisi pasar. "
                    "Gunakan manajemen risiko dan batas invalidasi yang jelas."
                )
            elif "makasih" in q_text or "halo" in q_text or "pagi" in q_text or "oke" in q_text:
                answer = "Sama-sama! Kalau butuh penjelasan konsep atau riset koin, kabari saja."
            else:
                answer = (
                    "Secara edukasi, konsep tersebut membantu membaca konteks pasar dan risiko "
                    "secara lebih terstruktur tanpa menjanjikan hasil tertentu."
                )
            text = json.dumps({"answer": answer}, ensure_ascii=False)
        elif system_instruction == ANALYSIS_SYSTEM_PROMPT:
            text = json.dumps(
                {
                    "summary": "Harga penutupan berada di {F1} berdasarkan candle tertutup.",
                    "supporting_factors": ["Data harga terakhir tercatat pada {F1}."],
                    "contradicting_factors": [
                        "Kondisi jangka pendek tetap memiliki ketidakpastian."
                    ],
                    "bullish_scenario": "Bertahan di atas {F1} menjaga struktur saat ini.",
                    "bearish_scenario": (
                        "Penurunan di bawah {F1} menandakan pelemahan jangka pendek."
                    ),
                    "limitations": ["Hanya data historis bursa."],
                    "fact_ids": ["F1"],
                },
                ensure_ascii=False,
            )
        else:
            text = "{}"
        return RawGeminiOutput(
            text=text,
            model_version="gemini-2.5-flash-healthy",
            prompt_tokens=60,
            output_tokens=30,
        )


def test_router_fixture_schema_and_coverage() -> None:
    actual_sha256 = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert actual_sha256 == FROZEN_FIXTURE_SHA256
    utterances = load_router_fixture()
    assert len(utterances) == 100
    assert len({u.utterance_id for u in utterances}) == 100
    assert {u.expected_intent for u in utterances} == EXPECTED_INTENTS
    texts = {u.text for u in utterances}
    for required in ("makasih", "terus?", "kalau 1h?", "apa itu RSI 70?"):
        assert required in texts
    assert any(
        "ada " in u.text.lower() for u in utterances if u.expected_intent == "MARKET_GENERAL"
    )
    assert any("ADA" in u.text for u in utterances if u.expected_intent == "ANALYZE")


def test_llm_interaction_row_has_routing_trace_columns() -> None:
    cols = set(LlmInteractionRow.__table__.columns.keys())
    expected_cols = {
        "layer",
        "detected_symbol",
        "intent",
        "confidence",
        "fallback_reason",
        "guard_violations",
    }
    assert expected_cols.issubset(cols)
    assert LlmInteractionRow.__table__.c.snapshot_id.nullable is True


@pytest.mark.asyncio
async def test_routing_trace_persisted_without_pii_across_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    repo = _RecordingMemoryRepo()
    memory = MemoryService(repo)
    transport = _ScriptedTransport(
        [
            # 1) Layer 2 classify_intent -> chat_qa
            json.dumps({"intent": "chat_qa", "symbols": [], "timeframe": None}),
            # 2) QA answer -> blocked by guard ("pasti")
            json.dumps({"answer": "RSI 70 pasti membuat harga berbalik turun."}),
            # 3) Layer 2 classify_intent -> chat_qa
            json.dumps({"intent": "chat_qa", "symbols": [], "timeframe": None}),
            # 4) QA answer -> valid
            json.dumps({"answer": "RSI 70 menandakan momentum beli historis tinggi."}),
            # 5) Layer 2 classify_intent -> fails (invalid json)
            "not-json",
            # 6) QA fallback -> empty response
            "",
        ]
    )
    narrator = GeminiNarrator(model="gemini-2.5-flash", transport=transport, max_guard_retries=0)
    bot = BotService(
        access=AccessController((111,)),
        analysis_service=_DiagnosticAnalysisService(),  # type: ignore[arg-type]
        narrator=narrator,
        memory=memory,
    )

    # Non-allowlisted user persists nothing
    await bot.handle_text_message(999, "cek BTC 4h rahasia pribadi")
    assert repo.interactions == []

    # 1) Hard refusal (spot hint) -> layer1 refusal_hard
    secret_phrase = "pesan-rahasia-user-123"
    await bot.handle_text_message(111, f"analisa BTC spot {secret_phrase}")
    rec_hard = repo.interactions[-1]
    assert rec_hard.layer == "layer1"
    assert rec_hard.intent == "refusal_hard"
    assert rec_hard.fallback_reason == "spot_market_unsupported"
    assert rec_hard.snapshot_id is None

    # 2) Whale wallet address -> never leaks address into detected_symbol or logs
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    await bot.handle_text_message(111, f"cek wallet {wallet}")
    rec_whale = repo.interactions[-1]
    assert rec_whale.layer == "layer1"
    assert rec_whale.intent == "whale"
    assert rec_whale.detected_symbol is None

    # 3) Chat QA guard rejected -> records guard_violations
    await bot.handle_text_message(111, "apa itu RSI 70?")
    rec_qa_guard = repo.interactions[-1]
    assert rec_qa_guard.layer == "layer2"
    assert rec_qa_guard.intent == "chat_qa"
    assert rec_qa_guard.fallback_reason == "qa_guard_rejected"
    assert "overconfidence_lexicon:pasti" in rec_qa_guard.guard_violations
    assert rec_qa_guard.validation_passed is False

    # 4) Chat QA valid
    await bot.handle_text_message(111, "apa itu RSI 70?")
    rec_qa_ok = repo.interactions[-1]
    assert rec_qa_ok.layer == "layer2"
    assert rec_qa_ok.intent == "chat_qa"
    assert rec_qa_ok.fallback_reason is None
    assert rec_qa_ok.validation_passed is True

    # 5) Classifier failure + QA failure
    await bot.handle_text_message(111, "halo bro")
    rec_cls_fail = repo.interactions[-1]
    assert rec_cls_fail.layer == "layer2_fallback"
    assert rec_cls_fail.intent == "chat_qa"
    assert rec_cls_fail.fallback_reason == "classifier_invalid_json_or_schema|qa_empty_response"

    # PII check: raw user strings & wallet address never appear in any record or routing_trace log
    for rec in repo.interactions:
        dumped = rec.model_dump_json()
        assert secret_phrase not in dumped
        assert wallet not in dumped
    trace_logs = [r.getMessage() for r in caplog.records if "routing_trace" in r.getMessage()]
    assert trace_logs
    for line in trace_logs:
        assert secret_phrase not in line
        assert wallet not in line


@dataclass(frozen=True)
class ScenarioDiagnostics:
    name: str
    totals: Counter[str]
    fallback_traces: Counter[str]
    cannot_analyze: Counter[str]
    reasons: Counter[str]
    guard_rejections: Counter[str]


async def _run_bot_scenario(
    name: str,
    *,
    utterances: tuple[RouterUtterance, ...],
    history: tuple[StoredMessage, ...],
    transport: Any,
) -> ScenarioDiagnostics:
    repo = _RecordingMemoryRepo()
    memory = MemoryService(repo)
    narrator = GeminiNarrator(
        model="gemini-2.5-flash",
        transport=transport,
        max_guard_retries=0,
    )
    bot = BotService(
        access=AccessController((111,), rate_limit_per_minute=500),
        analysis_service=_DiagnosticAnalysisService(),  # type: ignore[arg-type]
        narrator=narrator,
        memory=memory,
    )

    totals: Counter[str] = Counter()
    fallback_traces: Counter[str] = Counter()
    cannot_analyze: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    guard_rejections: Counter[str] = Counter()

    for utt in utterances:
        totals[utt.expected_intent] += 1
        repo.msgs[111] = list(history) if utt.has_history else []
        before_count = len(repo.interactions)
        chunks = await bot.handle_text_message(111, utt.text)
        reply = "\n".join(chunks)
        new_recs = repo.interactions[before_count:]
        assert len(new_recs) >= 1
        rec = new_recs[-1]
        if rec.fallback_reason is not None:
            fallback_traces[utt.expected_intent] += 1
            reasons[rec.fallback_reason] += 1
        if rec.guard_violations:
            guard_rejections[utt.expected_intent] += 1
        is_cannot_analyze = (
            "belum dapat menentukan simbol" in reply.lower()
            or "data tidak mencukupi" in reply.lower()
            or "anda tetap bisa menyebut nama koin" in reply.lower()
            or "belum didukung" in reply.lower()
        )
        if is_cannot_analyze:
            cannot_analyze[utt.expected_intent] += 1

    return ScenarioDiagnostics(
        name=name,
        totals=totals,
        fallback_traces=fallback_traces,
        cannot_analyze=cannot_analyze,
        reasons=reasons,
        guard_rejections=guard_rejections,
    )


def _print_scenario_report(diag: ScenarioDiagnostics) -> None:
    total_all = sum(diag.totals.values())
    fb_all = sum(diag.fallback_traces.values())
    ca_all = sum(diag.cannot_analyze.values())
    print(f"\nScenario [{diag.name}] per expected_intent:")
    for intent_name in sorted(EXPECTED_INTENTS):
        tot = diag.totals[intent_name]
        fb = diag.fallback_traces[intent_name]
        ca = diag.cannot_analyze[intent_name]
        grd = diag.guard_rejections[intent_name]
        print(
            f"  {intent_name:14s}: total={tot:2d} fallback_trace={fb:2d} "
            f"({fb / tot * 100.0:5.1f}%) cannot_analyze_or_coin_prompt={ca:2d} "
            f"({ca / tot * 100.0:5.1f}%) guard_hits={grd:2d}"
        )
    print(
        f"  {'TOTAL':14s}: total={total_all:2d} fallback_trace={fb_all:2d} "
        f"({fb_all / total_all * 100.0:5.1f}%) cannot_analyze={ca_all:2d} "
        f"({ca_all / total_all * 100.0:5.1f}%)"
    )
    print(f"  top fallback_reasons: {diag.reasons.most_common(10)}")


@pytest.mark.asyncio
async def test_evaluate_router_casual_id_fixture_diagnostics() -> None:
    """Run all 100 utterances through Layer-1, L2-unavailable, and L2-healthy scenarios."""
    utterances = load_router_fixture()
    history = (
        StoredMessage(
            role="user", content="/analyze BTC 4h", created_at=ANCHOR - timedelta(minutes=2)
        ),
        StoredMessage(
            role="assistant",
            content="[BTC/USDT:USDT 4h] Harga {F1} di atas EMA 200 ({F6}).",
            snapshot_id="snap-btc-seed",
            created_at=ANCHOR - timedelta(minutes=1),
        ),
    )

    # Pass A: Layer 1 only
    l1_total: Counter[str] = Counter()
    l1_unmatched: Counter[str] = Counter()
    l1_hard_refusal: Counter[str] = Counter()
    l1_false_pos_symbols: list[tuple[str, str, str]] = []
    l1_ada_missed: list[tuple[str, str, str]] = []
    l1_context_hijack: list[tuple[str, str, str]] = []
    l1_reasons: Counter[str] = Counter()

    for utt in utterances:
        l1_total[utt.expected_intent] += 1
        hist = history if utt.has_history else ()
        intent, trace = resolve_followup_intent_with_trace(utt.text, recent_messages=hist)
        if isinstance(intent, RefusalIntent):
            if intent.is_hard_refusal:
                l1_hard_refusal[utt.expected_intent] += 1
            else:
                l1_unmatched[utt.expected_intent] += 1
            if trace.fallback_reason:
                l1_reasons[trace.fallback_reason] += 1
            if utt.expected_intent == "ANALYZE" and utt.expected_symbol == "ADA/USDT:USDT":
                l1_ada_missed.append((utt.utterance_id, utt.text, trace.intent or "refusal"))
        else:
            if isinstance(intent, (AnalyzeIntent, CompareIntent, PineIntent)):
                sym_str = (
                    ",".join(intent.symbols) if isinstance(intent, CompareIntent) else intent.symbol
                )
                base = sym_str.split("/")[0]
                if base not in LEGIT_CRYPTO_BASES or (
                    base in {"GAS", "MEME", "OMG"} and utt.expected_intent != "ANALYZE"
                ):
                    l1_false_pos_symbols.append((utt.utterance_id, utt.text, sym_str))
                elif utt.expected_intent in ("EDU_QA", "MARKET_GENERAL", "SMALLTALK", "HELP"):
                    l1_context_hijack.append((utt.utterance_id, utt.text, sym_str))

    # Pass B1 & B2: Full BotService with L2 unavailable vs L2 healthy
    diag_unavailable = await _run_bot_scenario(
        "L2 unavailable",
        utterances=utterances,
        history=history,
        transport=_ScriptedTransport(fail=True),
    )
    diag_healthy = await _run_bot_scenario(
        "L2 healthy",
        utterances=utterances,
        history=history,
        transport=_HealthyFakeGeminiTransport(),
    )

    # Pass C: Conversational guard probe on common Indonesian educational disclaimers
    guard_probes = [
        "RSI 70 bukan jaminan harga pasti turun, tetap kelola risiko.",
        "Tidak ada indikator yang pasti 100% akurat di semua kondisi pasar.",
        "Hindari all-in pada satu posisi agar risiko tetap terkendali.",
        "Tidak ada strategi trading yang bebas risiko, selalu gunakan stop loss.",
        "RSI 70 merupakan level jenuh beli, bukan jaminan pembalikan arah.",
    ]
    guard_hits = {sample: validate_conversational_answer(sample).reasons for sample in guard_probes}

    print(
        f"\n=== DIAGNOSTIC REPORT (router_casual_id.yaml sha256={FROZEN_FIXTURE_SHA256[:12]}) ==="
    )
    print("Pass A (Layer 1 deterministic) per expected_intent:")
    for intent_name in sorted(EXPECTED_INTENTS):
        tot = l1_total[intent_name]
        unm = l1_unmatched[intent_name]
        hrd = l1_hard_refusal[intent_name]
        rate = ((unm + hrd) / tot * 100.0) if tot else 0.0
        print(
            f"  {intent_name:14s}: total={tot:2d} soft_unmatched={unm:2d} "
            f"hard_refusal={hrd:2d} l1_fallback_rate={rate:5.1f}%"
        )
    print(f"Layer 1 fallback_reasons: {dict(l1_reasons.most_common())}")
    print(f"False-positive symbol detections ({len(l1_false_pos_symbols)}): {l1_false_pos_symbols}")
    print(f"ADA false-negative misses ({len(l1_ada_missed)}): {l1_ada_missed}")
    print(f"Context hijacks ({len(l1_context_hijack)}): {l1_context_hijack}")
    _print_scenario_report(diag_unavailable)
    _print_scenario_report(diag_healthy)
    print(f"\nGuard false-positive probes: {guard_hits}")

    assert sum(l1_total.values()) == 100
