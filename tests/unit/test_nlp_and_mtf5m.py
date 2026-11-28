from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest

from crypto_chatbot.analysis.indicators import compute_technical_metrics
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.service import _build_mtf_overview
from crypto_chatbot.analysis.setup import build_trade_setup
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import StructureAnalysis
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.commands import (
    AnalyzeIntent,
    CompareIntent,
    PineIntent,
    RefusalIntent,
    ScanIntent,
    parse_analyze_args,
    parse_compare_args,
    resolve_followup_intent,
)
from crypto_chatbot.bot.handlers import BotService
from crypto_chatbot.llm.gemini import GeminiNarrator, RawGeminiOutput
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    FactorDirection,
    FactorItem,
    InputRange,
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
    TradeSetup,
)
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityStatus,
    Timeframe,
)
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    LlmInteractionRecord,
    PurgeResult,
    StoredMessage,
    UserPreferences,
)

ANCHOR = datetime(2026, 11, 23, 8, 0, tzinfo=UTC)


def _candle_tf(
    idx: int,
    *,
    o: float,
    h: float,
    low_p: float,
    c: float,
    tf: Timeframe,
) -> MarketCandle:
    open_t = ANCHOR + idx * tf.duration
    close_t = open_t + tf.duration
    return MarketCandle(
        provider="ccxt",
        venue="gate",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="GATE:BTCUSDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=tf,
        open_time=open_t,
        close_time=close_t,
        received_time=close_t,
        available_time=close_t,
        open=o,
        high=h,
        low=low_p,
        close=c,
        base_volume=500.0,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )


def _make_batch(
    tf: Timeframe,
    count: int = 50,
    base_price: float = 100.0,
    end_time: datetime = ANCHOR + timedelta(days=60),
) -> OhlcvBatch:
    candles: list[MarketCandle] = []
    price = base_price
    for i in range(count):
        open_t = end_time - (count - i) * tf.duration
        close_t = open_t + tf.duration
        o = price
        c = price * 1.002
        h = max(o, c) * 1.001
        low_p = min(o, c) * 0.999
        price = c
        candles.append(
            MarketCandle(
                provider="ccxt",
                venue="gate",
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                timeframe=tf,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=o,
                high=h,
                low=low_p,
                close=c,
                base_volume=500.0,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider="ccxt",
            venue="gate",
            market_type=MarketType.PERPETUAL,
            base_unit="BTC",
            quote_currency="USDT",
        ),
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="GATE:BTCUSDT.P",
        timeframe=tf,
        received_time=candles[-1].close_time + timedelta(seconds=5),
        candles=tuple(candles),
        quality_flags=(),
    )


def test_micro_5m_confirmation_attaches_note_and_appears_in_mtf_overview() -> None:
    metrics = TechnicalMetrics(
        last_close=100.0,
        ema_20=99.0,
        ema_50=97.0,
        ema_200=90.0,
        rsi_14=58.0,
        atr_14=2.0,
        atr_pct=2.0,
        adx_14=30.0,
        plus_di_14=28.0,
        minus_di_14=12.0,
        atr_percentile_100=45.0,
    )
    candles_4h = [
        _candle_tf(i, o=95.0, h=97.0, low_p=94.0, c=96.0, tf=Timeframe.H4) for i in range(5)
    ]
    as_of = candles_4h[-1].close_time
    struct = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(
            StructureEvent(
                event_type=StructureEventType.SWING_HIGH,
                event_time=as_of - timedelta(hours=16),
                confirmation_time=as_of - timedelta(hours=8),
                level=102.0,
            ),
        ),
        last_swing_lows=(
            StructureEvent(
                event_type=StructureEventType.SWING_LOW,
                event_time=as_of - timedelta(hours=20),
                confirmation_time=as_of - timedelta(hours=12),
                level=96.0,
            ),
        ),
    )
    smc = analyze_smc(candles_4h, structure=struct)

    # Construct steady rising 5m candles up to as_of
    m5_candles = []
    for i in range(40):
        open_t = as_of - timedelta(minutes=5 * (40 - i))
        close_t = open_t + timedelta(minutes=5)
        m5_candles.append(
            MarketCandle(
                provider="ccxt",
                venue="gate",
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                timeframe=Timeframe.M5,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=98.0 + i * 0.05,
                high=98.2 + i * 0.05,
                low=97.9 + i * 0.05,
                close=98.1 + i * 0.05,
                base_volume=200.0,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )

    no_micro = build_trade_setup(
        metrics=metrics,
        regime=Regime.TRENDING_UP,
        structure=struct,
        smc=smc,
        primary_timeframe=Timeframe.H4,
        as_of=as_of,
    )
    with_micro = build_trade_setup(
        metrics=metrics,
        regime=Regime.TRENDING_UP,
        structure=struct,
        smc=smc,
        primary_timeframe=Timeframe.H4,
        micro_candles=m5_candles,
        as_of=as_of,
    )

    assert no_micro.status is SetupStatus.READY
    assert with_micro.status is SetupStatus.READY
    assert with_micro.bias is SetupBias.LONG
    # SL must never be tightened by 5m candles
    assert with_micro.stop_loss == no_micro.stop_loss
    assert with_micro.ltf_confirmation is not None
    assert "5m" in with_micro.ltf_confirmation

    # MTF overview includes 5m when provided alongside 1d, 4h, 1h, 15m
    b_d1 = _make_batch(Timeframe.D1)
    b_h4 = _make_batch(Timeframe.H4)
    b_h1 = _make_batch(Timeframe.H1)
    b_m15 = _make_batch(Timeframe.M15)
    b_m5 = _make_batch(Timeframe.M5)
    _ = compute_technical_metrics(b_h4.candles)
    _ = classify_regime(metrics)

    overview = _build_mtf_overview(
        b_h4,
        mtf_batches={
            Timeframe.D1: b_d1,
            Timeframe.H1: b_h1,
            Timeframe.M15: b_m15,
            Timeframe.M5: b_m5,
        },
        as_of=b_h4.candles[-1].close_time,
        use_available_time=False,
    )
    assert tuple(ov.timeframe for ov in overview) == (
        Timeframe.D1,
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
    )


def test_fast_nlp_regex_routes_unified_mtf_scan_pine_and_preserves_refusals() -> None:
    history = [
        StoredMessage(role="assistant", content="[SUI/USDT:USDT 15m] ok", created_at=ANCHOR),
    ]

    # Casual unified MTF analysis request defaults to 4h (not stuck on prior 15m!)
    btc_intent = resolve_followup_intent(
        "coba analisa btc sekarang, ada setup bagus gak",
        recent_messages=history,
    )
    assert btc_intent == AnalyzeIntent(symbol="BTC/USDT:USDT", timeframe=Timeframe.H4, detail=False)

    # Scan query without explicit style/timeframe -> Multi-style ScanIntent (15m, 1h, 4h)
    scan_intent = resolve_followup_intent(
        "ada token yang bisa disetup gak?",
        recent_messages=history,
    )
    assert scan_intent == ScanIntent(
        timeframe=Timeframe.H4,
        timeframes=MULTI_STYLE_SCAN_TIMEFRAMES,
    )
    for general_scan_q in ("ada setup apa?", "setup apa sekarang?", "ada sinyal apa?"):
        assert resolve_followup_intent(general_scan_q, recent_messages=history) == ScanIntent(
            timeframe=Timeframe.H4,
            timeframes=MULTI_STYLE_SCAN_TIMEFRAMES,
        )

    # Pine Script request referencing recent symbol
    pine_recent = resolve_followup_intent(
        "bikinin pine scriptnya dong",
        recent_messages=history,
    )
    assert pine_recent == PineIntent(symbol="SUI/USDT:USDT", timeframe=Timeframe.M15)

    # Pine Script request with explicit coin resets to default 4h unified MTF
    pine_btc = resolve_followup_intent(
        "buatkan pine script BTC",
        recent_messages=history,
    )
    assert pine_btc == PineIntent(symbol="BTC/USDT:USDT", timeframe=Timeframe.H4)

    # Detail request
    detail_eth = resolve_followup_intent(
        "analisa ETH detail lengkap dong",
        recent_messages=history,
    )
    assert detail_eth == AnalyzeIntent(symbol="ETH/USDT:USDT", timeframe=Timeframe.H4, detail=True)

    # Trading style keywords in natural chat ("scalp", "intraday", "swing")
    scalp_sol = resolve_followup_intent(
        "coba analisa sol buat scalp",
        recent_messages=history,
    )
    assert scalp_sol == AnalyzeIntent(symbol="SOL/USDT:USDT", timeframe=Timeframe.M15, detail=False)

    intraday_btc = resolve_followup_intent(
        "cek btc buat intraday sekarang",
        recent_messages=history,
    )
    assert intraday_btc == AnalyzeIntent(
        symbol="BTC/USDT:USDT", timeframe=Timeframe.H1, detail=False
    )

    swing_eth = resolve_followup_intent(
        "analisa eth buat swing",
        recent_messages=history,
    )
    assert swing_eth == AnalyzeIntent(symbol="ETH/USDT:USDT", timeframe=Timeframe.H4, detail=False)

    scan_scalp = resolve_followup_intent(
        "ada token yang bisa disetup buat scalp gak?",
        recent_messages=history,
    )
    assert scan_scalp == ScanIntent(timeframe=Timeframe.M15)

    scan_intraday = resolve_followup_intent(
        "ada setup intraday gak?",
        recent_messages=history,
    )
    assert scan_intraday == ScanIntent(timeframe=Timeframe.H1)

    # Slash command parsers also accept style aliases ("scalp", "intraday", "swing")
    parsed_scalp = parse_analyze_args(["BTC", "scalp"])
    assert parsed_scalp.symbol == "BTC/USDT:USDT"
    assert parsed_scalp.timeframe is Timeframe.M15

    parsed_intraday = parse_analyze_args(["ETH", "intraday", "detail"])
    assert parsed_intraday.symbol == "ETH/USDT:USDT"
    assert parsed_intraday.timeframe is Timeframe.H1
    assert parsed_intraday.detail is True

    parsed_swing_cmp = parse_compare_args(["BTC", "ETH", "swing"])
    assert parsed_swing_cmp.timeframe is Timeframe.H4
    assert resolve_followup_intent("bandingkan BTC dan SOL buat scalp", recent_messages=()) == (
        CompareIntent(symbols=("BTC/USDT:USDT", "SOL/USDT:USDT"), timeframe=Timeframe.M15)
    )

    # Hard refusals remain untouched
    r_order = resolve_followup_intent("pasang order beli sekarang", recent_messages=history)
    assert isinstance(r_order, RefusalIntent)
    assert r_order.is_hard_refusal is True

    r_5m = resolve_followup_intent("analisa BTC 5m", recent_messages=history)
    assert isinstance(r_5m, RefusalIntent)
    assert r_5m.is_hard_refusal is True

    # Ambiguous lowercase / conceptual query returns soft refusal so Layer 2 LLM can classify
    r_soft = resolve_followup_intent("vvv sekarang gimana?", recent_messages=())
    assert isinstance(r_soft, RefusalIntent)
    assert r_soft.is_hard_refusal is False

    # Casual Indonesian words after cues must NOT be misread as tickers
    for casual_q in ("cek dulu dong", "entry kapan enaknya?"):
        res_empty = resolve_followup_intent(casual_q, recent_messages=())
        assert isinstance(res_empty, RefusalIntent)
        assert res_empty.is_hard_refusal is False

    for conceptual_q in (
        "cek dulu dong",
        "gimana cara atur psikologi pas loss?",
        "bedanya RSI sama MACD apa?",
    ):
        res_ctx = resolve_followup_intent(conceptual_q, recent_messages=history)
        assert isinstance(res_ctx, RefusalIntent)
        assert res_ctx.is_hard_refusal is False


class _LocalMemoryRepo:
    def __init__(self) -> None:
        self.prefs: dict[int, UserPreferences] = {}
        self.msgs: dict[int, list[StoredMessage]] = {}

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
        del record

    def forget_user_data(self, telegram_user_id: int) -> ForgetResult:
        del telegram_user_id
        return ForgetResult(
            deleted_messages=0,
            deleted_conversations=0,
            deleted_preferences=0,
            deleted_interactions=0,
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


class _LocalAnalysisService:
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
        return AnalysisSnapshot(
            snapshot_id=f"snap-{mapping.base}",
            provider="ccxt",
            venue="gate",
            canonical_symbol=mapping.canonical_symbol,
            provider_symbol=mapping.provider_symbol,
            market_type=MarketType.PERPETUAL,
            timeframe=timeframe,
            as_of=ANCHOR,
            input_range=InputRange(
                first_open_time=ANCHOR - timedelta(hours=400),
                last_open_time=ANCHOR - timeframe.duration,
                candle_count=100,
                sha256="c" * 64,
            ),
            calc_version="1.0.0",
            technical=TechnicalMetrics(last_close=25.0),
            trade_setup=TradeSetup(
                status=SetupStatus.WAIT,
                bias=SetupBias.NEUTRAL,
                wait_reason="Menunggu konfirmasi.",
            ),
            facts=(
                Fact(
                    fact_id="F1",
                    label="Close",
                    value=25.0,
                    unit="USDT",
                    display_format="price",
                    source="ccxt:gate",
                    as_of=ANCHOR,
                ),
            ),
            regime=Regime.TRENDING_UP,
            market_structure=MarketStructure.BULLISH,
            events=(
                StructureEvent(
                    event_type=StructureEventType.SWING_LOW,
                    event_time=ANCHOR - timedelta(hours=12),
                    confirmation_time=ANCHOR - timedelta(hours=4),
                    level=23.0,
                ),
            ),
            factors=(
                FactorItem(
                    rule_id="above_ema",
                    description="Close above EMA 200",
                    direction=FactorDirection.SUPPORTS_BULLISH,
                    fact_ids=("F1",),
                ),
            ),
            quality_status=QualityStatus.READY,
        )


class _ScriptedHybridTransport:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

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
        del model, contents, response_json_schema, max_output_tokens, temperature
        self.calls.append(system_instruction)
        text = self._responses.pop(0) if self._responses else "{}"
        return RawGeminiOutput(text=text, model_version="gemini-2.5-flash")


@pytest.mark.asyncio
async def test_hybrid_nlp_layer2_handles_lowercase_coin_scan_and_conversational_qa() -> None:
    narration_json = json.dumps(
        {
            "summary": "VVV berada pada harga {F1}.",
            "supporting_factors": ["Harga berada di atas EMA 200 ({F1})."],
            "contradicting_factors": [],
            "bullish_scenario": "Melanjutkan tren di atas {F1}.",
            "bearish_scenario": "Retracement apabila gagal bertahan di {F1}.",
            "fact_ids": ["F1"],
            "limitations": [],
        }
    )
    transport = _ScriptedHybridTransport(
        [
            # 1) Layer 2 intent classifier for "vvv sekarang gimana?"
            json.dumps({"intent": "analyze", "symbols": ["VVV"], "timeframe": None}),
            # 2) Narration JSON for VVV snapshot
            narration_json,
            # 3) Layer 2 intent classifier for "bedanya CHoCH sama BOS apa?"
            json.dumps({"intent": "chat_qa", "symbols": [], "timeframe": None}),
            # 4) Conversational Q&A JSON answer
            json.dumps(
                {
                    "answer": (
                        "CHoCH (Change of Character) menandai indikasi awal perubahan tren, "
                        "sedangkan BOS (Break of Structure) mengonfirmasi kelanjutan tren."
                    )
                }
            ),
        ]
    )

    narrator = GeminiNarrator(model="gemini-2.5-flash", transport=transport, max_guard_retries=0)
    memory = MemoryService(_LocalMemoryRepo())
    bot = BotService(
        access=AccessController((111,)),
        analysis_service=_LocalAnalysisService(),  # type: ignore[arg-type]
        narrator=narrator,
        memory=memory,
    )

    # 1. Lowercase coin without command -> Layer 2 classifies "analyze VVV"
    vvv_chunks = await bot.handle_text_message(111, "vvv sekarang gimana?")
    vvv_out = "\n".join(vvv_chunks)
    assert "VVV/USDT:USDT (4h)" in vvv_out

    # 2. Scan query -> Layer 1 multi-style ScanIntent without burning LLM classification call
    scan_chunks = await bot.handle_text_message(111, "ada token yang bisa disetup gak?")
    scan_out = "\n".join(scan_chunks)
    assert "Pemindai Setup Futures Multi-Style" in scan_out
    assert "(SCALP (15m) · INTRADAY (1h) · SWING (4h))" in scan_out

    # 3. Conceptual trading question -> Layer 2 classifies "chat_qa" -> Educational Q&A answer
    qa_chunks = await bot.handle_text_message(111, "bedanya CHoCH sama BOS apa?")
    qa_out = "\n".join(qa_chunks)
    assert "Change of Character" in qa_out
    assert "Break of Structure" in qa_out

    # Both the VVV turn and Q&A turn are saved cleanly in memory
    recent = memory.get_recent_messages(111)
    assert any("Change of Character" in m.content for m in recent if m.role == "assistant")


@pytest.mark.asyncio
async def test_casual_chat_flows_to_conversational_ai_even_when_classifier_fails() -> None:
    transport = _ScriptedHybridTransport(
        [
            "not-valid-json",
            json.dumps(
                {
                    "answer": (
                        "Halo! Santai bro, lagi mantau market aja nih. Kalau mau cek koin "
                        "atau lihat ada setup apa, langsung sebut aja ya."
                    )
                }
            ),
        ]
    )
    narrator = GeminiNarrator(model="gemini-2.5-flash", transport=transport, max_guard_retries=0)
    memory = MemoryService(_LocalMemoryRepo())
    bot = BotService(
        access=AccessController((111,)),
        analysis_service=_LocalAnalysisService(),  # type: ignore[arg-type]
        narrator=narrator,
        memory=memory,
    )

    chunks = await bot.handle_text_message(111, "halo bro, lagi santai aja nih")
    reply = "\n".join(chunks)
    assert "Santai bro" in reply
    assert "belum dapat menentukan simbol" not in reply.lower()
