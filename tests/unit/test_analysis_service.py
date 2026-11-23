from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot.analysis.service import (
    AnalysisService,
    build_analysis_snapshot,
)
from crypto_chatbot.bot.templates import (
    format_fact_value,
    format_timestamp_utc_wib,
    render_deterministic_snapshot,
    split_telegram_message,
)
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityFlag,
    QualityStatus,
    Timeframe,
)

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
FIXTURE_SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_4h_sample.json"


class FakeMarketRepository:
    def __init__(self) -> None:
        self._candles: dict[tuple[str, str, MarketType, Timeframe, datetime], MarketCandle] = {}
        self._snapshots: dict[str, AnalysisSnapshot] = {}

    def save_candles(self, candles: Sequence[MarketCandle]) -> int:
        inserted = 0
        for c in candles:
            key = (c.venue, c.canonical_symbol, c.market_type, c.timeframe, c.open_time)
            if key not in self._candles:
                self._candles[key] = c
                inserted += 1
        return inserted

    def load_candles(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType,
        timeframe: Timeframe,
        start_open_time: datetime | None = None,
        end_open_time: datetime | None = None,
        end_close_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[MarketCandle, ...]:
        matched = [
            c
            for (v, sym, mt, tf, _), c in self._candles.items()
            if v == venue
            and sym == canonical_symbol
            and mt is market_type
            and tf is timeframe
            and (start_open_time is None or c.open_time >= start_open_time)
            and (end_open_time is None or c.open_time <= end_open_time)
            and (end_close_time is None or c.close_time <= end_close_time)
        ]
        matched.sort(key=lambda c: c.open_time)
        if limit is not None:
            matched = matched[-limit:]
        return tuple(matched)

    def save_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        self._snapshots.setdefault(snapshot.snapshot_id, snapshot)

    def get_snapshot(self, snapshot_id: str) -> AnalysisSnapshot | None:
        return self._snapshots.get(snapshot_id)


class FakeMarketDataProvider:
    def __init__(self, batches: dict[Timeframe, OhlcvBatch]) -> None:
        self._batches = batches

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        batch = self._batches[timeframe]
        if len(batch.candles) > limit:
            return batch.model_copy(update={"candles": batch.candles[-limit:]})
        return batch

    async def close(self) -> None:
        return None


def _load_fixture_batch(
    path: Path,
    *,
    timeframe: Timeframe = Timeframe.H4,
    limit: int | None = None,
    stale_offset: timedelta = timedelta(seconds=1),
) -> OhlcvBatch:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw_ohlcv"]
    if limit is not None:
        raw = raw[-limit:]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    received = last_open + timeframe.duration + stale_offset
    return normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=timeframe,
        received_time=received,
        now=received,
    )


def _synthetic_batch(
    count: int,
    *,
    timeframe: Timeframe,
    end_close: datetime,
) -> OhlcvBatch:
    start_open = end_close - count * timeframe.duration
    rows = []
    for i in range(count):
        open_t = start_open + i * timeframe.duration
        c_p = 90_000.0 + i * 10.0
        rows.append([open_t.timestamp() * 1000.0, c_p, c_p + 50.0, c_p - 50.0, c_p + 5.0, 100.0])
    return normalize_ohlcv_rows(
        rows,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=timeframe,
        received_time=end_close + timedelta(seconds=5),
        now=end_close + timedelta(seconds=5),
    )


def test_build_snapshot_is_deterministic_and_complete() -> None:
    batch = _load_fixture_batch(FIXTURE_1000)

    snap1 = build_analysis_snapshot(batch)
    snap2 = build_analysis_snapshot(batch)

    assert snap1 == snap2
    assert snap1.quality_status is QualityStatus.READY
    assert snap1.input_range is not None
    assert snap1.input_range.candle_count == 1000
    assert snap1.technical is not None
    assert snap1.regime is not None
    assert snap1.market_structure is not None
    assert len(snap1.facts) > 10
    assert all(factor.is_probabilistic is False for factor in snap1.factors)


@pytest.mark.asyncio
async def test_analysis_service_persists_and_recomputes_identically() -> None:
    primary = _load_fixture_batch(FIXTURE_1000)
    as_of = primary.candles[-1].close_time
    h1_batch = _synthetic_batch(100, timeframe=Timeframe.H1, end_close=as_of)
    d1_batch = _synthetic_batch(
        100,
        timeframe=Timeframe.D1,
        end_close=as_of.replace(hour=0, minute=0, second=0, microsecond=0),
    )

    provider = FakeMarketDataProvider(
        {Timeframe.H4: primary, Timeframe.H1: h1_batch, Timeframe.D1: d1_batch}
    )
    repo = FakeMarketRepository()
    service = AnalysisService(provider=provider, repository=repo)

    original = await service.analyze("BTC", Timeframe.H4, include_mtf=True, persist=True)
    recomputed = service.recompute_from_stored(original.snapshot_id)

    assert original.technical is not None
    assert original.technical.mtf_rsi_alignment is not None
    assert recomputed.input_range == original.input_range
    assert recomputed.facts == original.facts
    assert recomputed.technical == original.technical
    assert recomputed.regime == original.regime
    assert recomputed.market_structure == original.market_structure
    assert recomputed.factors == original.factors
    assert recomputed.snapshot_id == original.snapshot_id


def test_quality_gates_stale_gap_short_and_provider_error() -> None:
    # Stale primary data -> INSUFFICIENT_DATA, no current metrics/facts presented
    stale_batch = _load_fixture_batch(FIXTURE_1000, stale_offset=timedelta(hours=4, minutes=10))
    stale_snap = build_analysis_snapshot(stale_batch)
    assert stale_snap.quality_status is QualityStatus.INSUFFICIENT_DATA
    assert QualityFlag.STALE in stale_snap.quality_flags
    assert stale_snap.technical is None
    assert stale_snap.facts == ()
    stale_text = render_deterministic_snapshot(stale_snap)
    assert "INSUFFICIENT_DATA" in stale_text
    assert "Fakta Terukur" not in stale_text

    # Gap in primary batch -> INSUFFICIENT_DATA
    data = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    gapped_rows = data["raw_ohlcv"][:500] + data["raw_ohlcv"][502:]
    last_open = datetime.fromtimestamp(float(gapped_rows[-1][0]) / 1000.0, tz=UTC)
    gap_batch = normalize_ohlcv_rows(
        gapped_rows,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=last_open + timedelta(hours=4, seconds=5),
    )
    gap_snap = build_analysis_snapshot(gap_batch)
    assert gap_snap.quality_status is QualityStatus.INSUFFICIENT_DATA
    assert QualityFlag.GAP in gap_snap.quality_flags
    assert gap_snap.technical is None
    assert gap_snap.facts == ()

    # Short history < 150 -> INSUFFICIENT_DATA
    short_sample = _load_fixture_batch(FIXTURE_SAMPLE)
    short_snap = build_analysis_snapshot(short_sample)
    assert short_snap.quality_status is QualityStatus.INSUFFICIENT_DATA

    # 200 candles -> DEGRADED, EMA 200 unavailable, valid descriptive snapshot
    degraded_batch = _load_fixture_batch(FIXTURE_1000, limit=200)
    degraded_snap = build_analysis_snapshot(degraded_batch)
    assert degraded_snap.quality_status is QualityStatus.DEGRADED
    assert QualityFlag.SHORT_HISTORY in degraded_snap.quality_flags
    assert degraded_snap.technical is not None
    assert degraded_snap.technical.ema_200 is None
    assert degraded_snap.technical.ema_50 is not None

    # Provider error stays distinct from data-shortage insufficient data
    err_batch = OhlcvBatch(
        metadata=ProviderMetadata(
            provider="ccxt",
            venue="binance",
            market_type=MarketType.SPOT,
            base_unit="BTC",
            quote_currency="USDT",
        ),
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        timeframe=Timeframe.H4,
        received_time=datetime(2026, 11, 1, 17, 0, tzinfo=UTC),
        candles=(),
        quality_flags=(QualityFlag.PROVIDER_ERROR,),
        error_message="Provider timeout fetching BTC/USDT 4h after 3 attempt(s)",
    )
    err_snap = build_analysis_snapshot(err_batch)
    assert err_snap.quality_status is QualityStatus.INSUFFICIENT_DATA
    assert QualityFlag.PROVIDER_ERROR in err_snap.quality_flags
    assert any("Provider timeout" in note for note in err_snap.limitations)


def test_corrupted_mtf_batch_does_not_poison_primary_snapshot() -> None:
    primary = _load_fixture_batch(FIXTURE_1000)
    bad_h1 = OhlcvBatch(
        metadata=primary.metadata,
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        timeframe=Timeframe.H1,
        received_time=primary.received_time,
        candles=(),
        quality_flags=(QualityFlag.PROVIDER_ERROR,),
        error_message="1h failed",
    )
    snap = build_analysis_snapshot(primary, mtf_batches={Timeframe.H1: bad_h1})

    assert snap.quality_status is QualityStatus.READY
    assert snap.technical is not None
    assert snap.technical.mtf_rsi_alignment is None
    assert any("1h" in note for note in snap.limitations)


def test_template_rendering_fact_mapping_utc_wib_escaping_and_splitting() -> None:
    batch = _load_fixture_batch(FIXTURE_1000)
    snap = build_analysis_snapshot(batch)
    rendered = render_deterministic_snapshot(snap)

    # Timestamp appears in UTC and WIB
    assert format_timestamp_utc_wib(snap.as_of) in rendered
    assert "UTC" in rendered
    assert "WIB" in rendered

    # Every snapshot Fact is explicitly rendered alongside its [F<n>] ID
    for fact in snap.facts:
        expected_line_token = f"[{fact.fact_id}] {fact.label}: {format_fact_value(fact)}"
        assert expected_line_token in rendered

    # Every event line references a valid fact ID from snapshot.facts
    known_fids = {fact.fact_id for fact in snap.facts}
    for line in rendered.splitlines():
        if line.strip().startswith("* ["):
            match = re.search(r"\[(F\d+)\]", line)
            assert match is not None
            assert match.group(1) in known_fids

    # Non-probabilistic wording preserved; no probability claim
    assert "Bukan Probabilitas" in rendered
    assert "bukan probabilitas keberhasilan" in rendered.lower()

    # HTML mode escapes hostile limitation text
    hostile_snap = snap.model_copy(
        update={"limitations": (*snap.limitations, "<script>alert(1)</script>")}
    )
    html_text = render_deterministic_snapshot(hostile_snap, html_mode=True)
    assert "<script>" not in html_text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text

    # Message splitting respects Telegram 4096-char ceiling
    long_text = "\n".join(f"Baris {i}: contoh teks analisis." for i in range(400))
    chunks = split_telegram_message(long_text, max_length=4096)
    assert len(chunks) > 1
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert "\n".join(chunks) == long_text
