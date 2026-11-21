from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from crypto_chatbot.analysis.service import build_analysis_snapshot
from crypto_chatbot.bot.templates import format_fact_value, render_deterministic_snapshot
from crypto_chatbot.llm.guard import validate_narration
from crypto_chatbot.llm.prompts import build_narration_user_payload
from crypto_chatbot.market_data.derivatives_quality import (
    check_cross_source_price,
    compare_funding_rates,
    normalize_funding_rate_payload,
    normalize_funding_rate_value,
    normalize_liquidation_events,
    normalize_open_interest_payload,
    normalize_open_interest_unit,
)
from crypto_chatbot.market_data.orderbook import (
    OrderBookSynchronizer,
    compute_orderbook_metrics,
    normalize_order_book_payload,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import SymbolError, resolve_mvp_spot_symbol, resolve_symbol
from crypto_chatbot.schemas.derivatives import (
    DerivativesContext,
    FundingRateBatch,
    FundingRateObservation,
    LiquidationBatch,
    LiquidationSide,
    OpenInterestBatch,
    OrderBookBatch,
    OrderBookDiff,
    ReferencePrice,
)
from crypto_chatbot.schemas.llm import LlmAnalysisNarration
from crypto_chatbot.schemas.market import MarketType, QualityFlag, QualityStatus, Timeframe

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
DERIV_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_derivatives_sample.json"
)
NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


def _spot_batch(*, end_close: datetime = NOW, timeframe: Timeframe = Timeframe.H4) -> Any:
    raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"]
    step_ms = int(timeframe.duration.total_seconds() * 1000)
    end_ms = int(end_close.timestamp() * 1000)
    start_ms = end_ms - len(raw) * step_ms
    aligned_rows = [
        [start_ms + idx * step_ms, r[1], r[2], r[3], r[4], r[5]] for idx, r in enumerate(raw)
    ]
    received = end_close + timedelta(seconds=5)
    return normalize_ohlcv_rows(
        aligned_rows,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=timeframe,
        received_time=received,
        now=received,
    )


def _healthy_deriv_context(
    *,
    spot_mid: float = 65_578.0,
    index_price: float = 65_578.0,
    observed_time: datetime = NOW,
) -> DerivativesContext:
    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    spot = resolve_symbol("BTC", market_type=MarketType.SPOT)
    received = observed_time + timedelta(seconds=5)

    funding = normalize_funding_rate_payload(
        {
            "symbol": perp.ccxt_symbol,
            "timestamp": int(observed_time.timestamp() * 1000),
            "fundingTimestamp": int((observed_time + timedelta(hours=8)).timestamp() * 1000),
            "fundingRate": 0.000125,
            "markPrice": index_price + 5.0,
            "indexPrice": index_price,
        },
        symbol=perp,
        received_time=received,
        funding_interval_hours=8.0,
        now=received,
    )
    oi = normalize_open_interest_payload(
        {
            "timestamp": int(observed_time.timestamp() * 1000),
            "openInterestAmount": 82_000.0,
            "openInterestValue": 82_000.0 * index_price,
        },
        [
            {
                "timestamp": int((observed_time - timedelta(hours=4)).timestamp() * 1000),
                "openInterestAmount": 78_000.0,
                "openInterestValue": 78_000.0 * index_price,
            }
        ],
        symbol=perp,
        timeframe=Timeframe.H4,
        received_time=received,
        reference_price=index_price,
    )
    ob = normalize_order_book_payload(
        {
            "timestamp": int(observed_time.timestamp() * 1000),
            "nonce": 1001,
            "bids": [
                [spot_mid - 2.0, 2.0],
                [spot_mid - 20.0, 5.0],
                [spot_mid - 150.0, 10.0],
            ],
            "asks": [
                [spot_mid + 2.0, 1.5],
                [spot_mid + 20.0, 4.0],
                [spot_mid + 150.0, 8.0],
            ],
        },
        symbol=spot,
        received_time=received,
    )
    liq = normalize_liquidation_events(
        (), symbol=perp, received_time=received, source_available=False
    )
    return DerivativesContext(funding=funding, open_interest=oi, order_book=ob, liquidations=liq)


def test_funding_intervals_normalize_and_compare() -> None:
    rate_1h_per_8h, ann_1h = normalize_funding_rate_value(0.0001, 1.0)
    rate_4h_per_8h, ann_4h = normalize_funding_rate_value(0.0004, 4.0)
    rate_8h_per_8h, ann_8h = normalize_funding_rate_value(0.0008, 8.0)

    assert rate_1h_per_8h == pytest.approx(0.0008)
    assert rate_4h_per_8h == pytest.approx(0.0008)
    assert rate_8h_per_8h == pytest.approx(0.0008)
    assert ann_1h == pytest.approx(ann_4h)
    assert ann_4h == pytest.approx(ann_8h)

    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    obs_1h = normalize_funding_rate_payload(
        {"timestamp": int(NOW.timestamp() * 1000), "fundingRate": 0.0001, "interval": "1h"},
        symbol=perp,
        received_time=NOW + timedelta(seconds=1),
        funding_interval_hours=None,
    ).observation
    obs_8h = normalize_funding_rate_payload(
        {"timestamp": int(NOW.timestamp() * 1000), "fundingRate": 0.0004, "interval": "8h"},
        symbol=perp,
        received_time=NOW + timedelta(seconds=1),
        funding_interval_hours=None,
    ).observation
    assert obs_1h is not None
    assert obs_8h is not None
    # 0.0001/1h (= 0.0008/8h) minus 0.0004/8h (= +0.0004/8h)
    assert compare_funding_rates(obs_1h, obs_8h) == pytest.approx(0.0004)

    with pytest.raises(ValueError, match="funding_interval_hours"):
        normalize_funding_rate_value(0.0001, 0.0)


def test_funding_rejects_spot_symbol_and_naive_timestamps() -> None:
    spot = resolve_symbol("BTC", market_type=MarketType.SPOT)
    with pytest.raises(ValueError, match="perpetual"):
        normalize_funding_rate_payload(
            {"fundingRate": 0.0001},
            symbol=spot,
            received_time=NOW,
            funding_interval_hours=8.0,
        )

    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    with pytest.raises(ValidationError):
        FundingRateObservation(
            provider="ccxt",
            venue="binance",
            canonical_symbol=perp.canonical_symbol,
            provider_symbol=perp.provider_symbol,
            observed_time=datetime(2026, 11, 21, 12, 0),  # naive datetime rejected
            received_time=NOW,
            available_time=NOW,
            funding_rate=0.0001,
            funding_interval_hours=8.0,
            rate_per_8h=0.0001,
            annualized_rate=0.0001 * (8760.0 / 8.0),
        )


def test_open_interest_unit_normalization_and_mismatch_detection() -> None:
    # Linear 1 contract = 1 BTC
    base, quote, flags = normalize_open_interest_unit(
        contracts=50_000.0,
        raw_unit="contracts",
        contract_size=1.0,
        quote_value=3_250_000_000.0,
        reference_price=65_000.0,
    )
    assert base == pytest.approx(50_000.0)
    assert quote == pytest.approx(3_250_000_000.0)
    assert flags == ()

    # Linear mini-contract: 1 contract = 0.001 BTC
    base_mini, quote_mini, flags_mini = normalize_open_interest_unit(
        contracts=1_000_000.0,
        raw_unit="contracts",
        contract_size=0.001,
        reference_price=65_000.0,
    )
    assert base_mini == pytest.approx(1_000.0)
    assert quote_mini == pytest.approx(65_000_000.0)
    assert flags_mini == ()

    # Quote-only venue
    base_q, quote_q, flags_q = normalize_open_interest_unit(
        contracts=650_000_000.0,
        raw_unit="quote",
        reference_price=65_000.0,
    )
    assert base_q == pytest.approx(10_000.0)
    assert quote_q == pytest.approx(650_000_000.0)
    assert flags_q == ()

    # Inverse USD-margined contract (e.g. $100 per contract)
    base_inv, quote_inv, flags_inv = normalize_open_interest_unit(
        contracts=10_000.0,
        raw_unit="inverse_contract",
        contract_size=100.0,
        reference_price=50_000.0,
        is_inverse=True,
    )
    assert quote_inv == pytest.approx(1_000_000.0)
    assert base_inv == pytest.approx(20.0)
    assert flags_inv == ()

    # Conflicting unit assumption triggers UNIT_MISMATCH
    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    bad_batch = normalize_open_interest_payload(
        {"openInterestAmount": 50_000_000.0, "openInterestValue": 50_000_000.0},
        [],
        symbol=perp,
        timeframe=Timeframe.H4,
        received_time=NOW,
        contract_size=1.0,
        reference_price=65_000.0,
    )
    assert QualityFlag.UNIT_MISMATCH in bad_batch.quality_flags
    assert QualityFlag.DERIVATIVES_UNAVAILABLE in bad_batch.quality_flags
    assert bad_batch.latest is None


def test_orderbook_depth_bands_imbalance_and_crossed_book_rejection() -> None:
    # Mid price = (99.95 + 100.05) / 2 = 100.0; spread = 0.10 = 10 bps
    # 10 bps band: bid >= 99.90, ask <= 100.10
    # 50 bps band: bid >= 99.50, ask <= 100.50
    bids = [(99.95, 2.0), (99.91, 1.0), (99.60, 5.0), (99.00, 100.0)]
    asks = [(100.05, 1.0), (100.20, 2.0), (101.00, 50.0)]
    _, _, best_bid, best_ask, mid, spread_abs, spread_bps, bands = compute_orderbook_metrics(
        bids, asks
    )
    assert best_bid == pytest.approx(99.95)
    assert best_ask == pytest.approx(100.05)
    assert mid == pytest.approx(100.0)
    assert spread_abs == pytest.approx(0.10)
    assert spread_bps == pytest.approx(10.0)

    b10 = next(b for b in bands if b.band_bps == 10)
    b50 = next(b for b in bands if b.band_bps == 50)
    assert b10.bid_depth_base == pytest.approx(3.0)
    assert b10.bid_depth_quote == pytest.approx(99.95 * 2.0 + 99.91 * 1.0)
    assert b10.ask_depth_base == pytest.approx(1.0)
    assert b10.ask_depth_quote == pytest.approx(100.05)
    assert b10.imbalance == pytest.approx(
        (b10.bid_depth_quote - b10.ask_depth_quote) / (b10.bid_depth_quote + b10.ask_depth_quote)
    )
    # 99.00 and 101.00 are outside +/- 50 bps
    assert b50.bid_depth_base == pytest.approx(8.0)
    assert b50.ask_depth_base == pytest.approx(3.0)

    # Crossed book returns controlled ORDERBOOK_UNAVAILABLE batch
    spot = resolve_symbol("BTC", market_type=MarketType.SPOT)
    crossed = normalize_order_book_payload(
        {"bids": [[101.0, 1.0]], "asks": [[100.0, 1.0]], "nonce": 10},
        symbol=spot,
        received_time=NOW,
    )
    assert crossed.snapshot is None
    assert QualityFlag.ORDERBOOK_UNAVAILABLE in crossed.quality_flags


@pytest.mark.asyncio
async def test_orderbook_sequence_gap_triggers_resync_and_handles_resync_failure() -> None:
    spot = resolve_symbol("BTC", market_type=MarketType.SPOT)
    sync = OrderBookSynchronizer()
    sync.load_snapshot(
        sequence_id=100,
        bids=[(65_000.0, 2.0)],
        asks=[(65_010.0, 2.0)],
        observed_time=NOW,
    )

    # Stale diff (u <= 100) ignored cleanly
    stale_ok = sync.apply_diff(
        OrderBookDiff(
            first_update_id=98,
            final_update_id=100,
            event_time=NOW + timedelta(seconds=1),
            bids=((64_900.0, 5.0),),
        )
    )
    assert stale_ok is True
    assert sync.last_update_id == 100

    # Contiguous futures-style diff (pu == 100) succeeds
    contiguous_ok = sync.apply_diff(
        OrderBookDiff(
            first_update_id=101,
            final_update_id=103,
            prev_final_update_id=100,
            event_time=NOW + timedelta(seconds=2),
            bids=((65_002.0, 3.0),),
        )
    )
    assert contiguous_ok is True
    assert sync.last_update_id == 103
    assert sync.resync_count == 0

    # Sequence gap: spot U > last + 1 (110 > 104) triggers snapshot resync
    resync_calls = 0

    async def _resync_ok() -> dict[str, Any]:
        nonlocal resync_calls
        resync_calls += 1
        return {
            "timestamp": int((NOW + timedelta(seconds=4)).timestamp() * 1000),
            "nonce": 120,
            "bids": [[65_004.0, 4.0], [64_980.0, 6.0]],
            "asks": [[65_008.0, 3.0], [65_030.0, 5.0]],
        }

    batch = await sync.apply_diff_or_resync(
        OrderBookDiff(
            first_update_id=110,
            final_update_id=112,
            event_time=NOW + timedelta(seconds=3),
            bids=((65_005.0, 1.0),),
        ),
        resync_fetch=_resync_ok,
        symbol=spot,
        received_time=NOW + timedelta(seconds=4),
    )
    assert resync_calls == 1
    assert sync.synchronized is True
    assert sync.last_update_id == 120
    assert batch.resync_count == 1
    assert batch.snapshot is not None
    assert batch.snapshot.best_bid == pytest.approx(65_004.0)
    assert QualityFlag.SEQUENCE_GAP in batch.quality_flags
    assert not batch.has_provider_error

    # Futures pu mismatch also triggers resync; if resync fails, emit controlled PROVIDER_ERROR
    async def _resync_fail() -> dict[str, Any]:
        raise TimeoutError("resync snapshot timeout")

    failed_batch = await sync.apply_diff_or_resync(
        OrderBookDiff(
            first_update_id=121,
            final_update_id=125,
            prev_final_update_id=119,  # != 120!
            event_time=NOW + timedelta(seconds=5),
            bids=((65_004.0, 2.0),),
        ),
        resync_fetch=_resync_fail,
        symbol=spot,
        received_time=NOW + timedelta(seconds=6),
        max_resync_attempts=2,
    )
    assert sync.synchronized is False
    assert failed_batch.snapshot is None
    assert failed_batch.has_provider_error is True
    assert QualityFlag.SEQUENCE_GAP in failed_batch.quality_flags
    assert QualityFlag.ORDERBOOK_UNAVAILABLE in failed_batch.quality_flags
    assert failed_batch.resync_count == 3


def test_cross_source_price_mismatch_thresholds_and_snapshot_impact() -> None:
    spot_batch = _spot_batch()

    # 1. Non-comparable timestamp (> 120s gap) does not trigger false mismatch
    non_comp = check_cross_source_price(
        source_a="a",
        price_a=65_000.0,
        time_a=NOW,
        source_b="b",
        price_b=68_000.0,
        time_b=NOW - timedelta(minutes=5),
    )
    assert non_comp.status == "non_comparable_time"

    # 2. Ok (< 50 bps): ~15 bps deviation stays READY
    ok_ctx = _healthy_deriv_context(spot_mid=65_578.0, index_price=65_480.0)
    ok_snap = build_analysis_snapshot(spot_batch, derivatives_context=ok_ctx)
    assert ok_snap.quality_status is QualityStatus.READY
    assert ok_snap.derivatives is not None
    assert ok_snap.derivatives.cross_source_deviation_bps is not None
    assert ok_snap.derivatives.cross_source_deviation_bps < 50.0

    # 3. Moderate mismatch (50..199 bps, ~100 bps): DEGRADED with PRICE_MISMATCH
    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    degrade_ctx = _healthy_deriv_context(spot_mid=65_578.0, index_price=65_578.0)
    degrade_ctx = degrade_ctx.model_copy(
        update={
            "reference_prices": (
                ReferencePrice(
                    provider="ref",
                    venue="other",
                    canonical_symbol=perp.canonical_symbol,
                    market_type=MarketType.PERPETUAL,
                    price=65_578.0 * 1.01,  # 100 bps deviation
                    observed_time=NOW,
                    received_time=NOW + timedelta(seconds=2),
                    available_time=NOW + timedelta(seconds=2),
                ),
            )
        }
    )
    degrade_snap = build_analysis_snapshot(spot_batch, derivatives_context=degrade_ctx)
    assert degrade_snap.quality_status is QualityStatus.DEGRADED
    assert QualityFlag.PRICE_MISMATCH in degrade_snap.quality_flags
    assert QualityFlag.SEVERE_PRICE_MISMATCH not in degrade_snap.quality_flags

    # 4. Severe mismatch (>= 200 bps, ~300 bps): INSUFFICIENT_DATA block
    block_ctx = _healthy_deriv_context(spot_mid=65_578.0, index_price=65_578.0 * 1.03)
    block_snap = build_analysis_snapshot(spot_batch, derivatives_context=block_ctx)
    assert block_snap.quality_status is QualityStatus.INSUFFICIENT_DATA
    assert QualityFlag.PRICE_MISMATCH in block_snap.quality_flags
    assert QualityFlag.SEVERE_PRICE_MISMATCH in block_snap.quality_flags
    assert block_snap.derivatives is None
    assert block_snap.facts == ()


def test_missing_derivatives_never_fabricates_neutral_values() -> None:
    spot_batch = _spot_batch()
    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    spot = resolve_symbol("BTC", market_type=MarketType.SPOT)
    received = NOW + timedelta(seconds=5)

    missing_ctx = DerivativesContext(
        funding=FundingRateBatch(
            canonical_symbol=perp.canonical_symbol,
            provider_symbol=perp.provider_symbol,
            venue="binance",
            received_time=received,
            observation=None,
            quality_flags=(QualityFlag.PROVIDER_ERROR, QualityFlag.DERIVATIVES_UNAVAILABLE),
            error_message="funding down",
        ),
        open_interest=OpenInterestBatch(
            canonical_symbol=perp.canonical_symbol,
            provider_symbol=perp.provider_symbol,
            venue="binance",
            timeframe=Timeframe.H4,
            received_time=received,
            latest=None,
            history=(),
            quality_flags=(QualityFlag.DERIVATIVES_UNAVAILABLE,),
        ),
        order_book=OrderBookBatch(
            canonical_symbol=spot.canonical_symbol,
            provider_symbol=spot.provider_symbol,
            venue="binance",
            market_type=MarketType.SPOT,
            received_time=received,
            snapshot=None,
            quality_flags=(QualityFlag.ORDERBOOK_UNAVAILABLE,),
        ),
        liquidations=LiquidationBatch(
            canonical_symbol=perp.canonical_symbol,
            provider_symbol=perp.provider_symbol,
            venue="binance",
            received_time=received,
            source_available=False,
        ),
    )

    snap = build_analysis_snapshot(spot_batch, derivatives_context=missing_ctx)
    assert snap.quality_status is QualityStatus.DEGRADED
    assert QualityFlag.DERIVATIVES_UNAVAILABLE in snap.quality_flags
    assert QualityFlag.ORDERBOOK_UNAVAILABLE in snap.quality_flags
    assert snap.derivatives is not None
    assert snap.derivatives.funding_rate is None
    assert snap.derivatives.funding_rate_per_8h is None
    assert snap.derivatives.open_interest_base is None
    assert snap.derivatives.open_interest_change_pct is None
    assert snap.derivatives.mid_price is None
    assert snap.derivatives.spread_bps is None
    assert snap.derivatives.imbalance_10bps is None
    assert snap.derivatives.liquidation_source_available is False
    assert snap.derivatives.liquidation_count is None
    assert snap.derivatives.liquidation_long_quote is None
    assert snap.derivatives.liquidation_short_quote is None

    # No derivatives rule emitted and no fabricated zero facts
    rule_ids = {f.rule_id for f in snap.factors}
    assert "funding_rate_8h" not in rule_ids
    assert "open_interest_change" not in rule_ids
    assert "orderbook_liquidity_imbalance" not in rule_ids
    fact_labels = {f.label for f in snap.facts}
    assert "Funding rate ternormalisasi (8h)" not in fact_labels
    assert "Spread bid-ask order book" not in fact_labels
    assert "Likuidasi posisi long" not in fact_labels

    rendered = render_deterministic_snapshot(snap)
    assert "tidak tersedia (nilai tidak direkayasa)" in rendered
    assert "tidak tersedia pada endpoint REST publik" in rendered


def test_liquidation_events_normalization_and_spot_isolation_and_guard() -> None:
    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    raw_deriv = json.loads(DERIV_FIXTURE.read_text(encoding="utf-8"))
    received = datetime.fromisoformat(raw_deriv["received_time"])

    with pytest.raises(SymbolError):
        resolve_symbol("BTC/USDT:USDT", market_type=MarketType.SPOT)

    liq_batch = normalize_liquidation_events(
        raw_deriv["liquidations"],
        symbol=perp,
        received_time=received,
        source_available=True,
    )
    assert liq_batch.source_available is True
    assert len(liq_batch.events) == 2
    assert liq_batch.events[0].side is LiquidationSide.LONG_LIQUIDATED
    assert liq_batch.events[0].quote_notional == pytest.approx(65_500.0 * 1.5)
    assert liq_batch.events[1].side is LiquidationSide.SHORT_LIQUIDATED

    spot_batch = _spot_batch()
    ctx = _healthy_deriv_context().model_copy(update={"liquidations": liq_batch})
    snap = build_analysis_snapshot(spot_batch, derivatives_context=ctx)
    assert snap.quality_status is QualityStatus.READY
    assert snap.derivatives is not None
    assert snap.derivatives.liquidation_source_available is True
    assert snap.derivatives.liquidation_count == 2

    facts_by_label = {f.label: f for f in snap.facts}
    f_funding = facts_by_label["Funding rate ternormalisasi (8h)"]
    f_spread = facts_by_label["Spread bid-ask order book"]
    assert f_funding.source == "binance:BTC/USDT:USDT"
    assert f_spread.source == "binance:BTC/USDT:spot"
    assert format_fact_value(f_funding).endswith("%")
    assert format_fact_value(f_spread).endswith(" bps")

    # LLM guard allows configured 8h / 10 bps / 50 bps and BTC/USDT:USDT ONLY on derivatives snap
    narration = LlmAnalysisNarration(
        summary=(
            f"Funding rate 8h untuk BTC/USDT:USDT berada di {{{f_funding.fact_id}}} "
            f"dengan spread order book 10 bps / 50 bps pada {{{f_spread.fact_id}}}."
        ),
        supporting_factors=("Harga spot BTC/USDT tetap dalam batas referensi.",),
        contradicting_factors=(),
        bullish_scenario="Likuiditas 10 bps tetap terpantau pada satu venue Binance.",
        bearish_scenario="Kondisi masih tidak pasti.",
        limitations=("Data derivatif hanya berasal dari satu venue (Binance).",),
        fact_ids=(f_funding.fact_id, f_spread.fact_id),
    )
    assert validate_narration(narration, snap).is_valid is True

    # Unconfigured bps token (15 bps) and wrong pair (ETH/USDT) still rejected
    bad_narration = narration.model_copy(
        update={"summary": f"Spread pada band 15 bps untuk ETH/USDT di {{{f_spread.fact_id}}}."}
    )
    bad_res = validate_narration(bad_narration, snap)
    assert bad_res.is_valid is False
    assert "raw_digit_detected" in bad_res.reasons
    assert "unsupported_symbol_mentioned" in bad_res.reasons

    payload = build_narration_user_payload(snap)
    assert "derivatives_provenance" in payload
    assert "BTC/USDT:USDT" in payload
