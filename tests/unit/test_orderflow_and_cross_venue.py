"""Unit tests for closed-bar order flow and cross-venue derivatives (Track 1 / ADR 0010)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from crypto_chatbot.analysis.cross_venue import compute_cross_venue_derivatives
from crypto_chatbot.analysis.orderflow import compute_closed_candle_orderflow
from crypto_chatbot.analysis.service import build_analysis_snapshot
from crypto_chatbot.bot.templates import (
    render_compact_snapshot,
    render_deterministic_snapshot,
)
from crypto_chatbot.market_data.bybit_derivatives import BybitDerivativesProvider
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.ml.features import extract_latest_production_features
from crypto_chatbot.schemas.derivatives import (
    CrossVenueDerivativesSummary,
    DerivativesContext,
    FundingRateBatch,
    FundingRateObservation,
    OrderFlowMetrics,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityFlag,
    Timeframe,
)
from crypto_chatbot.schemas.ml import MlExperimentConfig

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"


def _make_candle(
    idx: int,
    *,
    close_price: float = 100.0,
    base_volume: float = 10.0,
    taker_buy_base_volume: float | None = 6.0,
    as_of_base: datetime | None = None,
) -> MarketCandle:
    base_dt = as_of_base or datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    open_time = base_dt + timedelta(hours=idx)
    close_time = open_time + timedelta(hours=1)
    recv_time = close_time + timedelta(seconds=1)
    return MarketCandle(
        provider="binance",
        venue="binance",
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        market_type=MarketType.SPOT,
        timeframe=Timeframe.H1,
        open_time=open_time,
        close_time=close_time,
        received_time=recv_time,
        available_time=recv_time,
        open=close_price,
        high=close_price + 1.0,
        low=close_price - 1.0,
        close=close_price,
        base_volume=base_volume,
        quote_volume=base_volume * close_price,
        taker_buy_base_volume=taker_buy_base_volume,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )


def test_orderflow_none_when_no_taker_volume() -> None:
    candles = [_make_candle(i, taker_buy_base_volume=None) for i in range(25)]
    metrics = compute_closed_candle_orderflow(candles)
    assert metrics is None


def test_orderflow_metrics_calculation() -> None:
    # 25 candles with base_volume=10.0, taker_buy=7.0
    # delta per candle = 2 * 7.0 - 10.0 = +4.0
    # 20-bar sum = 20 * 4.0 = +80.0
    # taker_buy_ratio = (20 * 7.0) / (20 * 10.0) = 0.70
    candles = [_make_candle(i, base_volume=10.0, taker_buy_base_volume=7.0) for i in range(25)]
    metrics = compute_closed_candle_orderflow(candles, window=20)
    assert metrics is not None
    assert metrics.has_order_flow is True
    assert metrics.taker_buy_ratio is not None
    assert pytest.approx(metrics.taker_buy_ratio, abs=1e-5) == 0.70
    assert metrics.cvd_20 is not None
    assert pytest.approx(metrics.cvd_20, abs=1e-5) == 80.0
    assert metrics.cvd_divergence in {"bullish", "bearish", "none"}


def test_orderflow_divergence_detection() -> None:
    candles: list[MarketCandle] = []
    base_dt = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    for i in range(25):
        p = 100.0 - (i * 0.5)  # price dropping
        t_vol = 5.0 + (i * 0.2)  # taker buy rising
        candles.append(
            _make_candle(
                i,
                close_price=p,
                base_volume=10.0,
                taker_buy_base_volume=min(t_vol, 10.0),
                as_of_base=base_dt,
            )
        )
    metrics = compute_closed_candle_orderflow(candles, window=20, divergence_window=10)
    assert metrics is not None
    assert metrics.cvd_divergence == "bullish"


def test_cross_venue_derivatives_calculation() -> None:
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    binance_obs = FundingRateObservation(
        provider="binance",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BINANCE:BTCUSDT.P",
        venue="binance",
        market_type=MarketType.PERPETUAL,
        funding_rate=0.0001,
        rate_per_8h=0.0001,
        annualized_rate=0.0001 * (8760.0 / 8.0),
        funding_interval_hours=8.0,
        observed_time=ref_time,
        received_time=ref_time,
        available_time=ref_time,
        mark_price=50000.0,
        index_price=49990.0,
    )
    bybit_obs = FundingRateObservation(
        provider="bybit",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTCUSDT",
        venue="bybit",
        market_type=MarketType.PERPETUAL,
        funding_rate=0.0003,
        rate_per_8h=0.0003,
        annualized_rate=0.0003 * (8760.0 / 8.0),
        funding_interval_hours=8.0,
        observed_time=ref_time,
        received_time=ref_time,
        available_time=ref_time,
        mark_price=50050.0,
        index_price=49990.0,
    )

    primary_ctx = DerivativesContext(
        funding=FundingRateBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="BINANCE:BTCUSDT.P",
            venue="binance",
            received_time=ref_time,
            observation=binance_obs,
        )
    )
    secondary_ctx = DerivativesContext(
        funding=FundingRateBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="BTCUSDT",
            venue="bybit",
            received_time=ref_time,
            observation=bybit_obs,
        )
    )

    summary = compute_cross_venue_derivatives(primary_ctx, secondary_ctx)
    assert summary is not None
    assert summary.coverage_status == "full"
    assert "binance" in summary.venues
    assert "bybit" in summary.venues
    # funding dispersion: |0.0001 - 0.0003| * 10000 = 2.0 bps
    assert summary.funding_dispersion_bps is not None
    assert pytest.approx(summary.funding_dispersion_bps, abs=0.01) == 2.0
    # mark spread: |50000 - 50050| / 50025 * 10000 ~= 9.995 bps
    assert summary.mark_price_spread_bps is not None
    assert pytest.approx(summary.mark_price_spread_bps, abs=0.1) == 10.0


def test_cross_venue_derivatives_single_venue_fallback() -> None:
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    binance_obs = FundingRateObservation(
        provider="binance",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BINANCE:BTCUSDT.P",
        venue="binance",
        market_type=MarketType.PERPETUAL,
        funding_rate=0.0001,
        rate_per_8h=0.0001,
        annualized_rate=0.0001 * (8760.0 / 8.0),
        funding_interval_hours=8.0,
        observed_time=ref_time,
        received_time=ref_time,
        available_time=ref_time,
        mark_price=50000.0,
        index_price=49990.0,
    )
    primary_ctx = DerivativesContext(
        funding=FundingRateBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="BINANCE:BTCUSDT.P",
            venue="binance",
            received_time=ref_time,
            observation=binance_obs,
        )
    )

    summary = compute_cross_venue_derivatives(primary_ctx, None)
    assert summary is not None
    assert summary.coverage_status == "partial"
    assert summary.funding_dispersion_bps is None
    assert summary.mark_price_spread_bps is None


@pytest.mark.asyncio
async def test_bybit_derivatives_provider_success() -> None:
    mock_client = AsyncMock()
    mock_client.id = "bybit"
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    mock_client.fetch_funding_rate.return_value = {
        "symbol": "BTC/USDT:USDT",
        "fundingRate": 0.00015,
        "markPrice": 50000.0,
        "indexPrice": 49995.0,
        "interval": "8h",
        "timestamp": int(ref_time.timestamp() * 1000),
    }

    provider = BybitDerivativesProvider(
        client=mock_client,
        clock=lambda: ref_time,
    )
    batch = await provider.get_funding_rate("BTC/USDT")
    assert batch.venue == "bybit"
    assert batch.observation is not None
    assert batch.observation.funding_rate == 0.00015
    assert batch.observation.mark_price == 50000.0
    await provider.close()


@pytest.mark.asyncio
async def test_bybit_derivatives_provider_graceful_timeout() -> None:
    mock_client = AsyncMock()
    mock_client.id = "bybit"
    mock_client.fetch_funding_rate.side_effect = TimeoutError("Simulated CCXT timeout")

    provider = BybitDerivativesProvider(client=mock_client)
    batch = await provider.get_funding_rate("BTC/USDT")
    assert batch.venue == "bybit"
    assert batch.observation is None
    assert QualityFlag.DERIVATIVES_UNAVAILABLE in batch.quality_flags
    await provider.close()


def test_orderflow_feature_extraction_causal() -> None:
    candles = [_make_candle(i, base_volume=10.0, taker_buy_base_volume=6.5) for i in range(160)]
    config = MlExperimentConfig(
        canonical_symbol="BTC/USDT",
        timeframe=Timeframe.H1,
    )
    feature_row = extract_latest_production_features(candles, config=config)
    assert "taker_buy_ratio" in feature_row.features
    assert "cvd_zscore_20" in feature_row.features
    assert pytest.approx(feature_row.features["taker_buy_ratio"], abs=1e-4) == 0.65


def test_template_rendering_orderflow_and_cross_venue() -> None:
    data = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    raw = data["raw_ohlcv"]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    received = last_open + Timeframe.H4.duration + timedelta(seconds=1)
    batch = normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )
    base_snap = build_analysis_snapshot(batch)

    of = OrderFlowMetrics(
        taker_buy_ratio=0.65,
        cvd_20=120.5,
        cvd_divergence="bullish",
        has_order_flow=True,
    )
    cv = CrossVenueDerivativesSummary(
        venues=("binance", "bybit"),
        primary_venue="binance",
        funding_dispersion_bps=2.5,
        mark_price_spread_bps=5.0,
        coverage_status="full",
    )

    snapshot = base_snap.model_copy(
        update={
            "order_flow": of,
            "cross_venue_derivatives": cv,
        }
    )

    # Deterministic output contains order flow and multi-venue headers
    det_rendered = render_deterministic_snapshot(snapshot)
    assert "Mikrostruktur Order Flow (Candle Tutup)" in det_rendered
    assert "65,0%" in det_rendered
    assert "Divergensi Bullish" in det_rendered
    assert "Derivatif Lintas Venue (Multi-Venue)" in det_rendered
    assert "BINANCE, BYBIT" in det_rendered

    # Compact output contains compact chips
    comp_rendered = render_compact_snapshot(snapshot)
    assert "Order Flow (Closed):" in comp_rendered
    assert "Taker buy: 65,0%" in comp_rendered
    assert "Div: BULLISH" in comp_rendered
    assert "Lintas Venue (BINANCE/BYBIT):" in comp_rendered

    # Deterministic output contains order flow and multi-venue headers
    det_rendered = render_deterministic_snapshot(snapshot)
    assert "Mikrostruktur Order Flow (Candle Tutup)" in det_rendered
    assert "65,0%" in det_rendered
    assert "Divergensi Bullish" in det_rendered
    assert "Derivatif Lintas Venue (Multi-Venue)" in det_rendered
    assert "BINANCE, BYBIT" in det_rendered

    # Compact output contains compact chips
    comp_rendered = render_compact_snapshot(snapshot)
    assert "Order Flow (Closed):" in comp_rendered
    assert "Taker buy: 65,0%" in comp_rendered
    assert "Div: BULLISH" in comp_rendered
    assert "Lintas Venue (BINANCE/BYBIT):" in comp_rendered
