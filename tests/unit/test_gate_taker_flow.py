from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from crypto_chatbot.analysis.orderflow import compute_closed_candle_orderflow
from crypto_chatbot.market_data.gate_candles import normalize_gate_candlesticks
from crypto_chatbot.market_data.gate_taker_flow import (
    GateTradesClient,
    TradeTick,
    aggregate_gatedata_trades_archives,
    enrich_candles_with_taker_flow,
    parse_rest_trades_payload,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol, resolve_mvp_spot_symbol
from crypto_chatbot.ml.experiment import run_offline_experiment
from crypto_chatbot.ml.features import (
    FEATURE_FAMILIES,
    build_point_in_time_dataset,
    select_available_feature_families,
)
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe
from crypto_chatbot.schemas.ml import FeatureFamily, MlExperimentConfig

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"


def test_verified_sign_convention_link_sample() -> None:
    """Empirically verified sign rule: size > 0 is taker buy, size < 0 is taker sell."""
    bar_open = datetime.fromtimestamp(1795774500, tz=UTC)
    as_of = bar_open + timedelta(minutes=5)
    symbol = resolve_default_perp_symbol("LINK")
    batch = normalize_gate_candlesticks(
        [
            {
                "t": 1795774500,
                "o": "9.558",
                "h": "9.558",
                "l": "9.547",
                "c": "9.556",
                "v": 650,
                "sum": "6210.14",
            }
        ],
        symbol=symbol,
        provider_symbol="GATE:LINKUSDT.P",
        timeframe=Timeframe.M5,
        quanto_multiplier=Decimal("1"),
        received_time=as_of,
    )
    raw_trades = [
        {"id": 1, "create_time": 1795774510.1, "price": "9.558", "size": 539},
        {"id": 2, "create_time": 1795774610.2, "price": "9.556", "size": -111},
    ]
    ticks = parse_rest_trades_payload(raw_trades)
    res = enrich_candles_with_taker_flow(
        batch.candles,
        trades_or_bars=ticks,
        quanto_multiplier=Decimal("1"),
    )
    assert res.bars_matched == 1
    assert res.all_within_tolerance is True
    assert res.max_abs_diff_pct == 0.0
    assert res.candles[0].taker_buy_base_volume == pytest.approx(539.0)


def test_twenty_bar_crosscheck_sum_buy_plus_sell_within_half_percent() -> None:
    symbol = resolve_default_perp_symbol("BTC")
    mult = Decimal("0.0001")
    t0 = datetime(2026, 11, 27, 0, 0, tzinfo=UTC)
    candles: list[MarketCandle] = []
    trades: list[TradeTick] = []

    for i in range(20):
        open_t = t0 + timedelta(hours=i)
        close_t = open_t + timedelta(hours=1)
        buy_contracts = 20_000 + i * 500
        sell_contracts = 15_000 + (19 - i) * 300
        total_contracts = buy_contracts + sell_contracts
        candles.append(
            MarketCandle(
                provider="gate_rest",
                venue="gate",
                canonical_symbol=symbol.canonical_symbol,
                provider_symbol="GATE:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                timeframe=Timeframe.H1,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=60000.0 + i,
                high=60100.0 + i,
                low=59900.0 + i,
                close=60050.0 + i,
                raw_contracts=float(total_contracts),
                base_volume=float(Decimal(total_contracts) * mult),
                quote_volume=float(Decimal(total_contracts) * mult) * (60050.0 + i),
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
        trades.append(
            TradeTick(
                timestamp=open_t + timedelta(minutes=10),
                price=60010.0 + i,
                signed_contracts=float(buy_contracts),
                trade_id=i * 2 + 1,
            )
        )
        trades.append(
            TradeTick(
                timestamp=open_t + timedelta(minutes=30),
                price=60040.0 + i,
                signed_contracts=-float(sell_contracts),
                trade_id=i * 2 + 2,
            )
        )

    result = enrich_candles_with_taker_flow(
        candles,
        trades_or_bars=trades,
        quanto_multiplier=mult,
        max_deviation_pct=0.5,
    )
    assert result.bars_matched == 20
    assert result.bars_missing_trades == 0
    assert result.all_within_tolerance is True
    assert result.max_abs_diff_pct <= 0.5
    assert all(
        c.taker_buy_base_volume is not None and c.taker_buy_base_volume > 0.0
        for c in result.candles
    )

    flow = compute_closed_candle_orderflow(result.candles)
    assert flow is not None
    assert flow.has_order_flow is True
    assert 0.0 < flow.taker_buy_ratio < 1.0


def test_aggregate_gatedata_trades_archive_file(tmp_path: Path) -> None:
    gz_path = tmp_path / "BTC_USDT-202610.csv.gz"
    csv_body = (
        "1790812800.013589,844545515,83578.7,1\n"
        "1790812800.013603,844545516,83578.7,31\n"
        "1790812800.577279,844545519,83578.6,-3\n"
    )
    gz_path.write_bytes(gzip.compress(csv_body.encode()))
    stats = aggregate_gatedata_trades_archives(
        [gz_path],
        timeframe=Timeframe.H1,
        quanto_multiplier=Decimal("0.0001"),
    )
    bar_t = datetime.fromtimestamp(1790812800, tz=UTC)
    assert bar_t in stats
    assert stats[bar_t].taker_buy_contracts == 32.0
    assert stats[bar_t].taker_sell_contracts == 3.0
    assert stats[bar_t].taker_buy_base_volume == pytest.approx(0.0032)
    assert stats[bar_t].taker_sell_base_volume == pytest.approx(0.0003)


def test_orderflow_ml_ablation_keeps_all_features_when_taker_buy_populated() -> None:
    raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"]
    spot = resolve_mvp_spot_symbol("BTC")
    perp = resolve_default_perp_symbol("BTC")
    base_batch = normalize_ohlcv_rows(
        raw[:400],
        symbol=spot,
        timeframe=Timeframe.H4,
        received_time=datetime(2026, 11, 27, 0, 0, tzinfo=UTC),
        now=datetime(2026, 11, 27, 0, 0, tzinfo=UTC),
    )

    mult = Decimal("0.0001")
    trades: list[TradeTick] = []
    perp_candles: list[MarketCandle] = []
    for idx, c in enumerate(base_batch.candles):
        total_contracts = round(c.base_volume / 0.0001)
        buy_frac = 0.45 + 0.15 * ((idx % 5) / 4.0)
        buy_contracts = max(1, round(total_contracts * buy_frac))
        sell_contracts = max(1, total_contracts - buy_contracts)
        perp_candles.append(
            c.model_copy(
                update={
                    "provider": "gate_rest",
                    "venue": "gate",
                    "canonical_symbol": perp.canonical_symbol,
                    "provider_symbol": "GATE:BTCUSDT.P",
                    "market_type": MarketType.PERPETUAL,
                    "raw_contracts": float(buy_contracts + sell_contracts),
                    "base_volume": float(Decimal(buy_contracts + sell_contracts) * mult),
                }
            )
        )
        trades.append(
            TradeTick(
                timestamp=c.open_time + timedelta(minutes=15),
                price=c.close,
                signed_contracts=float(buy_contracts),
            )
        )
        trades.append(
            TradeTick(
                timestamp=c.open_time + timedelta(minutes=45),
                price=c.close,
                signed_contracts=-float(sell_contracts),
            )
        )

    enrichment = enrich_candles_with_taker_flow(
        perp_candles,
        trades_or_bars=trades,
        quanto_multiplier=mult,
    )
    assert enrichment.all_within_tolerance
    assert all(
        c.taker_buy_base_volume is not None and c.taker_buy_base_volume > 0.0
        for c in enrichment.candles
    )

    cfg = MlExperimentConfig(canonical_symbol=perp.canonical_symbol, timeframe=Timeframe.H4)
    rows, _ = build_point_in_time_dataset(enrichment.candles, config=cfg)
    selected, excluded = select_available_feature_families(
        rows,
        (FeatureFamily.TECHNICAL, FeatureFamily.STRUCTURE_PIVOT, FeatureFamily.ORDERFLOW),
    )
    assert excluded == ()
    for feat in FEATURE_FAMILIES[FeatureFamily.ORDERFLOW]:
        assert feat in selected

    report = run_offline_experiment(enrichment.candles, config=cfg)
    ablation_names = [a.family_name for a in report.ablations]
    assert "technical+structure_pivot+orderflow" in ablation_names


@pytest.mark.asyncio
async def test_gate_trades_client_fetches_paginated_window() -> None:
    t0 = datetime.fromtimestamp(1795774500, tz=UTC)
    t1 = t0 + timedelta(minutes=5)

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        if offset == 0:
            return httpx.Response(
                200,
                json=[
                    {"id": 1, "create_time": 1795774510.0, "price": "9.55", "size": 10},
                    {"id": 2, "create_time": 1795774520.0, "price": "9.54", "size": -4},
                ],
            )
        return httpx.Response(200, json=[])

    client = GateTradesClient(transport=httpx.MockTransport(handler))
    try:
        ticks = await client.fetch_trades_window("LINK_USDT", start_time=t0, end_time=t1)
    finally:
        await client.close()
    assert len(ticks) == 2
    assert ticks[0].is_taker_buy is True
    assert ticks[1].is_taker_buy is False
