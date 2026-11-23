from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot.analysis.backtest import (
    BacktestConfig,
    render_backtest_report,
    run_setup_backtest,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
ANCHOR = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _make_synthetic_candles(count: int = 320, *, start_price: float = 100.0) -> list[MarketCandle]:
    candles: list[MarketCandle] = []
    price = start_price
    for i in range(count):
        open_t = ANCHOR + i * timedelta(hours=4)
        close_t = open_t + timedelta(hours=4)
        # Steady uptrend with periodic pullbacks to trigger FVG/OB/EMA20 setups cleanly
        cycle = i % 14
        if cycle in (10, 11):
            delta = -0.008
        elif cycle == 12:
            delta = 0.004
        else:
            delta = 0.0055
        open_p = price
        close_p = price * (1.0 + delta)
        high_p = max(open_p, close_p) * 1.003
        low_p = min(open_p, close_p) * 0.997
        price = close_p
        candles.append(
            MarketCandle(
                provider="ccxt",
                venue="binance",
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="BINANCE:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                timeframe=Timeframe.H4,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                base_volume=1000.0 + i * 5.0,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return candles


def test_run_setup_backtest_requires_minimum_closed_candles() -> None:
    short = _make_synthetic_candles(100)
    with pytest.raises(ValueError, match="closed candles are required"):
        run_setup_backtest(short, config=BacktestConfig(warmup_bars=200))


def test_run_setup_backtest_zero_lookahead_and_reproducible_bootstrap() -> None:
    raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    raw_rows = raw["raw_ohlcv"] if isinstance(raw, dict) else raw
    last_ts = max(float(r[0]) for r in raw_rows)
    received = datetime.fromtimestamp(last_ts / 1000.0, tz=UTC) + timedelta(hours=5)
    batch = normalize_ohlcv_rows(
        raw_rows,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )

    cfg = BacktestConfig(
        warmup_bars=200, max_hold_bars=24, bootstrap_resamples=200, bootstrap_seed=7
    )
    report1 = run_setup_backtest(batch.candles, config=cfg)
    report2 = run_setup_backtest(batch.candles, config=cfg)

    assert report1.total_bars == len(batch.candles)
    assert report1.evaluated_bars > 0
    assert report1.wait_signals + report1.ready_signals == report1.evaluated_bars
    assert report1 == report2

    # Mutating future bars in the tail must never alter earlier historical signals
    split_point = 350
    tail_mutated = list(batch.candles)
    for k in range(split_point, len(tail_mutated)):
        c = tail_mutated[k]
        tail_mutated[k] = c.model_copy(
            update={
                "open": c.open * 2.5,
                "high": c.high * 2.5,
                "low": c.low * 2.5,
                "close": c.close * 2.5,
            }
        )
    mutated_report = run_setup_backtest(tail_mutated, config=cfg)
    cutoff_time = batch.candles[split_point - 1].close_time
    early_orig = [t for t in report1.trades if t.signal_time <= cutoff_time]
    early_mut = [t for t in mutated_report.trades if t.signal_time <= cutoff_time]
    for t1, t2 in zip(early_orig, early_mut, strict=False):
        if t1.exit_time is not None and t1.exit_time <= cutoff_time:
            assert t1 == t2


def test_run_setup_backtest_executes_trades_with_cost_deduction_and_no_overlap() -> None:
    candles = _make_synthetic_candles(360)
    cfg = BacktestConfig(warmup_bars=120, max_hold_bars=16, step_bars=1, bootstrap_resamples=150)
    report = run_setup_backtest(candles, config=cfg)

    assert report.evaluated_bars > 0
    assert report.ready_signals >= 0
    for trade in report.trades:
        assert trade.net_r <= trade.gross_r
        if trade.exit_time is not None:
            assert trade.exit_time > trade.signal_time

    # Sequential trades never overlap in time
    for prev, curr in zip(report.trades, report.trades[1:], strict=False):
        if prev.exit_time is not None:
            assert curr.signal_time >= prev.exit_time

    rendered = render_backtest_report(report)
    assert "BACKTEST DETERMINISTIK" in rendered
    assert "Expectancy Net" in rendered
    assert "bukan jaminan" in rendered
