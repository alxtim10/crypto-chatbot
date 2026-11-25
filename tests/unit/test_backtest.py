from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot.analysis.backtest import (
    BacktestConfig,
    RoundBoundaryMode,
    RoundDecisionEvent,
    render_backtest_report,
    render_dual_book_report,
    run_dual_paper_book_backtest,
    run_setup_backtest,
    send_round_events_webhook,
    write_decision_log_jsonl,
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


def test_dual_paper_book_backtest_and_plain_english_decision_log(tmp_path: Path) -> None:
    candles = _make_synthetic_candles(360)
    cfg = BacktestConfig(warmup_bars=120, max_hold_bars=16, step_bars=1, bootstrap_resamples=100)
    dual = run_dual_paper_book_backtest(
        candles,
        config=cfg,
        boundary_mode=RoundBoundaryMode.DAILY_UTC,
    )

    assert dual.always_on.total_bars == 360
    assert dual.flat_at_boundary.total_bars == 360
    assert dual.always_on.book_name == "always_on"
    assert dual.flat_at_boundary.book_name == "flat_daily_utc"
    assert len(dual.always_on.decision_log) > 0
    assert len(dual.flat_at_boundary.decision_log) > 0

    for ev in dual.flat_at_boundary.decision_log:
        assert ev.reason.startswith(f"{ev.action}:")

    if dual.flat_at_boundary.executed_trades > 0:
        actions = {ev.action for ev in dual.flat_at_boundary.decision_log}
        assert {"FILL", "FEE"}.issubset(actions)
    if dual.flat_at_boundary.boundary_flats > 0:
        assert any(ev.action == "FLAT" for ev in dual.flat_at_boundary.decision_log)

    rendered = render_dual_book_report(dual)
    assert "DUAL PAPER BOOK BACKTEST" in rendered
    assert "Always-On" in rendered
    assert "Flat (daily_utc)" in rendered

    out_path = tmp_path / "logs" / "rounds.jsonl"
    written = write_decision_log_jsonl(dual.flat_at_boundary.decision_log, out_path)
    assert written == len(dual.flat_at_boundary.decision_log)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == written
    first_loaded = RoundDecisionEvent.model_validate_json(lines[0])
    assert first_loaded == dual.flat_at_boundary.decision_log[0]


def test_send_round_events_webhook_validates_scheme_and_posts_payload() -> None:
    with pytest.raises(ValueError, match="valid hostname"):
        send_round_events_webhook("file:///etc/passwd", {"ok": True})
    with pytest.raises(ValueError, match="https://"):
        send_round_events_webhook("file://localhost/etc/passwd", {"ok": True})
    with pytest.raises(ValueError, match="https://"):
        send_round_events_webhook("http://example.com/hook", {"ok": True})

    captured: list[tuple[str, bytes, dict[str, str], float]] = []

    def _fake_sender(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> int:
        captured.append((url, body, headers, timeout))
        return 200

    status_https = send_round_events_webhook(
        "https://hooks.example.com/rounds",
        {"mode": "dual_book", "executed_trades": 3},
        sender=_fake_sender,
    )
    status_local = send_round_events_webhook(
        "http://127.0.0.1:8080/hook",
        {"mode": "single_book"},
        allow_localhost=True,
        sender=_fake_sender,
    )
    assert status_https == 200
    assert status_local == 200
    assert len(captured) == 2
    assert json.loads(captured[0][1].decode("utf-8"))["mode"] == "dual_book"

    for blocked_url in (
        "http://127.0.0.1:8080/hook",
        "https://10.0.0.1/hook",
        "https://169.254.169.254/latest/meta-data/",
        "https://metadata.google.internal/computeMetadata/v1/",
    ):
        with pytest.raises(ValueError, match=r"blocked|allow_localhost"):
            send_round_events_webhook(blocked_url, {"ok": True}, sender=_fake_sender)


def test_simulate_setup_causal_exit_gap_through_sl_and_gap_rejection() -> None:
    from crypto_chatbot.analysis.backtest import _simulate_setup_forward
    from crypto_chatbot.schemas.analysis import SetupBias, SetupStatus, TradeSetup

    setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.LONG,
        entry_low=99.0,
        entry_high=101.0,
        entry_mid=100.0,
        stop_loss=95.0,
        tp1=105.0,
        tp2=110.0,
        tp3=115.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
    )

    def _c(idx: int, o: float, h: float, low_p: float, c: float) -> MarketCandle:
        ot = ANCHOR + idx * timedelta(hours=4)
        ct = ot + timedelta(hours=4)
        return MarketCandle(
            provider="ccxt",
            venue="binance",
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="BINANCE:BTCUSDT.P",
            market_type=MarketType.PERPETUAL,
            timeframe=Timeframe.H4,
            open_time=ot,
            close_time=ct,
            received_time=ct,
            available_time=ct,
            open=o,
            high=h,
            low=low_p,
            close=c,
            base_volume=100.0,
            base_unit="BTC",
            quote_currency="USDT",
            is_closed=True,
        )

    # Bar 0 fills at 101, Bar 1 hits TP2 (110), Bar 2 trails SL to breakeven (101).
    # Exit must happen at Bar 2 (never rewound back to Bar 1!).
    sim = _simulate_setup_forward(
        setup,
        [
            _c(0, 102.0, 103.0, 100.0, 102.0),
            _c(1, 102.0, 111.0, 101.5, 109.0),
            _c(2, 103.0, 104.0, 100.0, 101.0),
        ],
        start_bar_index=10,
    )
    assert sim is not None
    outcome, _, _, _, exit_bar_idx, bars_held, _, _ = sim
    assert outcome == "hit_tp2"
    assert exit_bar_idx == 12
    assert bars_held == 3

    # Gap-through-SL on Bar 1 opening at 90 realizes actual gap loss (< -1.0R)
    sim_gap = _simulate_setup_forward(
        setup,
        [
            _c(0, 100.0, 102.0, 99.5, 100.5),
            _c(1, 90.0, 92.0, 89.0, 91.0),
        ],
        start_bar_index=10,
    )
    assert sim_gap is not None
    gap_outcome, _, _, gap_r, _, _, exit_px, _ = sim_gap
    assert gap_outcome == "hit_sl"
    assert exit_px == 90.0
    assert gap_r == pytest.approx(-2.0)

    # run_setup_backtest rejects candle series with time gaps
    gapped = _make_synthetic_candles(250)
    del gapped[150]
    with pytest.raises(ValueError, match="time gaps"):
        run_setup_backtest(gapped, config=BacktestConfig(warmup_bars=120))
