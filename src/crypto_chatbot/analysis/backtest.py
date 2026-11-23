"""Deterministic point-in-time setup replay backtester & bootstrap validator."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.analysis.indicators import compute_technical_metrics
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.setup import build_trade_setup
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.ml.statistics import (
    SharpeAdjustment,
    adjust_sharpe_for_structure,
    sample_sharpe_ratio,
)
from crypto_chatbot.schemas.analysis import (
    Regime,
    SetupBias,
    SetupStatus,
    TradeSetup,
)
from crypto_chatbot.schemas.market import MarketCandle, Timeframe, UtcDatetime
from crypto_chatbot.schemas.ml import CostAssumptions


class BacktestConfig(BaseModel):
    """Configuration for deterministic point-in-time trade setup replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    warmup_bars: int = Field(default=200, ge=50)
    step_bars: int = Field(default=1, ge=1)
    max_hold_bars: int = Field(default=24, ge=1)
    cooldown_bars: int = Field(default=1, ge=0)
    cost_assumptions: CostAssumptions = CostAssumptions()
    bootstrap_resamples: int = Field(default=1000, ge=100)
    bootstrap_seed: int = 42
    include_long: bool = True
    include_short: bool = True


class BacktestTrade(BaseModel):
    """Single simulated trade executed from a deterministic conditional setup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_time: UtcDatetime
    exit_time: UtcDatetime | None = None
    bias: SetupBias
    regime: Regime
    entry_low: float
    entry_high: float
    entry_mid: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    outcome: str
    gross_r: float
    net_r: float
    bars_held: int
    entry_source: str | None = None


class BootstrapSummary(BaseModel):
    """Bootstrap confidence bounds and drawdown distribution from realized trade R-multiples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resamples: int
    seed: int
    expectancy_r_ci_low: float
    expectancy_r_ci_high: float
    positive_expectancy_resample_rate: float = Field(ge=0.0, le=1.0)
    max_drawdown_r_p95: float = Field(ge=0.0)


class RegimeBreakdown(BaseModel):
    """Performance summary grouped by market trend/volatility regime at signal bar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime: Regime
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_net_r: float
    expectancy_r: float


class BacktestReport(BaseModel):
    """Complete deterministic backtest evaluation report with non-guarantee disclaimer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    source: str
    start_time: UtcDatetime
    end_time: UtcDatetime
    total_bars: int
    evaluated_bars: int
    wait_signals: int
    ready_signals: int
    untriggered_signals: int
    executed_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    gross_expectancy_r: float
    net_expectancy_r: float
    profit_factor: float | None = None
    total_net_r: float
    max_drawdown_r: float
    max_consecutive_losses: int
    sharpe_per_trade: float | None = None
    sharpe_adjustment: SharpeAdjustment | None = None
    bootstrap: BootstrapSummary | None = None
    regime_breakdown: tuple[RegimeBreakdown, ...]
    trades: tuple[BacktestTrade, ...]
    disclaimer: str


def _max_drawdown_r(net_rs: Sequence[float]) -> float:
    if not net_rs:
        return 0.0
    arr = np.asarray(net_rs, dtype=np.float64)
    equity = np.cumsum(arr)
    peaks = np.maximum(0.0, np.maximum.accumulate(equity))
    drawdowns = peaks - equity
    return float(np.max(drawdowns))


def _max_consecutive_losses(net_rs: Sequence[float]) -> int:
    max_streak = 0
    streak = 0
    for val in net_rs:
        if val < 0.0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return max_streak


def _simulate_setup_forward(
    setup: TradeSetup,
    future_candles: Sequence[MarketCandle],
    *,
    start_bar_index: int,
) -> tuple[str, datetime | None, float, int, int] | None:
    """Simulate a READY setup forward against future closed candles.

    Returns `(outcome, exit_time, gross_r, exit_bar_idx, bars_held)` if triggered, or
    `None` if price never touched the entry zone within `future_candles`.
    Conservative tie-break: if any candle triggers/is in position and hits SL on the same bar
    as a TP, SL (`-1.0R`) takes precedence.
    """
    if (
        setup.entry_low is None
        or setup.entry_high is None
        or setup.entry_mid is None
        or setup.stop_loss is None
        or setup.tp1 is None
        or setup.tp2 is None
        or setup.tp3 is None
        or setup.rr_tp1 is None
        or setup.rr_tp2 is None
        or setup.rr_tp3 is None
    ):
        return None

    risk = abs(setup.entry_mid - setup.stop_loss)
    if risk <= 0.0 or not future_candles:
        return None

    is_long = setup.bias is SetupBias.LONG
    triggered = False
    trigger_offset = 0
    highest_tp = 0
    tp_hit_times: dict[int, datetime] = {}
    tp_hit_bars: dict[int, int] = {}

    for offset, candle in enumerate(future_candles):
        bar_idx = start_bar_index + offset
        if not triggered:
            if candle.low <= setup.entry_high and candle.high >= setup.entry_low:
                triggered = True
                trigger_offset = offset
            else:
                continue

        if is_long:
            if candle.low <= setup.stop_loss:
                bars_held = max(1, offset - trigger_offset + 1)
                return "hit_sl", candle.close_time, -1.0, bar_idx, bars_held
            if candle.high >= setup.tp3:
                bars_held = max(1, offset - trigger_offset + 1)
                return "hit_tp3", candle.close_time, setup.rr_tp3, bar_idx, bars_held
            if candle.high >= setup.tp2 and highest_tp < 2:
                highest_tp = 2
                tp_hit_times[2] = candle.close_time
                tp_hit_bars[2] = bar_idx
            if candle.high >= setup.tp1 and highest_tp < 1:
                highest_tp = 1
                tp_hit_times[1] = candle.close_time
                tp_hit_bars[1] = bar_idx
        else:
            if candle.high >= setup.stop_loss:
                bars_held = max(1, offset - trigger_offset + 1)
                return "hit_sl", candle.close_time, -1.0, bar_idx, bars_held
            if candle.low <= setup.tp3:
                bars_held = max(1, offset - trigger_offset + 1)
                return "hit_tp3", candle.close_time, setup.rr_tp3, bar_idx, bars_held
            if candle.low <= setup.tp2 and highest_tp < 2:
                highest_tp = 2
                tp_hit_times[2] = candle.close_time
                tp_hit_bars[2] = bar_idx
            if candle.low <= setup.tp1 and highest_tp < 1:
                highest_tp = 1
                tp_hit_times[1] = candle.close_time
                tp_hit_bars[1] = bar_idx

    if not triggered:
        return None

    if highest_tp == 2:
        exit_bar = tp_hit_bars[2]
        bars_held = max(1, (exit_bar - start_bar_index) - trigger_offset + 1)
        return "hit_tp2", tp_hit_times[2], setup.rr_tp2, exit_bar, bars_held
    if highest_tp == 1:
        exit_bar = tp_hit_bars[1]
        bars_held = max(1, (exit_bar - start_bar_index) - trigger_offset + 1)
        return "hit_tp1", tp_hit_times[1], setup.rr_tp1, exit_bar, bars_held

    last_candle = future_candles[-1]
    last_bar_idx = start_bar_index + len(future_candles) - 1
    bars_held = max(1, (len(future_candles) - 1) - trigger_offset + 1)
    realized_r = (
        (last_candle.close - setup.entry_mid) / risk
        if is_long
        else (setup.entry_mid - last_candle.close) / risk
    )
    return "expired", last_candle.close_time, realized_r, last_bar_idx, bars_held


def run_setup_backtest(
    candles: Sequence[MarketCandle],
    *,
    config: BacktestConfig | None = None,
    ltf_candles: Sequence[MarketCandle] | None = None,
    use_available_time: bool = False,
) -> BacktestReport:
    """Replay `build_trade_setup` bar-by-bar with strict causal slicing and no overlap."""
    cfg = config or BacktestConfig()
    closed = sorted((c for c in candles if c.is_closed), key=lambda c: c.open_time)
    if len(closed) < cfg.warmup_bars + 2:
        raise ValueError(
            f"At least {cfg.warmup_bars + 2} closed candles are required for backtest "
            f"(got {len(closed)})"
        )

    evaluated_bars = 0
    wait_signals = 0
    ready_signals = 0
    untriggered_signals = 0
    trades: list[BacktestTrade] = []

    idx = cfg.warmup_bars - 1
    max_signal_idx = len(closed) - 2
    while idx <= max_signal_idx:
        evaluated_bars += 1
        history = closed[: idx + 1]
        as_of = history[-1].close_time

        ind = compute_technical_metrics(history)
        reg = classify_regime(ind.metrics)
        if reg is None:
            wait_signals += 1
            idx += cfg.step_bars
            continue
        struct = analyze_market_structure(history)
        smc = analyze_smc(history, structure=struct)

        setup = build_trade_setup(
            metrics=ind.metrics,
            regime=reg,
            structure=struct,
            smc=smc,
            primary_timeframe=history[-1].timeframe,
            ltf_candles=ltf_candles,
            as_of=as_of,
            use_available_time=use_available_time,
            cost_assumptions=cfg.cost_assumptions,
        )

        bias_disabled = (setup.bias is SetupBias.LONG and not cfg.include_long) or (
            setup.bias is SetupBias.SHORT and not cfg.include_short
        )
        if setup.status is not SetupStatus.READY or bias_disabled:
            wait_signals += 1
            idx += cfg.step_bars
            continue

        ready_signals += 1
        future = closed[idx + 1 : idx + 1 + cfg.max_hold_bars]
        sim = _simulate_setup_forward(setup, future, start_bar_index=idx + 1)
        if sim is None:
            untriggered_signals += 1
            idx += cfg.step_bars
            continue

        outcome, exit_time, gross_r, exit_bar_idx, bars_held = sim
        if (
            setup.entry_low is None
            or setup.entry_high is None
            or setup.entry_mid is None
            or setup.stop_loss is None
            or setup.tp1 is None
            or setup.tp2 is None
            or setup.tp3 is None
        ):
            idx += cfg.step_bars
            continue

        risk = abs(setup.entry_mid - setup.stop_loss)
        cost_r = (cfg.cost_assumptions.round_trip_cost_rate * setup.entry_mid) / risk
        net_r = gross_r - cost_r

        trades.append(
            BacktestTrade(
                signal_time=as_of,
                exit_time=exit_time,
                bias=setup.bias,
                regime=reg,
                entry_low=setup.entry_low,
                entry_high=setup.entry_high,
                entry_mid=setup.entry_mid,
                stop_loss=setup.stop_loss,
                tp1=setup.tp1,
                tp2=setup.tp2,
                tp3=setup.tp3,
                outcome=outcome,
                gross_r=round(gross_r, 4),
                net_r=round(net_r, 4),
                bars_held=bars_held,
                entry_source=setup.entry_source,
            )
        )
        idx = max(idx + cfg.step_bars, exit_bar_idx + cfg.cooldown_bars)

    executed_trades = len(trades)
    net_rs = [t.net_r for t in trades]
    gross_rs = [t.gross_r for t in trades]

    wins = sum(1 for val in net_rs if val > 0.0)
    losses = sum(1 for val in net_rs if val < 0.0)
    breakeven = executed_trades - wins - losses
    win_rate = (wins / executed_trades) if executed_trades > 0 else 0.0
    gross_exp = float(np.mean(gross_rs)) if gross_rs else 0.0
    net_exp = float(np.mean(net_rs)) if net_rs else 0.0

    pos_sum = sum(val for val in net_rs if val > 0.0)
    neg_sum = sum(abs(val) for val in net_rs if val < 0.0)
    profit_factor = (pos_sum / neg_sum) if neg_sum > 0.0 else None
    total_net_r = float(sum(net_rs))
    max_dd_r = _max_drawdown_r(net_rs)
    max_consec_losses = _max_consecutive_losses(net_rs)

    sharpe_trade: float | None = None
    sharpe_adj: SharpeAdjustment | None = None
    if executed_trades >= 10:
        arr = np.asarray(net_rs, dtype=np.float64)
        sharpe_trade = sample_sharpe_ratio(arr)
        sharpe_adj = adjust_sharpe_for_structure(arr)

    bootstrap: BootstrapSummary | None = None
    if executed_trades >= 5:
        arr = np.asarray(net_rs, dtype=np.float64)
        rng = np.random.default_rng(cfg.bootstrap_seed)
        samples = rng.choice(
            arr,
            size=(cfg.bootstrap_resamples, executed_trades),
            replace=True,
        )
        means = samples.mean(axis=1)
        ci_low, ci_high = np.percentile(means, [5.0, 95.0])
        pos_rate = float(np.mean(means > 0.0))

        eq = np.cumsum(samples, axis=1)
        peaks = np.maximum(0.0, np.maximum.accumulate(eq, axis=1))
        dds = np.max(peaks - eq, axis=1)
        dd_p95 = float(np.percentile(dds, 95.0))

        bootstrap = BootstrapSummary(
            resamples=cfg.bootstrap_resamples,
            seed=cfg.bootstrap_seed,
            expectancy_r_ci_low=float(ci_low),
            expectancy_r_ci_high=float(ci_high),
            positive_expectancy_resample_rate=min(1.0, max(0.0, pos_rate)),
            max_drawdown_r_p95=max(0.0, dd_p95),
        )

    by_regime: dict[Regime, list[BacktestTrade]] = {}
    for t in trades:
        by_regime.setdefault(t.regime, []).append(t)

    regime_breakdown: list[RegimeBreakdown] = []
    for reg_key in Regime:
        reg_trades = by_regime.get(reg_key)
        if not reg_trades:
            continue
        r_count = len(reg_trades)
        r_wins = sum(1 for t in reg_trades if t.net_r > 0.0)
        r_losses = sum(1 for t in reg_trades if t.net_r < 0.0)
        r_total = sum(t.net_r for t in reg_trades)
        regime_breakdown.append(
            RegimeBreakdown(
                regime=reg_key,
                trades=r_count,
                wins=r_wins,
                losses=r_losses,
                win_rate=r_wins / r_count,
                total_net_r=round(r_total, 4),
                expectancy_r=round(r_total / r_count, 4),
            )
        )

    first = closed[0]
    last = closed[-1]
    return BacktestReport(
        symbol=first.canonical_symbol,
        timeframe=first.timeframe,
        source=f"{first.provider}:{first.venue}",
        start_time=closed[cfg.warmup_bars - 1].close_time,
        end_time=last.close_time,
        total_bars=len(closed),
        evaluated_bars=evaluated_bars,
        wait_signals=wait_signals,
        ready_signals=ready_signals,
        untriggered_signals=untriggered_signals,
        executed_trades=executed_trades,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=win_rate,
        gross_expectancy_r=round(gross_exp, 4),
        net_expectancy_r=round(net_exp, 4),
        profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
        total_net_r=round(total_net_r, 4),
        max_drawdown_r=round(max_dd_r, 4),
        max_consecutive_losses=max_consec_losses,
        sharpe_per_trade=round(sharpe_trade, 4) if sharpe_trade is not None else None,
        sharpe_adjustment=sharpe_adj,
        bootstrap=bootstrap,
        regime_breakdown=tuple(regime_breakdown),
        trades=tuple(trades),
        disclaimer=(
            "Simulasi replay historis deterministik menggunakan candle tertutup & asumsi biaya "
            "eksekusi; hasil masa lalu bukan jaminan performa di masa depan."
        ),
    )


def render_backtest_report(report: BacktestReport) -> str:
    """Render a deterministic Indonesian backtest report for CLI output."""
    lines = [
        "=====================================================================",
        "        BACKTEST DETERMINISTIK SETUP FUTURES (REPLAY HISTORIS)       ",
        "=====================================================================",
        f"Simbol: {report.symbol} ({report.timeframe.value}) | Sumber: {report.source}",
        (
            f"Periode Evaluasi: {report.start_time.strftime('%Y-%m-%d %H:%M UTC')} s/d "
            f"{report.end_time.strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        (
            f"Total Candle: {report.total_bars} | Bar Dievaluasi: {report.evaluated_bars} "
            f"(WAIT: {report.wait_signals}, READY: {report.ready_signals}, "
            f"Tidak Tersentuh: {report.untriggered_signals})"
        ),
        "---------------------------------------------------------------------",
        f"Trade Tereksekusi : {report.executed_trades} "
        f"(Win: {report.wins}, Loss: {report.losses}, BE: {report.breakeven})",
        f"Win Rate          : {report.win_rate * 100.0:.2f}%",
        f"Expectancy Gross  : {report.gross_expectancy_r:+.4f} R / trade",
        f"Expectancy Net    : {report.net_expectancy_r:+.4f} R / trade (setelah fee + slippage)",
        f"Total Net Return  : {report.total_net_r:+.2f} R",
        (
            f"Profit Factor     : {report.profit_factor:.2f}"
            if report.profit_factor is not None
            else "Profit Factor     : N/A"
        ),
        f"Max Drawdown      : {report.max_drawdown_r:.2f} R",
        f"Max Loss Beruntun : {report.max_consecutive_losses}x",
    ]

    if report.sharpe_per_trade is not None and report.sharpe_adjustment is not None:
        lines.append(
            f"Sharpe / Trade    : {report.sharpe_per_trade:.3f} "
            f"(Autocorr-Adj: {report.sharpe_adjustment.autocorr_adjusted_sharpe:.3f}, "
            f"Non-Normal-Adj: {report.sharpe_adjustment.non_normal_adjusted_sharpe:.3f})"
        )

    if report.bootstrap is not None:
        bs = report.bootstrap
        lines.extend(
            [
                "---------------------------------------------------------------------",
                f"Validasi Bootstrap ({bs.resamples} resamples, seed={bs.seed}):",
                (
                    f"- Expectancy 90% CI (5th-95th) : "
                    f"[{bs.expectancy_r_ci_low:+.4f} R, {bs.expectancy_r_ci_high:+.4f} R]"
                ),
                (
                    f"- Rasio Resample Positif       : "
                    f"{bs.positive_expectancy_resample_rate * 100.0:.1f}%"
                ),
                f"- Max Drawdown P95             : {bs.max_drawdown_r_p95:.2f} R",
            ]
        )

    if report.regime_breakdown:
        lines.append("---------------------------------------------------------------------")
        lines.append("Breakdown Per Rezim Pasar:")
        for rb in report.regime_breakdown:
            lines.append(
                f"- {rb.regime.value:<16} | Trades: {rb.trades:<3} | "
                f"WR: {rb.win_rate * 100.0:5.1f}% | Net: {rb.total_net_r:+6.2f} R "
                f"(Exp: {rb.expectancy_r:+.3f} R)"
            )

    lines.extend(
        [
            "---------------------------------------------------------------------",
            f"Disclaimer: {report.disclaimer}",
            "=====================================================================",
        ]
    )
    return "\n".join(lines)
