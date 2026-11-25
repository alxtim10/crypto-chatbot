"""Deterministic point-in-time setup replay backtester & bootstrap validator."""

from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import urllib.parse
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

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


class RoundBoundaryMode(StrEnum):
    """Session boundary rule used when flattening a paper book on schedule."""

    NONE = "none"
    FUNDING_8H = "funding_8h"
    DAILY_UTC = "daily_utc"


class RoundDecisionEvent(BaseModel):
    """Plain-English audit record of a bar HOLD, FILL, FEE, EXIT, or boundary FLAT action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(ge=1)
    book: str
    timestamp: UtcDatetime
    symbol: str
    timeframe: Timeframe
    action: Literal["HOLD", "FILL", "FEE", "EXIT", "FLAT"]
    bias: SetupBias | None = None
    price: float | None = None
    fee_r: float | None = None
    net_r: float | None = None
    reason: str


class BacktestConfig(BaseModel):
    """Configuration for deterministic point-in-time trade setup replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    warmup_bars: int = Field(default=200, ge=50)
    max_lookback_bars: int = Field(default=1000, ge=200)
    step_bars: int = Field(default=1, ge=1)
    max_hold_bars: int = Field(default=24, ge=1)
    cooldown_bars: int = Field(default=1, ge=0)
    cost_assumptions: CostAssumptions = CostAssumptions()
    bootstrap_resamples: int = Field(default=1000, ge=100)
    bootstrap_seed: int = 42
    include_long: bool = True
    include_short: bool = True
    book_name: str = "always_on"
    flat_boundary: RoundBoundaryMode = RoundBoundaryMode.NONE
    record_decision_log: bool = True


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
    book_name: str = "always_on"
    flat_boundary: RoundBoundaryMode = RoundBoundaryMode.NONE
    boundary_flats: int = 0
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
    decision_log: tuple[RoundDecisionEvent, ...] = ()
    disclaimer: str


class DualBookComparisonReport(BaseModel):
    """Side-by-side paper replay: always-on book vs. flat-at-boundary book on same candles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    boundary_mode: RoundBoundaryMode
    always_on: BacktestReport
    flat_at_boundary: BacktestReport
    delta_net_expectancy_r: float
    delta_total_net_r: float
    delta_max_drawdown_r: float
    summary_note: str


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


def _is_round_boundary(candle: MarketCandle, boundary: RoundBoundaryMode) -> bool:
    if boundary is RoundBoundaryMode.FUNDING_8H:
        return candle.close_time.hour in (0, 8, 16) and candle.close_time.minute == 0
    if boundary is RoundBoundaryMode.DAILY_UTC:
        return candle.close_time.hour == 0 and candle.close_time.minute == 0
    return False


def _simulate_setup_forward(
    setup: TradeSetup,
    future_candles: Sequence[MarketCandle],
    *,
    start_bar_index: int,
    flat_boundary: RoundBoundaryMode = RoundBoundaryMode.NONE,
) -> tuple[str, datetime, datetime, float, int, int, float, float] | None:
    """Simulate a READY setup forward causally with scaled partial exits (40/40/20).

    Returns `(outcome, fill_time, exit_time, gross_r, exit_bar_idx, bars_held,
    exit_price, fill_price)` if triggered, or `None` if price never touched the entry zone
    within `future_candles`.
    - Entry fill uses the conservative side reached inside `[entry_low, entry_high]`.
    - Gap-through-SL at `candle.open` fills at actual `candle.open` rather than flat `-1.0R`.
    - Partial exits: 40% at TP1 (then trail SL to breakeven `fill_price`), 40% at TP2,
      and 20% at TP3. Never rewinds exit time/bar backward to a past TP bar.
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
    fill_time: datetime | None = None
    fill_price = setup.entry_mid
    trigger_offset = 0
    current_sl = setup.stop_loss
    remaining = 1.0
    realized_r = 0.0
    hit_tp1 = False
    hit_tp2 = False

    for offset, candle in enumerate(future_candles):
        bar_idx = start_bar_index + offset
        just_triggered = False
        if not triggered:
            if candle.low <= setup.entry_high and candle.high >= setup.entry_low:
                triggered = True
                just_triggered = True
                fill_time = candle.close_time
                trigger_offset = offset
                if is_long:
                    if candle.open >= setup.entry_high:
                        fill_price = setup.entry_high
                    else:
                        fill_price = candle.open
                elif candle.open <= setup.entry_low:
                    fill_price = setup.entry_low
                else:
                    fill_price = candle.open
            else:
                continue

        if fill_time is None:
            continue

        gap_sl = (candle.open <= current_sl) if is_long else (candle.open >= current_sl)
        touched_sl = gap_sl or (
            (candle.low <= current_sl) if is_long else (candle.high >= current_sl)
        )
        if touched_sl:
            exit_px = candle.open if gap_sl else current_sl
            ref_entry = (
                setup.entry_mid if (gap_sl and just_triggered and not hit_tp1) else fill_price
            )
            leg_r = (exit_px - ref_entry) / risk if is_long else (ref_entry - exit_px) / risk
            realized_r += remaining * leg_r
            bars_held = max(1, offset - trigger_offset + 1)
            sl_outcome = "hit_tp2" if hit_tp2 else ("hit_tp1" if hit_tp1 else "hit_sl")
            return (
                sl_outcome,
                fill_time,
                candle.close_time,
                realized_r,
                bar_idx,
                bars_held,
                exit_px,
                fill_price,
            )

        if not hit_tp1 and ((candle.high >= setup.tp1) if is_long else (candle.low <= setup.tp1)):
            hit_tp1 = True
            leg_tp1 = (
                (setup.tp1 - fill_price) / risk if is_long else (fill_price - setup.tp1) / risk
            )
            realized_r += 0.40 * leg_tp1
            remaining -= 0.40
            current_sl = fill_price

        if not hit_tp2 and ((candle.high >= setup.tp2) if is_long else (candle.low <= setup.tp2)):
            hit_tp2 = True
            leg_tp2 = (
                (setup.tp2 - fill_price) / risk if is_long else (fill_price - setup.tp2) / risk
            )
            realized_r += 0.40 * leg_tp2
            remaining -= 0.40

        if (candle.high >= setup.tp3) if is_long else (candle.low <= setup.tp3):
            leg_tp3 = (
                (setup.tp3 - fill_price) / risk if is_long else (fill_price - setup.tp3) / risk
            )
            realized_r += remaining * leg_tp3
            bars_held = max(1, offset - trigger_offset + 1)
            return (
                "hit_tp3",
                fill_time,
                candle.close_time,
                realized_r,
                bar_idx,
                bars_held,
                setup.tp3,
                fill_price,
            )

        if flat_boundary is not RoundBoundaryMode.NONE and _is_round_boundary(
            candle, flat_boundary
        ):
            bars_held = max(1, offset - trigger_offset + 1)
            leg_flat = (
                (candle.close - fill_price) / risk
                if is_long
                else (fill_price - candle.close) / risk
            )
            realized_r += remaining * leg_flat
            return (
                "flat_boundary",
                fill_time,
                candle.close_time,
                realized_r,
                bar_idx,
                bars_held,
                candle.close,
                fill_price,
            )

    if not triggered or fill_time is None:
        return None

    last_candle = future_candles[-1]
    last_bar_idx = start_bar_index + len(future_candles) - 1
    bars_held = max(1, (len(future_candles) - 1) - trigger_offset + 1)
    leg_end = (
        (last_candle.close - fill_price) / risk
        if is_long
        else (fill_price - last_candle.close) / risk
    )
    realized_r += remaining * leg_end
    final_outcome = "hit_tp2" if hit_tp2 else ("hit_tp1" if hit_tp1 else "expired")
    return (
        final_outcome,
        fill_time,
        last_candle.close_time,
        realized_r,
        last_bar_idx,
        bars_held,
        last_candle.close,
        fill_price,
    )


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
    for idx_bar in range(1, len(closed)):
        expected_step = closed[idx_bar - 1].timeframe.duration
        if (closed[idx_bar].open_time - closed[idx_bar - 1].open_time) != expected_step:
            raise ValueError(
                "Closed candle series contains time gaps or duplicate timestamps; "
                "contiguous closed candles are required for backtest replay."
            )

    evaluated_bars = 0
    wait_signals = 0
    ready_signals = 0
    untriggered_signals = 0
    boundary_flats = 0
    trades: list[BacktestTrade] = []
    decision_events: list[RoundDecisionEvent] = []

    idx = cfg.warmup_bars - 1
    max_signal_idx = len(closed) - 2
    while idx <= max_signal_idx:
        evaluated_bars += 1
        round_idx = evaluated_bars
        history = closed[max(0, idx + 1 - cfg.max_lookback_bars) : idx + 1]
        bar_candle = history[-1]
        as_of = bar_candle.close_time
        sym = bar_candle.canonical_symbol
        tf = bar_candle.timeframe

        def _append_event(
            action: Literal["HOLD", "FILL", "FEE", "EXIT", "FLAT"],
            *,
            ts: datetime,
            reason: str,
            bias: SetupBias | None = None,
            price: float | None = None,
            fee_r: float | None = None,
            net_r: float | None = None,
            round_index: int = round_idx,
            symbol: str = sym,
            timeframe: Timeframe = tf,
        ) -> None:
            if not cfg.record_decision_log:
                return
            decision_events.append(
                RoundDecisionEvent(
                    round_index=round_index,
                    book=cfg.book_name,
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                    action=action,
                    bias=bias,
                    price=price,
                    fee_r=fee_r,
                    net_r=net_r,
                    reason=reason,
                )
            )

        ind = compute_technical_metrics(history)
        reg = classify_regime(ind.metrics)
        if reg is None:
            wait_signals += 1
            _append_event(
                "HOLD",
                ts=as_of,
                price=bar_candle.close,
                reason="HOLD: insufficient regime classification on closed history.",
            )
            idx += cfg.step_bars
            continue
        struct = analyze_market_structure(history)
        smc = analyze_smc(history, structure=struct)

        setup = build_trade_setup(
            metrics=ind.metrics,
            regime=reg,
            structure=struct,
            smc=smc,
            primary_timeframe=tf,
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
            reason_txt = (
                "HOLD: directional bias disabled by config."
                if bias_disabled and setup.status is SetupStatus.READY
                else f"HOLD: setup WAIT ({setup.wait_reason or 'conditions not met'})."
            )
            _append_event(
                "HOLD",
                ts=as_of,
                bias=setup.bias,
                price=bar_candle.close,
                reason=reason_txt,
            )
            idx += cfg.step_bars
            continue

        ready_signals += 1
        future = closed[idx + 1 : idx + 1 + cfg.max_hold_bars]
        sim = _simulate_setup_forward(
            setup,
            future,
            start_bar_index=idx + 1,
            flat_boundary=cfg.flat_boundary,
        )
        if sim is None:
            untriggered_signals += 1
            _append_event(
                "HOLD",
                ts=as_of,
                bias=setup.bias,
                price=bar_candle.close,
                reason=(
                    f"HOLD: READY {setup.bias.value.upper()} setup expired untriggered "
                    f"within {cfg.max_hold_bars} bars."
                ),
            )
            idx += cfg.step_bars
            continue

        outcome, fill_time, exit_time, gross_r, exit_bar_idx, bars_held, exit_price, fill_price = (
            sim
        )
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
        exec_costs = cfg.cost_assumptions.model_copy(update={"funding_bps_per_24h": 0.0})
        exec_cost_r = (exec_costs.round_trip_cost_rate * fill_price) / risk
        held_hours = bars_held * (tf.duration.total_seconds() / 3600.0)
        funding_rate = (held_hours / 24.0) * (cfg.cost_assumptions.funding_bps_per_24h / 10_000.0)
        funding_r = (funding_rate * fill_price) / risk
        cost_r = exec_cost_r + funding_r
        net_r = gross_r - cost_r

        _append_event(
            "FILL",
            ts=fill_time,
            bias=setup.bias,
            price=round(fill_price, 6),
            reason=(
                f"FILL: {setup.bias.value.upper()} entry zone touched at "
                f"{fill_price:.4f} (SL {setup.stop_loss:.4f}, TP2 {setup.tp2:.4f})."
            ),
        )
        _append_event(
            "FEE",
            ts=fill_time,
            bias=setup.bias,
            price=round(fill_price, 6),
            fee_r=round(cost_r, 4),
            reason=(
                f"FEE: deducted exec+funding cost {cost_r:.4f} R "
                f"({exec_costs.round_trip_cost_rate * 10000:.1f} bps round-trip + "
                f"{held_hours:.1f}h funding)."
            ),
        )
        if outcome == "flat_boundary":
            boundary_flats += 1
            _append_event(
                "FLAT",
                ts=exit_time,
                bias=setup.bias,
                price=round(exit_price, 6),
                net_r=round(net_r, 4),
                reason=(
                    f"FLAT: forced flat at {cfg.flat_boundary.value} boundary "
                    f"(close {exit_price:.4f}, gross {gross_r:+.4f} R, net {net_r:+.4f} R)."
                ),
            )
        else:
            _append_event(
                "EXIT",
                ts=exit_time,
                bias=setup.bias,
                price=round(exit_price, 6),
                net_r=round(net_r, 4),
                reason=(
                    f"EXIT: {setup.bias.value.upper()} closed via {outcome} at "
                    f"{exit_price:.4f} (gross {gross_r:+.4f} R, net {net_r:+.4f} R, "
                    f"{bars_held} bars)."
                ),
            )

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
        book_name=cfg.book_name,
        flat_boundary=cfg.flat_boundary,
        boundary_flats=boundary_flats,
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
        decision_log=tuple(decision_events),
        disclaimer=(
            "Simulasi replay historis deterministik menggunakan candle tertutup & asumsi biaya "
            "eksekusi; hasil masa lalu bukan jaminan performa di masa depan."
        ),
    )


def run_dual_paper_book_backtest(
    candles: Sequence[MarketCandle],
    *,
    config: BacktestConfig | None = None,
    boundary_mode: RoundBoundaryMode = RoundBoundaryMode.DAILY_UTC,
    ltf_candles: Sequence[MarketCandle] | None = None,
    use_available_time: bool = False,
) -> DualBookComparisonReport:
    """Run Always-On vs Flat-at-Boundary paper books over the exact same closed candles."""
    if boundary_mode is RoundBoundaryMode.NONE:
        raise ValueError("boundary_mode must be 'daily_utc' or 'funding_8h' for dual book replay")

    base_cfg = config or BacktestConfig()
    always_cfg = base_cfg.model_copy(
        update={
            "book_name": "always_on",
            "flat_boundary": RoundBoundaryMode.NONE,
        }
    )
    flat_cfg = base_cfg.model_copy(
        update={
            "book_name": f"flat_{boundary_mode.value}",
            "flat_boundary": boundary_mode,
        }
    )

    always_report = run_setup_backtest(
        candles,
        config=always_cfg,
        ltf_candles=ltf_candles,
        use_available_time=use_available_time,
    )
    flat_report = run_setup_backtest(
        candles,
        config=flat_cfg,
        ltf_candles=ltf_candles,
        use_available_time=use_available_time,
    )

    delta_exp = round(flat_report.net_expectancy_r - always_report.net_expectancy_r, 4)
    delta_total = round(flat_report.total_net_r - always_report.total_net_r, 4)
    delta_dd = round(flat_report.max_drawdown_r - always_report.max_drawdown_r, 4)

    exp_desc = (
        f"meningkat {delta_exp:+.4f} R/trade"
        if delta_exp > 0
        else (f"turun {delta_exp:+.4f} R/trade" if delta_exp < 0 else "sama persis")
    )
    dd_desc = (
        f"turun {abs(delta_dd):.2f} R"
        if delta_dd < 0
        else (f"naik {delta_dd:.2f} R" if delta_dd > 0 else "tidak berubah")
    )
    summary_note = (
        f"Book flat di batas {boundary_mode.value} mencatat {flat_report.boundary_flats} "
        f"penutupan paksa; net expectancy {exp_desc} dan max drawdown {dd_desc} "
        "dibanding always-on."
    )

    return DualBookComparisonReport(
        boundary_mode=boundary_mode,
        always_on=always_report,
        flat_at_boundary=flat_report,
        delta_net_expectancy_r=delta_exp,
        delta_total_net_r=delta_total,
        delta_max_drawdown_r=delta_dd,
        summary_note=summary_note,
    )


def render_dual_book_report(report: DualBookComparisonReport) -> str:
    """Render side-by-side Indonesian comparison report for dual paper book replay."""
    a = report.always_on
    b = report.flat_at_boundary
    lines = [
        "=====================================================================",
        "         DUAL PAPER BOOK BACKTEST (ALWAYS-ON VS BOUNDARY FLAT)       ",
        "=====================================================================",
        f"Simbol: {a.symbol} ({a.timeframe.value}) | Batas Flat: {report.boundary_mode.value}",
        (
            f"Periode: {a.start_time.strftime('%Y-%m-%d %H:%M UTC')} s/d "
            f"{a.end_time.strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        "---------------------------------------------------------------------",
        f"Metrik              | Always-On          | Flat ({report.boundary_mode.value})",
        "--------------------+--------------------+---------------------------",
        f"Trades Tereksekusi  | {a.executed_trades:<18} | {b.executed_trades}",
        f"Win Rate            | {a.win_rate * 100.0:>16.2f}% | {b.win_rate * 100.0:.2f}%",
        (
            f"Net Expectancy / Tr | {a.net_expectancy_r:>+16.4f} R | "
            f"{b.net_expectancy_r:+.4f} R (delta {report.delta_net_expectancy_r:+.4f})"
        ),
        (
            f"Total Net Return    | {a.total_net_r:>+16.2f} R | "
            f"{b.total_net_r:+.2f} R (delta {report.delta_total_net_r:+.2f})"
        ),
        (
            f"Max Drawdown        | {a.max_drawdown_r:>16.2f} R | "
            f"{b.max_drawdown_r:.2f} R (delta {report.delta_max_drawdown_r:+.2f})"
        ),
        f"Forced Boundary Flat| {a.boundary_flats:<18} | {b.boundary_flats}",
        "---------------------------------------------------------------------",
        f"Ringkasan: {report.summary_note}",
        f"Decision Log Events: Always-On={len(a.decision_log)}, Flat={len(b.decision_log)}",
        f"Disclaimer: {a.disclaimer}",
        "=====================================================================",
    ]
    return "\n".join(lines)


def write_decision_log_jsonl(events: Sequence[RoundDecisionEvent], path: Path) -> int:
    """Write plain-English decision events (`HOLD`, `FILL`, `FEE`, `EXIT`, `FLAT`) to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [event.model_dump_json() for event in events]
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")
    return len(lines)


def _is_blocked_webhook_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    )


def send_round_events_webhook(
    url: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float = 10.0,
    allow_localhost: bool = False,
    sender: Callable[[str, bytes, dict[str, str], float], int] | None = None,
) -> int:
    """POST structured round events & summary JSON to a validated external webhook."""
    parsed = urllib.parse.urlparse(url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Webhook URL must include a valid hostname.")

    is_explicit_local = host in {"localhost", "127.0.0.1", "::1"}
    if scheme != "https" and not (scheme == "http" and allow_localhost and is_explicit_local):
        raise ValueError(
            "Webhook URL must use https:// (or http://localhost with allow_localhost=True)."
        )
    if is_explicit_local and not allow_localhost:
        raise ValueError("Localhost/loopback webhook targets require allow_localhost=True.")

    if (host.endswith((".local", ".internal")) or host == "metadata.google.internal") and not (
        allow_localhost and is_explicit_local
    ):
        raise ValueError(f"Webhook host '{host}' is blocked by SSRF policy.")

    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if (
        literal_ip is not None
        and _is_blocked_webhook_ip(literal_ip)
        and not (allow_localhost and literal_ip.is_loopback)
    ):
        raise ValueError(f"Webhook target IP '{host}' is blocked by SSRF policy.")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if sender is not None:
        return int(sender(url, body, headers, timeout_seconds))

    target_path = parsed.path or "/"
    if parsed.query:
        target_path = f"{target_path}?{parsed.query}"
    port = parsed.port or (443 if scheme == "https" else 80)

    if literal_ip is None and not (allow_localhost and is_explicit_local):
        try:
            addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError(f"Failed to resolve webhook hostname '{host}': {exc}") from exc
        if not addr_infos:
            raise ValueError(f"Webhook hostname '{host}' returned no DNS addresses.")
        for info in addr_infos:
            raw_ip = str(info[4][0]).split("%", 1)[0]
            resolved_ip = ipaddress.ip_address(raw_ip)
            if _is_blocked_webhook_ip(resolved_ip):
                raise ValueError(
                    f"Webhook hostname '{host}' resolves to blocked IP '{resolved_ip}'."
                )

    conn: http.client.HTTPConnection = (
        http.client.HTTPSConnection(host, port=port, timeout=timeout_seconds)
        if scheme == "https"
        else http.client.HTTPConnection(host, port=port, timeout=timeout_seconds)
    )
    try:
        conn.request("POST", target_path, body=body, headers=headers)
        resp = conn.getresponse()
        resp.read()
        return int(resp.status)
    finally:
        conn.close()


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
