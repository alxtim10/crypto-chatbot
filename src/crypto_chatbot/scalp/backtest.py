"""SC-10 & SC-11: Event-driven 1m scalp backtest engine and pre-registered validation report.

Uses the exact shared code paths as live:
- `select_scalp_mode_from_candles` (§3)
- `evaluate_trend_pullback_setup` / `evaluate_range_fade_setup` (§4)
- `evaluate_scalp_filters` (§6)
- `settle_scalp_trade_1m` (§9.2, with `human_delay_sec=90`)

Validation protocol (§9.3) & kill criteria (§9.4):
- Contiguity check on 1m bars
- Walk-forward monthly folds + final 3-month untouched holdout
- Breakdown per mode, per symbol, per hour-of-day (UTC), per weekday
- Block-bootstrap 95% CI (by day), profit factor, max DD (R)
- Cost sensitivity at 1.0x / 1.5x / 2.0x + break-even cost multiplier
- DSR with registered trial count
- Random-entry benchmark with 95% CI of (strategy - random) difference
- Markdown report generator (`docs/research/scalp_v1-backtest.md`)
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from crypto_chatbot.analysis.bootstrap import stationary_block_bootstrap_ci
from crypto_chatbot.analysis.indicators import compute_technical_metrics
from crypto_chatbot.market_data.resampling import resample_candles
from crypto_chatbot.ml.statistics import deflated_sharpe_ratio, sample_sharpe_ratio
from crypto_chatbot.scalp.config import MacroEventWindow, ScalpConfig
from crypto_chatbot.scalp.filters import ScalpFilterInputs, evaluate_scalp_filters
from crypto_chatbot.scalp.mode import ScalpMode, select_scalp_mode_from_candles
from crypto_chatbot.scalp.settlement import (
    ScalpOutcomeStatus,
    ScalpSettlementOutcome,
    ScalpTradeSpec,
    settle_scalp_trade_1m,
)
from crypto_chatbot.scalp.setups import (
    detect_valid_dealing_range,
    evaluate_range_fade_setup,
    evaluate_trend_pullback_setup,
)
from crypto_chatbot.schemas.market import MarketCandle, Timeframe


class ContiguityReport(BaseModel):
    """1m candle contiguity audit report (SC-10)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    total_bars: int
    start_time_utc: datetime | None = None
    end_time_utc: datetime | None = None
    gap_count: int = 0
    missing_minutes: int = 0
    duplicate_count: int = 0
    out_of_order_count: int = 0
    is_contiguous: bool = True


def check_1m_contiguity(
    candles_1m: Sequence[MarketCandle],
    *,
    symbol: str = "UNKNOWN",
) -> ContiguityReport:
    """Audit 1m candle sequence for gaps, duplicates, and ordering issues."""
    if not candles_1m:
        return ContiguityReport(symbol=symbol, total_bars=0, is_contiguous=True)

    gap_count = 0
    missing_minutes = 0
    duplicate_count = 0
    out_of_order_count = 0
    prev_open = candles_1m[0].open_time.astimezone(UTC)

    for candle in candles_1m[1:]:
        cur_open = candle.open_time.astimezone(UTC)
        delta_sec = int((cur_open - prev_open).total_seconds())
        if delta_sec == 60:
            pass
        elif delta_sec == 0:
            duplicate_count += 1
        elif delta_sec < 0:
            out_of_order_count += 1
        else:
            gap_count += 1
            missing_minutes += max(0, (delta_sec // 60) - 1)
        prev_open = cur_open

    return ContiguityReport(
        symbol=symbol,
        total_bars=len(candles_1m),
        start_time_utc=candles_1m[0].open_time.astimezone(UTC),
        end_time_utc=candles_1m[-1].close_time.astimezone(UTC),
        gap_count=gap_count,
        missing_minutes=missing_minutes,
        duplicate_count=duplicate_count,
        out_of_order_count=out_of_order_count,
        is_contiguous=(gap_count == 0 and duplicate_count == 0 and out_of_order_count == 0),
    )


class ScalpBacktestTrade(BaseModel):
    """Single simulated scalp trade from the event-driven 1m backtest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    mode: ScalpMode
    direction: Literal["long", "short"]
    signal_time_utc: datetime
    entry_limit: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_bps: float
    expected_cost_bps: float
    outcome: ScalpSettlementOutcome


class ScalpBucketStats(BaseModel):
    """Summary statistics for a slice (overall, mode, symbol, hour, weekday, or fold)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    signals: int
    filled_trades: int
    fill_rate: float
    win_rate: float
    expectancy_gross_r: float
    expectancy_net_r: float
    ci_low_r: float | None = None
    ci_high_r: float | None = None
    profit_factor: float | None = None
    max_dd_r: float


class CostSweepPoint(BaseModel):
    """Performance at a given cost multiplier (1.0x / 1.5x / 2.0x)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    multiplier: float
    filled_trades: int
    expectancy_net_r: float
    ci_low_r: float | None = None
    ci_high_r: float | None = None
    win_rate: float
    profit_factor: float | None = None


class RandomBenchmarkComparison(BaseModel):
    """Comparison against random-entry benchmark (§9.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    strategy_expectancy_r: float
    random_expectancy_r: float
    diff_expectancy_r: float
    diff_ci_low_r: float | None = None
    diff_ci_high_r: float | None = None
    beats_random: bool


class ScalpBacktestResult(BaseModel):
    """Full output of `run_scalp_backtest_1m`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = "scalp_v1"
    seed: int
    cost_multiplier: float
    contiguity: tuple[ContiguityReport, ...]
    trades: tuple[ScalpBacktestTrade, ...]
    decisions_evaluated: int
    filters_blocked_count: int


class ScalpValidationReport(BaseModel):
    """Pre-registered validation report (§9.3 & §9.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_id: str
    registered_trials_count: int
    params_sha256: str
    seed: int
    window_start_utc: datetime | None = None
    window_end_utc: datetime | None = None
    overall: ScalpBucketStats
    holdout: ScalpBucketStats
    walk_forward_folds: tuple[ScalpBucketStats, ...] = ()
    by_mode: tuple[ScalpBucketStats, ...] = ()
    by_symbol: tuple[ScalpBucketStats, ...] = ()
    by_hour_utc: tuple[ScalpBucketStats, ...] = ()
    by_weekday: tuple[ScalpBucketStats, ...] = ()
    cost_sweep: tuple[CostSweepPoint, ...] = ()
    breakeven_cost_multiplier: float | None = None
    dsr_probability: float | None = None
    sample_sharpe_daily: float | None = None
    random_benchmark: RandomBenchmarkComparison
    passes_shadow_gate: bool
    kill_flags: tuple[str, ...] = ()


def _compute_max_dd_r(net_rs: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in net_rs:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_profit_factor(net_rs: Sequence[float]) -> float | None:
    gains = sum(r for r in net_rs if r > 0.0)
    losses = abs(sum(r for r in net_rs if r < 0.0))
    if losses <= 1e-12:
        return None if gains <= 0.0 else float("inf")
    return gains / losses


def _daily_r_series(trades: Sequence[ScalpBacktestTrade]) -> tuple[list[str], list[float]]:
    by_day: dict[str, float] = defaultdict(float)
    for t in trades:
        if not t.outcome.filled:
            continue
        day_key = t.signal_time_utc.astimezone(UTC).strftime("%Y-%m-%d")
        by_day[day_key] += t.outcome.net_r
    days = sorted(by_day.keys())
    return days, [by_day[d] for d in days]


def summarize_trades_bucket(
    label: str,
    trades: Sequence[ScalpBacktestTrade],
    *,
    seed: int = 42,
    bootstrap_resamples: int = 500,
) -> ScalpBucketStats:
    """Compute required §9.3 stats with daily block-bootstrap 95% CI."""
    signals = len(trades)
    filled = [t for t in trades if t.outcome.filled]
    n_filled = len(filled)
    fill_rate = (n_filled / signals) if signals > 0 else 0.0

    if n_filled == 0:
        return ScalpBucketStats(
            label=label,
            signals=signals,
            filled_trades=0,
            fill_rate=fill_rate,
            win_rate=0.0,
            expectancy_gross_r=0.0,
            expectancy_net_r=0.0,
            ci_low_r=None,
            ci_high_r=None,
            profit_factor=None,
            max_dd_r=0.0,
        )

    net_rs = [t.outcome.net_r for t in filled]
    gross_rs = [t.outcome.gross_r for t in filled]
    win_rate = sum(1 for r in net_rs if r > 0.0) / n_filled
    exp_net = float(np.mean(net_rs))
    exp_gross = float(np.mean(gross_rs))
    pf = _compute_profit_factor(net_rs)
    if pf is not None and not np.isfinite(pf):
        pf = 999.0
    max_dd = _compute_max_dd_r(net_rs)

    # Block bootstrap by day (per §9.3)
    _, daily_rs = _daily_r_series(filled)
    ci_low: float | None = None
    ci_high: float | None = None
    if len(daily_rs) >= 2:
        # Scale daily mean back to per-trade expectancy via trades/day
        trades_per_day = n_filled / len(daily_rs)
        d_lo, d_hi, _ = stationary_block_bootstrap_ci(
            daily_rs,
            resamples=bootstrap_resamples,
            mean_block_length=min(5.0, max(1.0, len(daily_rs) / 3.0)),
            seed=seed,
        )
        ci_low = d_lo / max(1e-9, trades_per_day)
        ci_high = d_hi / max(1e-9, trades_per_day)
    elif len(net_rs) >= 2:
        ci_low, ci_high, _ = stationary_block_bootstrap_ci(
            net_rs,
            resamples=bootstrap_resamples,
            mean_block_length=1.0,
            seed=seed,
        )

    return ScalpBucketStats(
        label=label,
        signals=signals,
        filled_trades=n_filled,
        fill_rate=fill_rate,
        win_rate=win_rate,
        expectancy_gross_r=exp_gross,
        expectancy_net_r=exp_net,
        ci_low_r=ci_low,
        ci_high_r=ci_high,
        profit_factor=pf,
        max_dd_r=max_dd,
    )


def run_scalp_backtest_1m(
    candles_1m_by_symbol: Mapping[str, Sequence[MarketCandle]],
    *,
    config: ScalpConfig | None = None,
    cost_multiplier: float = 1.0,
    seed: int = 42,
    macro_events: Sequence[MacroEventWindow] = (),
    rolling_spread_bps_by_symbol: Mapping[str, float] | None = None,
    tick_size_by_symbol: Mapping[str, float] | None = None,
    warmup_bars_1h: int = 60,
    step_5m_bars: int = 1,
) -> ScalpBacktestResult:
    """Event-driven 1m scalp backtest using the exact live decision and 1m settlement paths."""
    cfg = config or ScalpConfig()
    spread_map = rolling_spread_bps_by_symbol or {}
    tick_map = tick_size_by_symbol or {}

    contiguity_reports: list[ContiguityReport] = []
    all_trades: list[ScalpBacktestTrade] = []
    decisions_evaluated = 0
    filters_blocked = 0

    for symbol, raw_1m in sorted(candles_1m_by_symbol.items()):
        bars_1m = sorted((c for c in raw_1m if c.is_closed), key=lambda c: c.open_time)
        contiguity_reports.append(check_1m_contiguity(bars_1m, symbol=symbol))
        if len(bars_1m) < max(120, warmup_bars_1h * 60):
            continue

        bars_5m = resample_candles(bars_1m, target_timeframe=Timeframe.M5).candles
        bars_15m = resample_candles(bars_1m, target_timeframe=Timeframe.M15).candles
        bars_1h = resample_candles(bars_1m, target_timeframe=Timeframe.H1).candles
        if len(bars_1h) < warmup_bars_1h or len(bars_15m) < 25 or len(bars_5m) < 20:
            continue

        # Index 1m bars by open_time for fast forward slicing
        open_times_1m = [c.open_time.astimezone(UTC) for c in bars_1m]
        close_times_15m = [c.close_time.astimezone(UTC) for c in bars_15m]
        close_times_1h = [c.close_time.astimezone(UTC) for c in bars_1h]

        idx_1h = 0
        idx_15m = 0
        idx_1m = 0
        last_sl_time: datetime | None = None
        daily_realized_r: dict[str, float] = defaultdict(float)
        busy_until: datetime | None = None

        start_5m_idx = max(20, warmup_bars_1h * 12)
        for i5 in range(start_5m_idx, len(bars_5m), max(1, step_5m_bars)):
            decision_time = bars_5m[i5].close_time.astimezone(UTC)
            if busy_until is not None and decision_time < busy_until:
                continue

            while idx_1h < len(close_times_1h) and close_times_1h[idx_1h] <= decision_time:
                idx_1h += 1
            while idx_15m < len(close_times_15m) and close_times_15m[idx_15m] <= decision_time:
                idx_15m += 1
            while idx_1m < len(open_times_1m) and open_times_1m[idx_1m] < decision_time:
                idx_1m += 1

            if idx_1h < warmup_bars_1h or idx_15m < 25:
                continue

            decisions_evaluated += 1
            win_1h = bars_1h[max(0, idx_1h - 250) : idx_1h]
            win_15m = bars_15m[max(0, idx_15m - 120) : idx_15m]
            win_5m = bars_5m[max(0, i5 - 60) : i5 + 1]

            tech_1h = compute_technical_metrics(win_1h).metrics
            est_rt_bps = (
                cfg.costs.maker_fee_bps
                + cfg.costs.taker_fee_bps
                + spread_map.get(symbol, cfg.costs.fallback_spread_bps) / 2.0
                + cfg.costs.fallback_taker_slippage_bps
            ) * cost_multiplier
            dealing_range = detect_valid_dealing_range(
                candles_1h=win_1h,
                candles_15m=win_15m,
                atr_1h=tech_1h.atr_14,
                expected_round_trip_cost_bps=est_rt_bps,
                config=cfg,
            )

            mode_res = select_scalp_mode_from_candles(
                win_1h,
                win_15m,
                as_of=decision_time,
                has_valid_range=dealing_range is not None,
                range_low=dealing_range.range_low if dealing_range else None,
                range_high=dealing_range.range_high if dealing_range else None,
                config=cfg,
            )
            if mode_res.mode is ScalpMode.NO_TRADE:
                continue

            # Evaluate setup
            if mode_res.mode in (ScalpMode.TREND_PULLBACK_LONG, ScalpMode.TREND_PULLBACK_SHORT):
                setup = evaluate_trend_pullback_setup(
                    mode=mode_res.mode,
                    canonical_symbol=symbol,
                    candles_15m=win_15m,
                    candles_5m=win_5m,
                    decision_time_utc=decision_time,
                    config=cfg,
                    rolling_median_spread_bps=spread_map.get(symbol),
                    cost_multiplier=cost_multiplier,
                )
            elif mode_res.mode is ScalpMode.RANGE_FADE and dealing_range is not None:
                setup = evaluate_range_fade_setup(
                    canonical_symbol=symbol,
                    dealing_range=dealing_range,
                    candles_5m=win_5m,
                    decision_time_utc=decision_time,
                    config=cfg,
                    rolling_median_spread_bps=spread_map.get(symbol),
                    cost_multiplier=cost_multiplier,
                )
            else:
                continue

            if setup.status != "READY" or setup.direction is None or setup.cost_breakdown is None:
                continue

            # Evaluate §6 blocking filters
            day_key = decision_time.strftime("%Y-%m-%d")
            flt = evaluate_scalp_filters(
                ScalpFilterInputs(
                    decision_time_utc=decision_time,
                    canonical_symbol=symbol,
                    direction=setup.direction,
                    macro_events=tuple(macro_events),
                    realized_r_today_utc=daily_realized_r[day_key],
                    last_sl_time_for_symbol_utc=last_sl_time,
                ),
                config=cfg.filters,
            )
            if flt.blocked:
                filters_blocked += 1
                continue

            # Settle on 1m bars via shared SC-09 engine with human delay
            horizon_bars = (cfg.entry_valid_bars_5m * 5) + (cfg.max_hold_bars_5m * 5) + 5
            fwd_1m = bars_1m[idx_1m : idx_1m + horizon_bars]
            if (
                setup.entry_limit is None
                or setup.stop_loss is None
                or setup.tp1 is None
                or setup.tp2 is None
                or setup.risk_bps is None
            ):
                continue
            trade_spec = ScalpTradeSpec.from_cost_breakdown(
                direction=setup.direction,
                entry_limit=setup.entry_limit,
                stop_loss=setup.stop_loss,
                tp1=setup.tp1,
                tp2=setup.tp2,
                signal_time_utc=decision_time,
                cost=setup.cost_breakdown,
                tp1_close_frac=cfg.tp1_close_frac,
                human_delay_sec=cfg.human_delay_sec,
                entry_valid_minutes=cfg.entry_valid_bars_5m * 5,
                max_hold_minutes=cfg.max_hold_bars_5m * 5,
                tick_size=tick_map.get(symbol, 1e-6),
            )
            outcome = settle_scalp_trade_1m(trade_spec, fwd_1m)
            all_trades.append(
                ScalpBacktestTrade(
                    symbol=symbol,
                    mode=mode_res.mode,
                    direction=setup.direction,
                    signal_time_utc=decision_time,
                    entry_limit=setup.entry_limit,
                    stop_loss=setup.stop_loss,
                    tp1=setup.tp1,
                    tp2=setup.tp2,
                    risk_bps=setup.risk_bps,
                    expected_cost_bps=setup.cost_breakdown.expected_round_trip_bps,
                    outcome=outcome,
                )
            )
            if outcome.filled and outcome.exit_time_utc is not None:
                busy_until = outcome.exit_time_utc
                daily_realized_r[day_key] += outcome.net_r
                if outcome.status is ScalpOutcomeStatus.SL_HIT:
                    last_sl_time = outcome.exit_time_utc

    all_trades.sort(key=lambda t: t.signal_time_utc)
    return ScalpBacktestResult(
        seed=seed,
        cost_multiplier=cost_multiplier,
        contiguity=tuple(contiguity_reports),
        trades=tuple(all_trades),
        decisions_evaluated=decisions_evaluated,
        filters_blocked_count=filters_blocked,
    )


def run_random_entry_benchmark(
    strategy_trades: Sequence[ScalpBacktestTrade],
    candles_1m_by_symbol: Mapping[str, Sequence[MarketCandle]],
    *,
    config: ScalpConfig | None = None,
    seed: int = 42,
    bootstrap_resamples: int = 500,
) -> RandomBenchmarkComparison:
    """Random-entry benchmark (§9.3): same symbols, timestamps, SL/TP & costs, random direction."""
    cfg = config or ScalpConfig()
    rng = np.random.default_rng(seed)

    # Pre-index 1m bars per symbol
    indexed: dict[str, tuple[list[MarketCandle], list[datetime]]] = {}
    for sym, raw in candles_1m_by_symbol.items():
        bars = sorted((c for c in raw if c.is_closed), key=lambda c: c.open_time)
        indexed[sym] = (bars, [c.open_time.astimezone(UTC) for c in bars])

    strat_rs: list[float] = []
    rand_rs: list[float] = []

    for trade in strategy_trades:
        if not trade.outcome.filled:
            continue
        strat_rs.append(trade.outcome.net_r)
        entry_data = indexed.get(trade.symbol)
        if entry_data is None:
            continue
        bars_1m, open_times = entry_data
        # Binary search for first bar >= signal_time
        sig_t = trade.signal_time_utc.astimezone(UTC)
        pos = bisect_left(open_times, sig_t)
        horizon = (cfg.entry_valid_bars_5m * 5) + (cfg.max_hold_bars_5m * 5) + 5
        fwd = bars_1m[pos : pos + horizon]
        if not fwd:
            continue

        rand_dir: Literal["long", "short"] = "long" if bool(rng.integers(0, 2)) else "short"
        ref_px = fwd[0].open
        risk_dist = abs(trade.entry_limit - trade.stop_loss)
        tp1_dist = abs(trade.tp1 - trade.entry_limit)
        tp2_dist = abs(trade.tp2 - trade.entry_limit)
        if rand_dir == "long":
            sl_px = max(1e-8, ref_px - risk_dist)
            tp1_px = ref_px + tp1_dist
            tp2_px = ref_px + tp2_dist
        else:
            sl_px = ref_px + risk_dist
            tp1_px = max(1e-8, ref_px - tp1_dist)
            tp2_px = max(1e-8, ref_px - tp2_dist)

        spec = ScalpTradeSpec(
            direction=rand_dir,
            entry_limit=ref_px,
            stop_loss=sl_px,
            tp1=tp1_px,
            tp2=tp2_px,
            signal_time_utc=sig_t,
            tp1_close_frac=cfg.tp1_close_frac,
            human_delay_sec=cfg.human_delay_sec,
            entry_valid_minutes=cfg.entry_valid_bars_5m * 5,
            max_hold_minutes=cfg.max_hold_bars_5m * 5,
            entry_maker_bps=cfg.costs.maker_fee_bps,
            tp_maker_bps=cfg.costs.maker_fee_bps,
            sl_taker_bps=(
                cfg.costs.taker_fee_bps
                + cfg.costs.fallback_spread_bps / 2.0
                + cfg.costs.fallback_taker_slippage_bps
            ),
        )
        rand_out = settle_scalp_trade_1m(spec, fwd)
        if rand_out.filled:
            rand_rs.append(rand_out.net_r)

    strat_exp = float(np.mean(strat_rs)) if strat_rs else 0.0
    rand_exp = float(np.mean(rand_rs)) if rand_rs else 0.0
    diff = strat_exp - rand_exp

    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    n_pair = min(len(strat_rs), len(rand_rs))
    if n_pair >= 2:
        paired_diffs = np.asarray(strat_rs[:n_pair]) - np.asarray(rand_rs[:n_pair])
        diff_ci_low, diff_ci_high, _ = stationary_block_bootstrap_ci(
            paired_diffs,
            resamples=bootstrap_resamples,
            mean_block_length=min(5.0, max(1.0, n_pair / 4.0)),
            seed=seed,
        )

    beats_random = diff_ci_low is not None and diff_ci_low > 0.0
    return RandomBenchmarkComparison(
        seed=seed,
        strategy_expectancy_r=strat_exp,
        random_expectancy_r=rand_exp,
        diff_expectancy_r=diff,
        diff_ci_low_r=diff_ci_low,
        diff_ci_high_r=diff_ci_high,
        beats_random=beats_random,
    )


def compute_scalp_validation_report(
    *,
    base_result: ScalpBacktestResult,
    candles_1m_by_symbol: Mapping[str, Sequence[MarketCandle]],
    trial_id: str,
    params_sha256: str,
    registered_trials_count: int = 1,
    config: ScalpConfig | None = None,
    seed: int = 42,
    bootstrap_resamples: int = 500,
    holdout_days: int = 90,
) -> ScalpValidationReport:
    """Build §9.3 validation report (holdout, walk-forward, cost sweep, DSR, kill flags)."""
    cfg = config or ScalpConfig()
    trades = base_result.trades

    win_start = trades[0].signal_time_utc if trades else None
    win_end = trades[-1].signal_time_utc if trades else None

    overall = summarize_trades_bucket(
        "overall", trades, seed=seed, bootstrap_resamples=bootstrap_resamples
    )

    # Split final `holdout_days` (3 months) vs walk-forward monthly folds
    if win_end is not None and win_start is not None and (win_end - win_start).days > holdout_days:
        holdout_cutoff = win_end - timedelta(days=holdout_days)
        wf_trades = [t for t in trades if t.signal_time_utc < holdout_cutoff]
        holdout_trades = [t for t in trades if t.signal_time_utc >= holdout_cutoff]
    else:
        wf_trades = list(trades)
        holdout_trades = list(trades)

    holdout = summarize_trades_bucket(
        "holdout_3m", holdout_trades, seed=seed, bootstrap_resamples=bootstrap_resamples
    )

    # Monthly walk-forward folds
    by_month: dict[str, list[ScalpBacktestTrade]] = defaultdict(list)
    for t in wf_trades:
        by_month[t.signal_time_utc.astimezone(UTC).strftime("%Y-%m")].append(t)
    wf_folds = tuple(
        summarize_trades_bucket(m, by_month[m], seed=seed, bootstrap_resamples=bootstrap_resamples)
        for m in sorted(by_month)
    )

    # Breakdowns: mode, symbol, hour-of-day (UTC), weekday
    def _group_stats(
        key_fn: Any,
    ) -> tuple[ScalpBucketStats, ...]:
        groups: dict[str, list[ScalpBacktestTrade]] = defaultdict(list)
        for t in trades:
            groups[str(key_fn(t))].append(t)
        return tuple(
            summarize_trades_bucket(
                k, groups[k], seed=seed, bootstrap_resamples=bootstrap_resamples
            )
            for k in sorted(groups)
        )

    weekday_names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    by_mode = _group_stats(lambda t: t.mode.value)
    by_symbol = _group_stats(lambda t: t.symbol)
    by_hour = _group_stats(lambda t: f"{t.signal_time_utc.astimezone(UTC).hour:02d}:00")
    by_weekday = _group_stats(
        lambda t: (
            f"{t.signal_time_utc.astimezone(UTC).weekday()}-{weekday_names[t.signal_time_utc.astimezone(UTC).weekday()]}"
        )
    )

    # Cost sensitivity sweep at 1.0x / 1.5x / 2.0x (exact re-pricing of trade costs)
    filled_trades = [t for t in trades if t.outcome.filled]
    cost_sweep: list[CostSweepPoint] = []
    for mult in (1.0, 1.5, 2.0):
        adj_rs = [t.outcome.gross_r - (t.outcome.cost_r * mult) for t in filled_trades]
        if adj_rs:
            exp_m = float(np.mean(adj_rs))
            wr_m = sum(1 for r in adj_rs if r > 0.0) / len(adj_rs)
            pf_m = _compute_profit_factor(adj_rs)
            if pf_m is not None and not np.isfinite(pf_m):
                pf_m = 999.0
            ci_lo_m: float | None = None
            ci_hi_m: float | None = None
            if len(adj_rs) >= 2:
                ci_lo_m, ci_hi_m, _ = stationary_block_bootstrap_ci(
                    adj_rs,
                    resamples=bootstrap_resamples,
                    mean_block_length=min(5.0, max(1.0, len(adj_rs) / 4.0)),
                    seed=seed,
                )
        else:
            exp_m, wr_m, pf_m, ci_lo_m, ci_hi_m = 0.0, 0.0, None, None, None
        cost_sweep.append(
            CostSweepPoint(
                multiplier=mult,
                filled_trades=len(adj_rs),
                expectancy_net_r=exp_m,
                ci_low_r=ci_lo_m,
                ci_high_r=ci_hi_m,
                win_rate=wr_m,
                profit_factor=pf_m,
            )
        )

    # Break-even cost multiplier: mean(gross_r) / mean(cost_r)
    breakeven_mult: float | None = None
    if filled_trades:
        mean_gross = float(np.mean([t.outcome.gross_r for t in filled_trades]))
        mean_cost = float(np.mean([t.outcome.cost_r for t in filled_trades]))
        if mean_cost > 1e-12:
            breakeven_mult = mean_gross / mean_cost

    # DSR with registered trial count on daily R returns
    _, daily_rs = _daily_r_series(filled_trades)
    dsr_prob: float | None = None
    sr_daily: float | None = None
    if len(daily_rs) >= 3 and float(np.std(daily_rs)) > 1e-12:
        sr_daily = float(sample_sharpe_ratio(daily_rs))
        dsr_res = deflated_sharpe_ratio(
            daily_rs,
            trials=max(1, registered_trials_count),
            variance_of_trial_sharpes=0.25 if registered_trials_count > 1 else None,
            obs_frequency="daily",
        )
        dsr_prob = float(dsr_res.deflated_sharpe_ratio)

    # Random-entry benchmark
    rand_cmp = run_random_entry_benchmark(
        trades,
        candles_1m_by_symbol,
        config=cfg,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )

    # §9.4 Kill criteria
    kill_flags: list[str] = []
    if overall.signals > 0 and overall.fill_rate < 0.40:
        kill_flags.append(f"LOW_FILL_RATE({overall.fill_rate * 100:.1f}%<40%)")
    if overall.expectancy_gross_r > 0.0 and overall.expectancy_net_r <= 0.0:
        kill_flags.append("COSTS_ERASE_GROSS_EDGE")
    pos_symbols = [s.label for s in by_symbol if s.filled_trades > 0 and s.expectancy_net_r > 0.0]
    if len(by_symbol) >= 2 and len(pos_symbols) <= 1 and overall.expectancy_net_r > 0.0:
        kill_flags.append("EDGE_CONCENTRATED_SINGLE_SYMBOL")

    # §9.3 Acceptance gate for going to shadow
    sweep_15 = next((p for p in cost_sweep if abs(p.multiplier - 1.5) < 1e-6), None)
    holdout_15_ok = sweep_15 is not None and sweep_15.expectancy_net_r >= 0.0
    active_modes_enough = bool(by_mode) and all(m.filled_trades >= 300 for m in by_mode)
    passes_shadow = (
        holdout.expectancy_net_r > 0.0
        and holdout_15_ok
        and active_modes_enough
        and rand_cmp.beats_random
        and not kill_flags
    )

    return ScalpValidationReport(
        trial_id=trial_id,
        registered_trials_count=registered_trials_count,
        params_sha256=params_sha256,
        seed=seed,
        window_start_utc=win_start,
        window_end_utc=win_end,
        overall=overall,
        holdout=holdout,
        walk_forward_folds=wf_folds,
        by_mode=by_mode,
        by_symbol=by_symbol,
        by_hour_utc=by_hour,
        by_weekday=by_weekday,
        cost_sweep=tuple(cost_sweep),
        breakeven_cost_multiplier=breakeven_mult,
        dsr_probability=dsr_prob,
        sample_sharpe_daily=sr_daily,
        random_benchmark=rand_cmp,
        passes_shadow_gate=passes_shadow,
        kill_flags=tuple(kill_flags),
    )


def render_scalp_validation_markdown(report: ScalpValidationReport) -> str:
    """Render `docs/research/scalp_v1-backtest.md` with full §9.3 tables and trial IDs."""
    kill_str = ", ".join(report.kill_flags) if report.kill_flags else "NONE"
    be_str = (
        f"{report.breakeven_cost_multiplier:.2f}x"
        if report.breakeven_cost_multiplier is not None
        else "N/A"
    )
    sr_str = (
        f"{report.sample_sharpe_daily:.3f}" if report.sample_sharpe_daily is not None else "N/A"
    )
    dsr_str = f"{report.dsr_probability:.4f}" if report.dsr_probability is not None else "N/A"
    lines: list[str] = [
        "# `scalp_v1` Backtest Validation Report",
        "",
        f"- **Trial ID**: `{report.trial_id}`",
        f"- **Registered trials (N)**: `{report.registered_trials_count}`",
        f"- **Params SHA-256**: `{report.params_sha256}`",
        f"- **Seed**: `{report.seed}`",
        f"- **Window (UTC)**: `{report.window_start_utc}` → `{report.window_end_utc}`",
        f"- **Shadow Gate Passed (§9.3)**: `{'YES' if report.passes_shadow_gate else 'NO'}`",
        f"- **Kill Flags (§9.4)**: `{kill_str}`",
        "",
        "## 1. Overall & Holdout Summary",
        "",
        "| Slice | Signals | Filled | Fill Rate | Win Rate "
        "| Gross E[R] | Net E[R] | 95% Block-Bootstrap CI | PF | Max DD (R) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for s in (report.overall, report.holdout):
        ci = (
            f"[{s.ci_low_r:+.3f}, {s.ci_high_r:+.3f}]"
            if s.ci_low_r is not None and s.ci_high_r is not None
            else "-"
        )
        pf_str = f"{s.profit_factor:.2f}" if s.profit_factor is not None else "-"
        lines.append(
            f"| {s.label} | {s.signals} | {s.filled_trades} | {s.fill_rate * 100:.1f}% "
            f"| {s.win_rate * 100:.1f}% | {s.expectancy_gross_r:+.3f}R "
            f"| {s.expectancy_net_r:+.3f}R | {ci} | {pf_str} | {s.max_dd_r:.2f}R |"
        )

    lines.extend(
        [
            "",
            "## 2. Cost Sensitivity & DSR",
            "",
            f"- **Break-even cost multiplier**: `{be_str}`",
            f"- **Daily Sample Sharpe**: `{sr_str}`",
            f"- **Deflated Sharpe Ratio (DSR)**: `{dsr_str}`",
            "",
            "| Cost Multiplier | Filled | Net E[R] | 95% CI | Win Rate | PF |",
            "|---:|---:|---:|---|---:|---:|",
        ]
    )
    for pt in report.cost_sweep:
        ci = (
            f"[{pt.ci_low_r:+.3f}, {pt.ci_high_r:+.3f}]"
            if pt.ci_low_r is not None and pt.ci_high_r is not None
            else "-"
        )
        pf_str = f"{pt.profit_factor:.2f}" if pt.profit_factor is not None else "-"
        lines.append(
            f"| {pt.multiplier:.1f}x | {pt.filled_trades} | {pt.expectancy_net_r:+.3f}R "
            f"| {ci} | {pt.win_rate * 100:.1f}% | {pf_str} |"
        )

    rb = report.random_benchmark
    rb_ci = (
        f"[{rb.diff_ci_low_r:+.3f}, {rb.diff_ci_high_r:+.3f}]"
        if rb.diff_ci_low_r is not None and rb.diff_ci_high_r is not None
        else "-"
    )
    lines.extend(
        [
            "",
            "## 3. Random-Entry Benchmark",
            "",
            f"- Strategy E[R]: `{rb.strategy_expectancy_r:+.3f}R`",
            f"- Random-Entry E[R]: `{rb.random_expectancy_r:+.3f}R`",
            f"- Difference (Strategy - Random): `{rb.diff_expectancy_r:+.3f}R` (95% CI: `{rb_ci}`)",
            f"- Beats Random Benchmark: `{'YES' if rb.beats_random else 'NO'}`",
        ]
    )

    for title, buckets in (
        ("4. Breakdown by Mode", report.by_mode),
        ("5. Breakdown by Symbol", report.by_symbol),
        ("6. Walk-Forward Monthly Folds", report.walk_forward_folds),
        ("7. Breakdown by Hour (UTC)", report.by_hour_utc),
        ("8. Breakdown by Weekday", report.by_weekday),
    ):
        if not buckets:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Bucket | Signals | Filled | Fill Rate | Win Rate | Net E[R] | Max DD (R) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for b in buckets:
            lines.append(
                f"| {b.label} | {b.signals} | {b.filled_trades} | {b.fill_rate * 100:.1f}% "
                f"| {b.win_rate * 100:.1f}% | {b.expectancy_net_r:+.3f}R | {b.max_dd_r:.2f}R |"
            )

    return "\n".join(lines) + "\n"


def write_scalp_validation_report(
    report: ScalpValidationReport,
    *,
    output_path: Path = Path("docs/research/scalp_v1-backtest.md"),
) -> Path:
    """Write rendered markdown validation report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_scalp_validation_markdown(report), encoding="utf-8")
    return output_path
