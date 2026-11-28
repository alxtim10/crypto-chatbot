"""Deterministic macro benchmark context, correlation, beta, and positioning regime (ADR 0013)."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np

from crypto_chatbot.analysis.indicators import (
    compute_technical_metrics,
    ema_series,
    rsi_series,
)
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.market_data.macro_provider import is_dxy_market_closed
from crypto_chatbot.schemas.analysis import (
    DerivativesMetrics,
    MarketStructure,
    Regime,
)
from crypto_chatbot.schemas.derivatives import OrderFlowMetrics
from crypto_chatbot.schemas.macro import (
    BtcBenchmarkMetrics,
    DxySnapshotMetrics,
    MacroBenchmarkContext,
    PositioningRegime,
)
from crypto_chatbot.schemas.market import MarketCandle, Timeframe


def correlation_critical_value(n: int, *, alpha: float = 0.05) -> float:
    """Two-sided 5% Pearson correlation critical value `r_crit(n, 0.05)` (PRD v1.7 §4.3).

    Uses exact `r_crit = t_{1-alpha/2, df} / sqrt(df + t^2)` where `df = n - 2`.
    Gives ~0.4438 at n=20 and ~0.2542 at n=60.
    """
    if n <= 2:
        return 1.0
    df = float(n - 2)
    # Two-sided 95% (z = 1.95996398454) Cornish-Fisher expansion for Student-t quantile
    z = 1.959963984540054 if alpha <= 0.05 else 1.6448536269514722
    g1 = (z**3 + z) / (4.0 * df)
    g2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    g3 = (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
    t_crit = z + g1 + g2 + g3
    return float(t_crit / math.sqrt(df + t_crit * t_crit))


def asof_join_macro_candles(
    target_candles: Sequence[MarketCandle],
    macro_candles: Sequence[MarketCandle],
    *,
    max_staleness: timedelta | None = None,
) -> tuple[list[tuple[MarketCandle, MarketCandle]], int]:
    """Backward point-in-time as-of join between crypto candles and macro candles (PRD §3.3).

    For each crypto candle `t`, picks the latest macro candle with `available_time <= close_time_t`
    within `max_staleness` (default: 72h for 1d covering weekends, else `2 * timeframe.duration`).
    Returns `(joined_pairs, stale_excluded_count)`.
    """
    if not target_candles or not macro_candles:
        return [], 0
    sorted_target = sorted(target_candles, key=lambda c: c.close_time)
    sorted_macro = sorted(macro_candles, key=lambda c: c.available_time)
    macro_times = [c.available_time.astimezone(UTC) for c in sorted_macro]

    tf = sorted_target[0].timeframe
    staleness_limit = max_staleness or (
        timedelta(hours=72) if tf is Timeframe.D1 else tf.duration * 2
    )

    joined: list[tuple[MarketCandle, MarketCandle]] = []
    stale_count = 0
    for tc in sorted_target:
        tc_time = tc.close_time.astimezone(UTC)
        pos = bisect_right(macro_times, tc_time) - 1
        if pos < 0:
            stale_count += 1
            continue
        mc = sorted_macro[pos]
        age = tc_time - macro_times[pos]
        if age > staleness_limit:
            stale_count += 1
            continue
        joined.append((tc, mc))
    return joined, stale_count


def classify_positioning_regime(
    *,
    price_ret_pct: float | None,
    oi_change_pct: float | None,
    cvd: float | None,
    funding_rate_8h: float | None,
) -> PositioningRegime:
    """Classify futures positioning regime from price delta, OI delta, CVD, and funding."""
    # 1. Crowded squeeze risk check: extreme funding or funding opposing price direction
    if funding_rate_8h is not None and abs(funding_rate_8h) >= 0.0005:
        if funding_rate_8h > 0 and price_ret_pct is not None and price_ret_pct < 0:
            return PositioningRegime.CROWDED_SQUEEZE_RISK
        if funding_rate_8h < 0 and price_ret_pct is not None and price_ret_pct > 0:
            return PositioningRegime.CROWDED_SQUEEZE_RISK
        if abs(funding_rate_8h) >= 0.0010:
            return PositioningRegime.CROWDED_SQUEEZE_RISK

    if price_ret_pct is None or oi_change_pct is None:
        return PositioningRegime.NEUTRAL_POSITIONING

    # 2. Long Buildup: Price UP + OI UP + CVD Buyer Aggression
    if price_ret_pct >= 0 and oi_change_pct > 0 and (cvd is None or cvd >= 0):
        return PositioningRegime.LONG_BUILDUP

    # 3. Short Buildup: Price DOWN + OI UP + CVD Seller Aggression
    if price_ret_pct < 0 and oi_change_pct > 0 and (cvd is None or cvd <= 0):
        return PositioningRegime.SHORT_BUILDUP

    # 4. Short Covering: Price UP + OI DOWN + CVD Buyer Aggression
    if price_ret_pct > 0 and oi_change_pct < 0 and (cvd is None or cvd >= 0):
        return PositioningRegime.SHORT_COVERING

    # 5. Long Liquidation Flush: Price DOWN + OI DOWN + CVD Seller Aggression
    if price_ret_pct < 0 and oi_change_pct < 0 and (cvd is None or cvd <= 0):
        return PositioningRegime.LONG_LIQUIDATION_FLUSH

    return PositioningRegime.NEUTRAL_POSITIONING


def compute_rolling_correlation(
    series_a: Sequence[float],
    series_b: Sequence[float],
    window: int = 20,
) -> float | None:
    """Compute Pearson correlation coefficient r in [-1.0, 1.0] over the last `window` points."""
    if len(series_a) != len(series_b) or len(series_a) < 3:
        return None

    a = np.asarray(series_a[-window:], dtype=np.float64)
    b = np.asarray(series_b[-window:], dtype=np.float64)

    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return None

    a_v = a[valid]
    b_v = b[valid]

    a_diff = a_v - np.mean(a_v)
    b_diff = b_v - np.mean(b_v)

    var_a = np.sum(a_diff**2)
    var_b = np.sum(b_diff**2)
    denom = math.sqrt(var_a * var_b)

    if denom <= 1e-12:
        return None

    r = float(np.sum(a_diff * b_diff) / denom)
    return round(max(-1.0, min(1.0, r)), 4)


def compute_rolling_beta(
    target_returns: Sequence[float],
    btc_returns: Sequence[float],
    window: int = 20,
) -> float | None:
    """Compute rolling Beta = Cov(target, btc) / Var(btc) over the last `window` points."""
    if len(target_returns) != len(btc_returns) or len(target_returns) < 3:
        return None

    t = np.asarray(target_returns[-window:], dtype=np.float64)
    b = np.asarray(btc_returns[-window:], dtype=np.float64)

    valid = np.isfinite(t) & np.isfinite(b)
    if np.sum(valid) < 3:
        return None

    t_v = t[valid]
    b_v = b[valid]

    b_diff = b_v - np.mean(b_v)
    var_b = np.sum(b_diff**2) / (len(b_v) - 1)

    if var_b <= 1e-12:
        return None

    t_diff = t_v - np.mean(t_v)
    cov = np.sum(t_diff * b_diff) / (len(t_v) - 1)

    beta = float(cov / var_b)
    if not math.isfinite(beta):
        return None
    return round(beta, 4)


def build_macro_benchmark_context(
    *,
    target_candles: Sequence[MarketCandle],
    target_canonical_symbol: str,
    as_of: datetime,
    dxy_candles: Sequence[MarketCandle] | None = None,
    btc_candles: Sequence[MarketCandle] | None = None,
    derivatives: DerivativesMetrics | None = None,
    order_flow: OrderFlowMetrics | None = None,
    dxy_corr_window: int = 20,
) -> MacroBenchmarkContext:
    """Build complete, causal MacroBenchmarkContext respecting point-in-time constraints."""
    # Filter target candles strictly up to as_of
    valid_targets = [c for c in target_candles if c.close_time <= as_of and c.is_closed]
    if not valid_targets:
        return MacroBenchmarkContext(as_of=as_of)

    # 1. Compute positioning regime
    price_ret: float | None = None
    if len(valid_targets) >= 2:
        price_ret = (valid_targets[-1].close / valid_targets[-2].close - 1.0) * 100.0

    oi_change_pct = derivatives.open_interest_change_pct if derivatives is not None else None
    cvd = order_flow.cvd_20 if order_flow is not None else None
    funding_8h = derivatives.funding_rate_per_8h if derivatives is not None else None

    positioning_regime = classify_positioning_regime(
        price_ret_pct=price_ret,
        oi_change_pct=oi_change_pct,
        cvd=cvd,
        funding_rate_8h=funding_8h,
    )

    # 2. Process DXY Context
    dxy_metrics: DxySnapshotMetrics | None = None
    if dxy_candles:
        valid_dxy = [c for c in dxy_candles if c.close_time <= as_of and c.is_closed]
        if valid_dxy:
            last_dxy = valid_dxy[-1]
            dxy_closes = [c.close for c in valid_dxy]

            dxy_ret_1: float | None = None
            if len(dxy_closes) >= 2:
                dxy_ret_1 = round((dxy_closes[-1] / dxy_closes[-2] - 1.0) * 100.0, 4)

            dxy_ret_20: float | None = None
            if len(dxy_closes) >= 21:
                dxy_ret_20 = round((dxy_closes[-1] / dxy_closes[-21] - 1.0) * 100.0, 4)

            dxy_ema_20_arr = ema_series(dxy_closes, 20)
            dxy_ema_20 = (
                float(dxy_ema_20_arr[-1])
                if len(dxy_ema_20_arr) > 0 and np.isfinite(dxy_ema_20_arr[-1])
                else None
            )

            dxy_ema_50_arr = ema_series(dxy_closes, 50)
            dxy_ema_50 = (
                float(dxy_ema_50_arr[-1])
                if len(dxy_ema_50_arr) > 0 and np.isfinite(dxy_ema_50_arr[-1])
                else None
            )

            dxy_rsi_arr, _ = rsi_series(dxy_closes, 14)
            dxy_rsi_14 = (
                float(dxy_rsi_arr[-1])
                if len(dxy_rsi_arr) > 0 and np.isfinite(dxy_rsi_arr[-1])
                else None
            )

            trend: Literal["bullish", "bearish", "neutral"] = "neutral"
            if dxy_ema_20 is not None and dxy_ema_50 is not None:
                if dxy_ema_20 > dxy_ema_50:
                    trend = "bullish"
                elif dxy_ema_20 < dxy_ema_50:
                    trend = "bearish"

            # Backward as-of join (handles FX weekend gaps, PRD v1.7 §3.3)
            corr_dxy: float | None = None
            n_corr_obs = 0
            joined_dxy, _stale_dxy = asof_join_macro_candles(valid_targets, valid_dxy)
            aligned_t_dxy = [tc.close for tc, _mc in joined_dxy]
            aligned_d = [mc.close for _tc, mc in joined_dxy]
            if len(aligned_t_dxy) >= 4:
                t_rets = [
                    aligned_t_dxy[i] / aligned_t_dxy[i - 1] - 1.0
                    for i in range(1, len(aligned_t_dxy))
                ]
                d_rets = [aligned_d[i] / aligned_d[i - 1] - 1.0 for i in range(1, len(aligned_d))]
                n_corr_obs = min(len(t_rets), dxy_corr_window)
                corr_dxy = compute_rolling_correlation(t_rets, d_rets, window=dxy_corr_window)

            r_crit = correlation_critical_value(n_corr_obs) if n_corr_obs > 2 else 1.0
            macro_impact: Literal["headwind", "tailwind", "neutral"] = "neutral"
            if corr_dxy is not None and abs(corr_dxy) >= r_crit:
                if trend == "bullish" and corr_dxy < 0.0:
                    macro_impact = "headwind"
                elif (trend == "bearish" and corr_dxy < 0.0) or (
                    trend == "bullish" and corr_dxy > 0.0
                ):
                    macro_impact = "tailwind"
                elif trend == "bearish" and corr_dxy > 0.0:
                    macro_impact = "headwind"

            is_closed = is_dxy_market_closed(as_of)

            dxy_metrics = DxySnapshotMetrics(
                symbol="DXY",
                as_of=last_dxy.close_time,
                last_close=last_dxy.close,
                change_1_bar_pct=dxy_ret_1,
                change_20_bar_pct=dxy_ret_20,
                ema_20=dxy_ema_20,
                ema_50=dxy_ema_50,
                rsi_14=dxy_rsi_14,
                trend=trend,
                rolling_corr_20=corr_dxy,
                macro_impact=macro_impact,
                is_weekend_or_closed=is_closed,
            )

    # 3. Process BTC Benchmark
    btc_metrics: BtcBenchmarkMetrics | None = None
    is_btc_target = "BTC" in target_canonical_symbol.upper()

    if is_btc_target:
        btc_metrics = BtcBenchmarkMetrics(
            canonical_symbol="BTC/USDT:USDT",
            as_of=valid_targets[-1].close_time,
            last_close=valid_targets[-1].close,
            leader_alignment="self",
        )
    elif btc_candles:
        valid_btc = [c for c in btc_candles if c.close_time <= as_of and c.is_closed]
        if valid_btc:
            last_btc = valid_btc[-1]
            btc_map = {c.close_time: c.close for c in valid_btc}

            aligned_target: list[float] = []
            aligned_btc: list[float] = []
            for tc in valid_targets:
                if tc.close_time in btc_map:
                    aligned_target.append(tc.close)
                    aligned_btc.append(btc_map[tc.close_time])

            rs_20: float | None = None
            btc_ret_20: float | None = None
            corr_btc: float | None = None
            beta_btc: float | None = None

            if len(aligned_target) >= 21:
                t_ret_20_val = aligned_target[-1] / aligned_target[-21] - 1.0
                b_ret_20_val = aligned_btc[-1] / aligned_btc[-21] - 1.0
                btc_ret_20 = round(b_ret_20_val * 100.0, 2)
                rs_20 = round((t_ret_20_val - b_ret_20_val) * 100.0, 2)

            if len(aligned_target) >= 4:
                t_rets = [
                    aligned_target[i] / aligned_target[i - 1] - 1.0
                    for i in range(1, len(aligned_target))
                ]
                b_rets = [
                    aligned_btc[i] / aligned_btc[i - 1] - 1.0 for i in range(1, len(aligned_btc))
                ]
                corr_btc = compute_rolling_correlation(t_rets, b_rets, window=20)
                beta_btc = compute_rolling_beta(t_rets, b_rets, window=20)

            # Analyze BTC trend/structure if enough candles
            btc_regime: Regime | None = None
            btc_structure_enum: MarketStructure | None = None
            if len(valid_btc) >= 60:
                try:
                    btc_ind = compute_technical_metrics(valid_btc)
                    btc_regime = classify_regime(btc_ind.metrics)
                    btc_st = analyze_market_structure(valid_btc)
                    btc_structure_enum = btc_st.market_structure
                except Exception:  # noqa: S110
                    pass

            leader_alignment: Literal["aligned", "divergent_warning", "self"] = "aligned"
            if btc_regime == Regime.TRENDING_DOWN or btc_structure_enum == MarketStructure.BEARISH:
                leader_alignment = "divergent_warning"

            btc_metrics = BtcBenchmarkMetrics(
                canonical_symbol="BTC/USDT:USDT",
                as_of=last_btc.close_time,
                last_close=last_btc.close,
                change_20_bar_pct=btc_ret_20,
                regime=btc_regime,
                market_structure=btc_structure_enum,
                relative_strength_20_pct=rs_20,
                rolling_corr_20=corr_btc,
                rolling_beta_20=beta_btc,
                leader_alignment=leader_alignment,
            )

    return MacroBenchmarkContext(
        as_of=as_of,
        dxy=dxy_metrics,
        btc_benchmark=btc_metrics,
        positioning_regime=positioning_regime,
    )
