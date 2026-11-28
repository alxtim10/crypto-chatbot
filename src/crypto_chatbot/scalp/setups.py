"""SC-05: Deterministic Trend Pullback and Range Fade setup detectors for `scalp_v1` (§4)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.analysis.indicators import compute_technical_metrics
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.scalp.config import ScalpConfig
from crypto_chatbot.scalp.costs import ScalpCostBreakdown, estimate_scalp_trade_costs
from crypto_chatbot.scalp.mode import ScalpMode
from crypto_chatbot.schemas.analysis import StructureEventType
from crypto_chatbot.schemas.derivatives import OrderBookSnapshot
from crypto_chatbot.schemas.market import MarketCandle


class ValidDealingRange(BaseModel):
    """Validated 1h/15m dealing range for Range Fade (§4.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    range_low: float = Field(gt=0.0)
    range_high: float = Field(gt=0.0)
    width_price: float = Field(gt=0.0)
    width_bps: float = Field(gt=0.0)
    source_timeframe: Literal["1h", "15m"]


class ScalpSetupResult(BaseModel):
    """Output of SC-05 setup evaluation (`READY` or `WAIT` with explicit reason code)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["READY", "WAIT"]
    mode: ScalpMode
    direction: Literal["long", "short"] | None = None
    reason_code: str
    reason_detail: str
    zone_source: str | None = None
    trigger_type: str | None = None
    entry_low: float | None = Field(default=None, gt=0.0)
    entry_high: float | None = Field(default=None, gt=0.0)
    entry_limit: float | None = Field(default=None, gt=0.0)
    stop_loss: float | None = Field(default=None, gt=0.0)
    tp1: float | None = Field(default=None, gt=0.0)
    tp2: float | None = Field(default=None, gt=0.0)
    risk_bps: float | None = Field(default=None, gt=0.0)
    gross_rr_tp1: float | None = None
    gross_rr_tp2: float | None = None
    net_rr_tp2: float | None = None
    entry_valid_until_utc: datetime | None = None
    max_hold_minutes: int = 90
    cost_breakdown: ScalpCostBreakdown | None = None


def _wait(mode: ScalpMode, code: str, detail: str) -> ScalpSetupResult:
    return ScalpSetupResult(
        status="WAIT",
        mode=mode,
        reason_code=code,
        reason_detail=detail,
    )


def detect_valid_dealing_range(
    *,
    candles_1h: Sequence[MarketCandle],
    candles_15m: Sequence[MarketCandle],
    atr_1h: float | None,
    expected_round_trip_cost_bps: float,
    config: ScalpConfig | None = None,
) -> ValidDealingRange | None:
    """Detect a valid 1h/15m dealing range from the last two confirmed swings (§4.2).

    Valid if:
      - width >= 6 * expected_round_trip_cost
      - width >= 1.5 * ATR(1h)
    """
    cfg = config or ScalpConfig()
    if atr_1h is None or atr_1h <= 0.0:
        return None

    for tf_label, candles in (("1h", candles_1h), ("15m", candles_15m)):
        if len(candles) < 10:
            continue
        struct = analyze_market_structure(candles)
        swing_highs = [ev for ev in struct.pivots if ev.event_type is StructureEventType.SWING_HIGH]
        swing_lows = [ev for ev in struct.pivots if ev.event_type is StructureEventType.SWING_LOW]
        if not swing_highs or not swing_lows:
            continue
        rng_high = swing_highs[-1].level
        rng_low = swing_lows[-1].level
        if rng_high <= rng_low or rng_low <= 0.0:
            continue
        width_price = rng_high - rng_low
        mid = (rng_high + rng_low) / 2.0
        width_bps = (width_price / mid) * 10_000.0
        min_cost_bps = cfg.range_min_cost_mult * max(0.0, expected_round_trip_cost_bps)
        min_atr_price = cfg.range_min_atr_1h_mult * atr_1h
        if width_bps >= min_cost_bps and width_price >= min_atr_price:
            return ValidDealingRange(
                range_low=rng_low,
                range_high=rng_high,
                width_price=width_price,
                width_bps=width_bps,
                source_timeframe=tf_label,  # type: ignore[arg-type]
            )
    return None


def _compute_net_rr(
    *,
    entry_price: float,
    target: float,
    stop_loss: float,
    tp_round_trip_bps: float,
) -> float:
    gross_reward = abs(target - entry_price)
    gross_risk = abs(entry_price - stop_loss)
    if gross_risk <= 0.0:
        return 0.0
    cost_price = (max(0.0, tp_round_trip_bps) / 10_000.0) * entry_price
    net_reward = gross_reward - cost_price
    if net_reward <= 0.0:
        return 0.0
    return net_reward / gross_risk


def _apply_common_rules(
    *,
    mode: ScalpMode,
    direction: Literal["long", "short"],
    canonical_symbol: str,
    zone_source: str,
    trigger_type: str,
    entry_low: float,
    entry_high: float,
    entry_limit: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    decision_time_utc: datetime,
    config: ScalpConfig,
    orderbook: OrderBookSnapshot | None,
    rolling_median_spread_bps: float | None,
    funding_rate_8h: float | None,
    cost_multiplier: float = 1.0,
) -> ScalpSetupResult:
    """Enforce §4.3 common rules: min/max R_bps, cost gate <= 15% of 1R, net R:R TP2 >= 1.3."""
    if direction == "long" and not (stop_loss < entry_limit < tp1 <= tp2):
        return _wait(mode, "INVALID_GEOMETRY", "Long geometry requires SL < entry < TP1 <= TP2")
    if direction == "short" and not (tp2 <= tp1 < entry_limit < stop_loss):
        return _wait(mode, "INVALID_GEOMETRY", "Short geometry requires TP2 <= TP1 < entry < SL")

    risk = abs(entry_limit - stop_loss)
    risk_bps = (risk / entry_limit) * 10_000.0

    if risk_bps < config.min_r_bps:
        return _wait(
            mode,
            "RISK_TOO_TIGHT",
            f"R_bps {risk_bps:.1f} < min_r_bps {config.min_r_bps:.1f} (stop inside noise)",
        )
    if risk_bps > config.max_r_bps:
        return _wait(
            mode,
            "RISK_TOO_WIDE",
            f"R_bps {risk_bps:.1f} > max_r_bps {config.max_r_bps:.1f}",
        )

    max_hold_minutes = config.max_hold_bars_5m * 5
    cost = estimate_scalp_trade_costs(
        canonical_symbol=canonical_symbol,
        direction=direction,
        entry_price=entry_limit,
        stop_loss=stop_loss,
        config=config.costs,
        orderbook=orderbook,
        rolling_median_spread_bps=rolling_median_spread_bps,
        funding_rate_8h=funding_rate_8h,
        decision_time_utc=decision_time_utc,
        max_hold_minutes=max_hold_minutes,
        cost_multiplier=cost_multiplier,
    )

    if not cost.passes_cost_gate:
        return _wait(
            mode,
            "COST_GATE_FAILED",
            (
                f"Expected cost {cost.expected_round_trip_bps:.2f} bps "
                f"({cost.cost_pct_of_r * 100:.1f}% of 1R) exceeds "
                f"{config.costs.max_cost_pct_of_r * 100:.0f}% limit"
            ),
        )

    gross_rr_tp1 = abs(tp1 - entry_limit) / risk
    gross_rr_tp2 = abs(tp2 - entry_limit) / risk
    net_rr_tp2 = _compute_net_rr(
        entry_price=entry_limit,
        target=tp2,
        stop_loss=stop_loss,
        tp_round_trip_bps=cost.tp_round_trip_bps,
    )
    if net_rr_tp2 < config.min_net_rr_tp2:
        return _wait(
            mode,
            "NET_RR_TP2_TOO_LOW",
            f"Net R:R to TP2 is {net_rr_tp2:.2f} < minimum {config.min_net_rr_tp2:.2f}",
        )

    valid_until = decision_time_utc.astimezone(UTC) + timedelta(
        minutes=config.entry_valid_bars_5m * 5
    )
    return ScalpSetupResult(
        status="READY",
        mode=mode,
        direction=direction,
        reason_code="SETUP_READY",
        reason_detail=(
            f"{mode.value} ready via {zone_source} + {trigger_type} "
            f"(R={risk_bps:.1f} bps, net R:R={net_rr_tp2:.2f})"
        ),
        zone_source=zone_source,
        trigger_type=trigger_type,
        entry_low=min(entry_low, entry_high),
        entry_high=max(entry_low, entry_high),
        entry_limit=entry_limit,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk_bps=risk_bps,
        gross_rr_tp1=gross_rr_tp1,
        gross_rr_tp2=gross_rr_tp2,
        net_rr_tp2=net_rr_tp2,
        entry_valid_until_utc=valid_until,
        max_hold_minutes=max_hold_minutes,
        cost_breakdown=cost,
    )


def evaluate_trend_pullback_setup(
    *,
    mode: Literal[ScalpMode.TREND_PULLBACK_LONG, ScalpMode.TREND_PULLBACK_SHORT],
    canonical_symbol: str,
    candles_15m: Sequence[MarketCandle],
    candles_5m: Sequence[MarketCandle],
    decision_time_utc: datetime,
    config: ScalpConfig | None = None,
    orderbook: OrderBookSnapshot | None = None,
    rolling_median_spread_bps: float | None = None,
    funding_rate_8h: float | None = None,
    cost_multiplier: float = 1.0,
) -> ScalpSetupResult:
    """Evaluate SCALP-SPEC-v1 §4.1 Trend Pullback setup."""
    cfg = config or ScalpConfig()
    direction: Literal["long", "short"] = (
        "long" if mode is ScalpMode.TREND_PULLBACK_LONG else "short"
    )

    if len(candles_15m) < 25 or len(candles_5m) < 20:
        return _wait(mode, "INSUFFICIENT_LTF_BARS", "Need >=25 15m bars and >=20 5m bars")

    tech_15m = compute_technical_metrics(candles_15m).metrics
    tech_5m = compute_technical_metrics(candles_5m).metrics
    atr_5m = tech_5m.atr_14
    atr_15m = tech_15m.atr_14
    ema20_15m = tech_15m.ema_20
    if atr_5m is None or atr_5m <= 0.0 or atr_15m is None or atr_15m <= 0.0:
        return _wait(mode, "ATR_UNAVAILABLE", "ATR(14) unavailable on 5m or 15m")

    last_5m = candles_5m[-1]
    smc_15m = analyze_smc(candles_15m)

    # 1. Zone selection: active 15m FVG/OB within 1.5*ATR(5m); else EMA20(15m) +/- 0.15*ATR(15m)
    zone_low: float | None = None
    zone_high: float | None = None
    zone_source: str | None = None
    max_dist = cfg.zone_max_atr_5m * atr_5m

    candidate_zones = (
        [
            ("15m_bullish_fvg", smc_15m.active_bullish_fvg),
            ("15m_bullish_ob", smc_15m.active_bullish_ob),
        ]
        if direction == "long"
        else [
            ("15m_bearish_fvg", smc_15m.active_bearish_fvg),
            ("15m_bearish_ob", smc_15m.active_bearish_ob),
        ]
    )
    for label, z in candidate_zones:
        if z is None:
            continue
        dist = 0.0
        if last_5m.close < z.zone_low:
            dist = z.zone_low - last_5m.close
        elif last_5m.close > z.zone_high:
            dist = last_5m.close - z.zone_high
        if dist <= max_dist:
            zone_low = z.zone_low
            zone_high = z.zone_high
            zone_source = label
            break

    if zone_low is None or zone_high is None or zone_source is None:
        if ema20_15m is None or ema20_15m <= 0.0:
            return _wait(mode, "NO_PULLBACK_ZONE", "No active 15m FVG/OB or EMA20 zone")
        band = cfg.ema20_zone_atr_15m * atr_15m
        zone_low = ema20_15m - band
        zone_high = ema20_15m + band
        zone_source = "15m_ema20_band"

    # 2. Check recent price touch of zone (within last 6 closed 5m bars)
    recent_5m = candles_5m[-6:]
    touched = any(c.low <= zone_high and c.high >= zone_low for c in recent_5m)
    if not touched:
        return _wait(
            mode,
            "ZONE_NOT_TOUCHED",
            f"Price has not touched {zone_source} "
            f"({zone_low:.4f}-{zone_high:.4f}) in last 6 5m bars",
        )

    # 3. Trigger on last closed 5m candle: 5m CHoCH/BOS in trend direction OR rejection candle
    smc_5m = analyze_smc(candles_5m)
    desired_breaks = (
        {StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH}
        if direction == "long"
        else {StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH}
    )
    latest_brk = smc_5m.latest_structure_break
    has_5m_break = (
        latest_brk is not None
        and latest_brk.event_type in desired_breaks
        and latest_brk.confirmation_time == last_5m.close_time
    )

    body = abs(last_5m.close - last_5m.open)
    min_body = cfg.rejection_min_body_atr_5m * atr_5m
    if direction == "long":
        is_rejection = (
            last_5m.close > last_5m.open
            and last_5m.low <= zone_high
            and last_5m.close > zone_high
            and body >= min_body
        )
    else:
        is_rejection = (
            last_5m.close < last_5m.open
            and last_5m.high >= zone_low
            and last_5m.close < zone_low
            and body >= min_body
        )

    if has_5m_break:
        trigger_type = "5m_choch_bos"
    elif is_rejection:
        trigger_type = "5m_rejection_candle"
    else:
        return _wait(
            mode,
            "NO_5M_TRIGGER",
            f"No 5m CHoCH/BOS or rejection candle (body>={min_body:.4f}) at {zone_source}",
        )

    # 4. Entry limit at the zone edge (manual limit order, never chase market)
    entry_limit = (zone_low + zone_high) / 2.0
    if direction == "long":
        swing_inside = min(c.low for c in recent_5m)
        stop_loss = swing_inside - cfg.sl_buffer_atr_5m * atr_5m
        risk = entry_limit - stop_loss
        if risk <= 0.0:
            return _wait(mode, "INVALID_SL", "Computed SL >= entry_limit for LONG")
        tp1 = entry_limit + cfg.tp1_r * risk
        tp2_cap = entry_limit + cfg.tp2_max_r * risk
        # Nearest opposing 15m swing high above TP1
        struct_15m = analyze_market_structure(candles_15m)
        opp_highs = [
            ev.level
            for ev in struct_15m.pivots
            if ev.event_type is StructureEventType.SWING_HIGH and ev.level > tp1
        ]
        tp2 = min(min(opp_highs), tp2_cap) if opp_highs else tp2_cap
    else:
        swing_inside = max(c.high for c in recent_5m)
        stop_loss = swing_inside + cfg.sl_buffer_atr_5m * atr_5m
        risk = stop_loss - entry_limit
        if risk <= 0.0:
            return _wait(mode, "INVALID_SL", "Computed SL <= entry_limit for SHORT")
        tp1 = entry_limit - cfg.tp1_r * risk
        tp2_cap = entry_limit - cfg.tp2_max_r * risk
        struct_15m = analyze_market_structure(candles_15m)
        opp_lows = [
            ev.level
            for ev in struct_15m.pivots
            if ev.event_type is StructureEventType.SWING_LOW and ev.level < tp1
        ]
        tp2 = max(max(opp_lows), tp2_cap) if opp_lows else tp2_cap

    return _apply_common_rules(
        mode=mode,
        direction=direction,
        canonical_symbol=canonical_symbol,
        zone_source=zone_source,
        trigger_type=trigger_type,
        entry_low=zone_low,
        entry_high=zone_high,
        entry_limit=entry_limit,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        decision_time_utc=decision_time_utc,
        config=cfg,
        orderbook=orderbook,
        rolling_median_spread_bps=rolling_median_spread_bps,
        funding_rate_8h=funding_rate_8h,
        cost_multiplier=cost_multiplier,
    )


def evaluate_range_fade_setup(
    *,
    canonical_symbol: str,
    dealing_range: ValidDealingRange,
    candles_5m: Sequence[MarketCandle],
    decision_time_utc: datetime,
    config: ScalpConfig | None = None,
    orderbook: OrderBookSnapshot | None = None,
    rolling_median_spread_bps: float | None = None,
    funding_rate_8h: float | None = None,
    cost_multiplier: float = 1.0,
) -> ScalpSetupResult:
    """Evaluate SCALP-SPEC-v1 §4.2 Range Fade setup (5m liquidity sweep of range edge)."""
    cfg = config or ScalpConfig()
    mode = ScalpMode.RANGE_FADE

    if len(candles_5m) < 20:
        return _wait(mode, "INSUFFICIENT_5M_BARS", "Need >=20 5m bars for Range Fade")

    tech_5m = compute_technical_metrics(candles_5m).metrics
    atr_5m = tech_5m.atr_14
    if atr_5m is None or atr_5m <= 0.0:
        return _wait(mode, "ATR_UNAVAILABLE", "ATR(14) unavailable on 5m")

    last = candles_5m[-1]
    rng_low = dealing_range.range_low
    rng_high = dealing_range.range_high
    midpoint = (rng_low + rng_high) / 2.0

    # Liquidity sweep of range low (wick below, close back inside) -> LONG fade
    swept_low = last.low < rng_low and last.close >= rng_low
    # Liquidity sweep of range high (wick above, close back inside) -> SHORT fade
    swept_high = last.high > rng_high and last.close <= rng_high

    if swept_low and not swept_high:
        direction: Literal["long", "short"] = "long"
        zone_low = rng_low
        zone_high = min(rng_low + 0.15 * atr_5m, last.close)
        if zone_high < zone_low:
            zone_high = zone_low
        entry_limit = (zone_low + zone_high) / 2.0
        stop_loss = last.low - cfg.sl_buffer_atr_5m * atr_5m
        risk = entry_limit - stop_loss
        if risk <= 0.0:
            return _wait(mode, "INVALID_SL", "Range fade LONG SL >= entry")
        tp1 = midpoint
        tp2_edge = rng_low + cfg.range_tp2_edge_frac * (rng_high - rng_low)
        tp2 = max(tp1, min(tp2_edge, entry_limit + cfg.tp2_max_r * risk))
        return _apply_common_rules(
            mode=mode,
            direction=direction,
            canonical_symbol=canonical_symbol,
            zone_source=f"range_low_{dealing_range.source_timeframe}",
            trigger_type="5m_liquidity_sweep_low",
            entry_low=zone_low,
            entry_high=zone_high,
            entry_limit=entry_limit,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            decision_time_utc=decision_time_utc,
            config=cfg,
            orderbook=orderbook,
            rolling_median_spread_bps=rolling_median_spread_bps,
            funding_rate_8h=funding_rate_8h,
            cost_multiplier=cost_multiplier,
        )

    if swept_high and not swept_low:
        direction = "short"
        zone_high = rng_high
        zone_low = max(rng_high - 0.15 * atr_5m, last.close)
        if zone_low > zone_high:
            zone_low = zone_high
        entry_limit = (zone_low + zone_high) / 2.0
        stop_loss = last.high + cfg.sl_buffer_atr_5m * atr_5m
        risk = stop_loss - entry_limit
        if risk <= 0.0:
            return _wait(mode, "INVALID_SL", "Range fade SHORT SL <= entry")
        tp1 = midpoint
        tp2_edge = rng_high - cfg.range_tp2_edge_frac * (rng_high - rng_low)
        tp2 = min(tp1, max(tp2_edge, entry_limit - cfg.tp2_max_r * risk))
        return _apply_common_rules(
            mode=mode,
            direction=direction,
            canonical_symbol=canonical_symbol,
            zone_source=f"range_high_{dealing_range.source_timeframe}",
            trigger_type="5m_liquidity_sweep_high",
            entry_low=zone_low,
            entry_high=zone_high,
            entry_limit=entry_limit,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            decision_time_utc=decision_time_utc,
            config=cfg,
            orderbook=orderbook,
            rolling_median_spread_bps=rolling_median_spread_bps,
            funding_rate_8h=funding_rate_8h,
            cost_multiplier=cost_multiplier,
        )

    return _wait(
        mode,
        "NO_RANGE_EDGE_SWEEP",
        f"5m candle did not sweep range edges ({rng_low:.4f}-{rng_high:.4f}) and close inside",
    )
