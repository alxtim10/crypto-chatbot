"""SC-02: Per-symbol, per-order-type execution cost model for `scalp_v1` (SCALP-SPEC-v1 §5).

Order types per spec:
- Entry: maker limit (zero spread crossing, zero L2 slippage, maker fee)
- TP1 / TP2: maker limit (zero spread crossing, zero L2 slippage, maker fee)
- SL / Time stop: taker market (half-spread + L2 depth slippage + taker fee)
- Funding: signed historical/current funding if an 8h settlement falls inside the holding window

Legacy flat cost model remains available via `ScalpCostConfig.legacy_flat_mode=True`
so historical/ablation trials can compare against the old flat assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.scalp.config import ScalpCostConfig
from crypto_chatbot.schemas.derivatives import OrderBookLevel, OrderBookSnapshot


class OrderType(StrEnum):
    """Order execution type for fee/spread/slippage accounting."""

    MAKER = "maker"
    TAKER = "taker"


class CostFlag(StrEnum):
    """Transparency flags attached when live L2 inputs are unavailable and fallbacks are used."""

    SPREAD_ESTIMATED = "SPREAD_ESTIMATED"
    SLIPPAGE_ESTIMATED = "SLIPPAGE_ESTIMATED"
    LEGACY_FLAT_MODEL = "LEGACY_FLAT_MODEL"


class CostLegBreakdown(BaseModel):
    """Single leg (entry or exit) execution cost breakdown in basis points."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    leg: Literal["entry", "exit_tp", "exit_sl", "exit_time_stop"]
    order_type: OrderType
    fee_bps: float = Field(ge=0.0)
    half_spread_bps: float = Field(ge=0.0)
    slippage_bps: float = Field(ge=0.0)

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.half_spread_bps + self.slippage_bps


class ScalpCostBreakdown(BaseModel):
    """Full round-trip expected execution cost breakdown for a scalp setup (SCALP-SPEC-v1 §5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str
    direction: Literal["long", "short"]
    vip_tier: str
    order_notional_usd: float = Field(gt=0.0)
    entry_leg: CostLegBreakdown
    tp_exit_leg: CostLegBreakdown
    sl_exit_leg: CostLegBreakdown
    funding_bps: float = 0.0
    funding_settlements_in_window: int = Field(default=0, ge=0)
    # Expected round-trip cost used for gating (conservative: maker entry + taker exit + funding)
    expected_round_trip_bps: float = Field(ge=0.0)
    # Maker-entry + maker-TP round-trip cost (used for net R:R to TP)
    tp_round_trip_bps: float = Field(ge=0.0)
    # Maker-entry + taker-SL round-trip cost (used for SL / break-even+costs)
    sl_round_trip_bps: float = Field(ge=0.0)
    risk_bps: float = Field(gt=0.0)
    cost_pct_of_r: float = Field(ge=0.0)
    passes_cost_gate: bool
    flags: tuple[CostFlag, ...] = ()


def compute_l2_half_spread_bps(orderbook: OrderBookSnapshot) -> float | None:
    """Return half-spread in bps from an L2 snapshot, or None if unavailable."""
    if orderbook.spread_bps is not None and orderbook.spread_bps >= 0.0:
        return orderbook.spread_bps / 2.0
    if orderbook.best_bid > 0.0 and orderbook.best_ask > 0.0 and orderbook.mid_price > 0.0:
        spread_bps = ((orderbook.best_ask - orderbook.best_bid) / orderbook.mid_price) * 10_000.0
        return max(0.0, spread_bps / 2.0)
    return None


def compute_l2_taker_slippage_bps(
    levels: Sequence[OrderBookLevel],
    *,
    notional_usd: float,
    reference_price: float,
) -> float | None:
    """Compute volume-weighted L2 market-impact slippage in bps beyond top-of-book.

    `levels` should be the book side being swept:
    - asks (ascending price) for a taker BUY (e.g., closing a SHORT)
    - bids (descending price) for a taker SELL (e.g., closing a LONG)

    Returns `None` if levels are empty or depth is insufficient to fill `notional_usd`.
    """
    if not levels or notional_usd <= 0.0 or reference_price <= 0.0:
        return None

    remaining_usd = notional_usd
    filled_qty = 0.0
    filled_usd = 0.0
    top_price = levels[0].price
    if top_price <= 0.0:
        return None

    for level in levels:
        if level.price <= 0.0 or level.quantity <= 0.0:
            continue
        level_usd = level.price * level.quantity
        take_usd = min(remaining_usd, level_usd)
        filled_usd += take_usd
        filled_qty += take_usd / level.price
        remaining_usd -= take_usd
        if remaining_usd <= 1e-9:
            break

    if remaining_usd > 1e-6 or filled_qty <= 0.0:
        return None

    vwap = filled_usd / filled_qty
    # Slippage beyond top-of-book (half-spread is accounted for separately)
    slippage_bps = (abs(vwap - top_price) / reference_price) * 10_000.0
    return max(0.0, slippage_bps)


def count_funding_settlements_in_window(
    start_utc: datetime,
    hold_minutes: int,
) -> int:
    """Count Gate 8h funding settlements (00:00, 08:00, 16:00 UTC) falling inside `(start, end]`."""
    start = start_utc.astimezone(UTC)
    end = start + timedelta(minutes=max(0, hold_minutes))
    if end <= start:
        return 0

    count = 0
    cursor_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor_day <= end + timedelta(days=1):
        for hour in (0, 8, 16):
            settlement = cursor_day.replace(hour=hour)
            if start < settlement <= end:
                count += 1
        cursor_day += timedelta(days=1)
    return count


def signed_funding_cost_bps(
    *,
    direction: Literal["long", "short"],
    funding_rate_8h: float | None,
    settlements_in_window: int,
) -> float:
    """Signed funding cost in bps (>0 means trader pays, <0 means trader receives).

    Positive funding rate: LONG pays (+), SHORT receives (-).
    Negative funding rate: LONG receives (-), SHORT pays (+).
    """
    if funding_rate_8h is None or settlements_in_window <= 0:
        return 0.0
    sign = 1.0 if direction == "long" else -1.0
    return sign * funding_rate_8h * 10_000.0 * settlements_in_window


def estimate_scalp_trade_costs(
    *,
    canonical_symbol: str,
    direction: Literal["long", "short"],
    entry_price: float,
    stop_loss: float,
    config: ScalpCostConfig | None = None,
    orderbook: OrderBookSnapshot | None = None,
    rolling_median_spread_bps: float | None = None,
    funding_rate_8h: float | None = None,
    decision_time_utc: datetime | None = None,
    max_hold_minutes: int = 90,
    cost_multiplier: float = 1.0,
) -> ScalpCostBreakdown:
    """Compute per-symbol, per-order-type execution cost breakdown (SCALP-SPEC-v1 §5).

    `cost_multiplier` supports cost sensitivity sweeps (1.0x, 1.5x, 2.0x) in backtests (§9.3).
    """
    cfg = config or ScalpCostConfig()
    if entry_price <= 0.0:
        raise ValueError("entry_price must be > 0")
    risk_bps = (abs(entry_price - stop_loss) / entry_price) * 10_000.0
    if risk_bps <= 0.0:
        raise ValueError("stop_loss must differ from entry_price")

    settlements = (
        count_funding_settlements_in_window(decision_time_utc, max_hold_minutes)
        if decision_time_utc is not None
        else 0
    )
    funding_bps = signed_funding_cost_bps(
        direction=direction,
        funding_rate_8h=funding_rate_8h,
        settlements_in_window=settlements,
    )

    if cfg.legacy_flat_mode:
        half_flat = (cfg.legacy_flat_round_trip_bps * cost_multiplier) / 2.0
        entry_leg = CostLegBreakdown(
            leg="entry",
            order_type=OrderType.MAKER,
            fee_bps=half_flat,
            half_spread_bps=0.0,
            slippage_bps=0.0,
        )
        tp_leg = CostLegBreakdown(
            leg="exit_tp",
            order_type=OrderType.MAKER,
            fee_bps=half_flat,
            half_spread_bps=0.0,
            slippage_bps=0.0,
        )
        sl_leg = CostLegBreakdown(
            leg="exit_sl",
            order_type=OrderType.TAKER,
            fee_bps=half_flat,
            half_spread_bps=0.0,
            slippage_bps=0.0,
        )
        total_rt = max(
            0.0, cfg.legacy_flat_round_trip_bps * cost_multiplier + max(0.0, funding_bps)
        )
        cost_pct = total_rt / risk_bps
        return ScalpCostBreakdown(
            canonical_symbol=canonical_symbol,
            direction=direction,
            vip_tier=cfg.vip_tier,
            order_notional_usd=cfg.order_notional_usd,
            entry_leg=entry_leg,
            tp_exit_leg=tp_leg,
            sl_exit_leg=sl_leg,
            funding_bps=funding_bps,
            funding_settlements_in_window=settlements,
            expected_round_trip_bps=total_rt,
            tp_round_trip_bps=max(0.0, entry_leg.total_bps + tp_leg.total_bps + funding_bps),
            sl_round_trip_bps=max(0.0, entry_leg.total_bps + sl_leg.total_bps + funding_bps),
            risk_bps=risk_bps,
            cost_pct_of_r=cost_pct,
            passes_cost_gate=cost_pct <= cfg.max_cost_pct_of_r,
            flags=(CostFlag.LEGACY_FLAT_MODEL,),
        )

    flags: list[CostFlag] = []

    # Half-spread for taker legs
    half_spread: float | None = None
    if orderbook is not None:
        half_spread = compute_l2_half_spread_bps(orderbook)
    if half_spread is None:
        flags.append(CostFlag.SPREAD_ESTIMATED)
        if rolling_median_spread_bps is not None and rolling_median_spread_bps >= 0.0:
            half_spread = rolling_median_spread_bps / 2.0
        else:
            half_spread = cfg.fallback_spread_bps / 2.0

    # Slippage for taker legs (SL exit sells into bids for LONG, buys into asks for SHORT)
    taker_slippage: float | None = None
    if orderbook is not None:
        book_side = orderbook.bids if direction == "long" else orderbook.asks
        taker_slippage = compute_l2_taker_slippage_bps(
            book_side,
            notional_usd=cfg.order_notional_usd,
            reference_price=orderbook.mid_price if orderbook.mid_price > 0 else entry_price,
        )
    if taker_slippage is None:
        flags.append(CostFlag.SLIPPAGE_ESTIMATED)
        taker_slippage = cfg.fallback_taker_slippage_bps

    entry_leg = CostLegBreakdown(
        leg="entry",
        order_type=OrderType.MAKER,
        fee_bps=cfg.maker_fee_bps * cost_multiplier,
        half_spread_bps=0.0,
        slippage_bps=0.0,
    )
    tp_leg = CostLegBreakdown(
        leg="exit_tp",
        order_type=OrderType.MAKER,
        fee_bps=cfg.maker_fee_bps * cost_multiplier,
        half_spread_bps=0.0,
        slippage_bps=0.0,
    )
    sl_leg = CostLegBreakdown(
        leg="exit_sl",
        order_type=OrderType.TAKER,
        fee_bps=cfg.taker_fee_bps * cost_multiplier,
        half_spread_bps=half_spread * cost_multiplier,
        slippage_bps=taker_slippage * cost_multiplier,
    )

    tp_rt = entry_leg.total_bps + tp_leg.total_bps + funding_bps
    sl_rt = entry_leg.total_bps + sl_leg.total_bps + funding_bps
    # Conservative expected cost for the 15% of 1R gate: maker entry + taker exit + positive funding
    expected_rt = max(0.0, entry_leg.total_bps + sl_leg.total_bps + max(0.0, funding_bps))
    cost_pct_of_r = expected_rt / risk_bps

    return ScalpCostBreakdown(
        canonical_symbol=canonical_symbol,
        direction=direction,
        vip_tier=cfg.vip_tier,
        order_notional_usd=cfg.order_notional_usd,
        entry_leg=entry_leg,
        tp_exit_leg=tp_leg,
        sl_exit_leg=sl_leg,
        funding_bps=funding_bps,
        funding_settlements_in_window=settlements,
        expected_round_trip_bps=expected_rt,
        tp_round_trip_bps=tp_rt,
        sl_round_trip_bps=sl_rt,
        risk_bps=risk_bps,
        cost_pct_of_r=cost_pct_of_r,
        passes_cost_gate=cost_pct_of_r <= cfg.max_cost_pct_of_r,
        flags=tuple(flags),
    )
