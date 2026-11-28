"""SC-09: Shared 1m settlement engine for `scalp_v1` (SCALP-SPEC-v1 §4, §5, §9.2).

Single source of truth used identically by:
- `settle-setups --strategy scalp_v1` (live / shadow forward settlement every 1 min)
- `run_scalp_backtest_1m()` (event-driven 1m backtest engine)

Guarantees that the exact same trade fixture produces the exact same outcome in both paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.scalp.costs import ScalpCostBreakdown
from crypto_chatbot.schemas.market import MarketCandle


class ScalpOutcomeStatus(StrEnum):
    """Canonical settlement states for `scalp_v1` trades."""

    PENDING = "pending"
    OPEN_FILLED = "entry_triggered"
    CANCELLED_TP1_BEFORE_FILL = "cancelled_tp1_before_fill"
    EXPIRED_UNFILLED = "expired_unfilled"
    SL_HIT = "sl_hit"
    TP1_THEN_BE = "tp1_then_be"
    TP1_THEN_TIME_STOP = "tp1_then_time_stop"
    TP2_HIT = "tp2_hit"
    TIME_STOP = "time_stop"


class ScalpTradeSpec(BaseModel):
    """Immutable trade specification passed into `settle_scalp_trade_1m`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: Literal["long", "short"]
    entry_limit: float = Field(gt=0.0)
    stop_loss: float = Field(gt=0.0)
    tp1: float = Field(gt=0.0)
    tp2: float = Field(gt=0.0)
    signal_time_utc: datetime
    tp1_close_frac: float = Field(default=0.60, gt=0.0, lt=1.0)
    human_delay_sec: int = Field(default=0, ge=0)
    entry_valid_minutes: int = Field(default=15, ge=1)
    max_hold_minutes: int = Field(default=90, ge=1)
    tick_size: float = Field(default=1e-6, gt=0.0)
    # Cost rates in bps per leg
    entry_maker_bps: float = Field(default=2.0, ge=0.0)
    tp_maker_bps: float = Field(default=2.0, ge=0.0)
    sl_taker_bps: float = Field(default=7.0, ge=0.0)
    funding_bps: float = 0.0

    @classmethod
    def from_cost_breakdown(
        cls,
        *,
        direction: Literal["long", "short"],
        entry_limit: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        signal_time_utc: datetime,
        cost: ScalpCostBreakdown,
        tp1_close_frac: float = 0.60,
        human_delay_sec: int = 0,
        entry_valid_minutes: int = 15,
        max_hold_minutes: int = 90,
        tick_size: float = 1e-6,
    ) -> ScalpTradeSpec:
        return cls(
            direction=direction,
            entry_limit=entry_limit,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            signal_time_utc=signal_time_utc,
            tp1_close_frac=tp1_close_frac,
            human_delay_sec=human_delay_sec,
            entry_valid_minutes=entry_valid_minutes,
            max_hold_minutes=max_hold_minutes,
            tick_size=tick_size,
            entry_maker_bps=cost.entry_leg.total_bps,
            tp_maker_bps=cost.tp_exit_leg.total_bps,
            sl_taker_bps=cost.sl_exit_leg.total_bps,
            funding_bps=cost.funding_bps,
        )


class ScalpSettlementOutcome(BaseModel):
    """Deterministic settlement result for a single `scalp_v1` trade."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ScalpOutcomeStatus
    filled: bool
    fill_time_utc: datetime | None = None
    fill_price: float | None = None
    tp1_hit: bool = False
    tp1_time_utc: datetime | None = None
    exit_time_utc: datetime | None = None
    exit_price: float | None = None
    gross_r: float = 0.0
    cost_r: float = 0.0
    net_r: float = 0.0
    bars_held_1m: int = 0
    is_final: bool = False


def settle_scalp_trade_1m(
    spec: ScalpTradeSpec,
    candles_1m: Sequence[MarketCandle],
    *,
    as_of_utc: datetime | None = None,
) -> ScalpSettlementOutcome:
    """Settle a `scalp_v1` trade deterministically on closed 1m candles (§4, §5, §9.2).

    Rules:
    1. Order becomes active at `signal_time + human_delay_sec` and expires at
       `signal_time + entry_valid_minutes`.
    2. Before fill: if TP1 is touched first -> `CANCELLED_TP1_BEFORE_FILL`.
    3. Maker fill rule: a 1m bar fills the limit order ONLY if it trades THROUGH the limit
       by >= `tick_size` (touch alone is NOT a fill).
    4. Intrabar order on every 1m bar: SL checked BEFORE TP (conservative).
    5. Gap through SL fills at `bar.open`.
    6. TP1 closes `tp1_close_frac` (60%) as maker and moves SL to break-even + costs.
    7. TP2 closes remaining 40% as maker.
    8. Time stop closes remaining position at `bar.close` as taker once
       `fill_time + max_hold_minutes` is reached.
    """
    sig_time = spec.signal_time_utc.astimezone(UTC)
    order_active_at = sig_time + timedelta(seconds=spec.human_delay_sec)
    entry_expiry_at = sig_time + timedelta(minutes=spec.entry_valid_minutes)

    risk_price = abs(spec.entry_limit - spec.stop_loss)
    if risk_price <= 0.0:
        raise ValueError("entry_limit and stop_loss must differ")

    is_long = spec.direction == "long"
    # Break-even + round-trip costs (maker entry + taker BE stop + positive funding)
    be_cost_bps = spec.entry_maker_bps + spec.sl_taker_bps + max(0.0, spec.funding_bps)
    be_offset = (be_cost_bps / 10_000.0) * spec.entry_limit
    be_stop = spec.entry_limit + be_offset if is_long else spec.entry_limit - be_offset

    # Filter strictly closed 1m bars ordered chronologically
    cutoff = as_of_utc.astimezone(UTC) if as_of_utc is not None else None
    bars = sorted(
        (
            c
            for c in candles_1m
            if c.is_closed
            and c.close_time.astimezone(UTC) > sig_time
            and (cutoff is None or c.close_time.astimezone(UTC) <= cutoff)
        ),
        key=lambda c: c.open_time.astimezone(UTC),
    )

    filled = False
    fill_time: datetime | None = None
    tp1_hit = False
    tp1_time: datetime | None = None
    active_sl = spec.stop_loss
    time_stop_at: datetime | None = None
    bars_held = 0

    def _cost_in_r(exit_leg_bps: float, frac: float) -> float:
        leg_bps = spec.entry_maker_bps + exit_leg_bps + spec.funding_bps
        cost_price = (leg_bps / 10_000.0) * spec.entry_limit
        return frac * (cost_price / risk_price)

    def _gross_in_r(exit_price: float, frac: float) -> float:
        pnl = (exit_price - spec.entry_limit) if is_long else (spec.entry_limit - exit_price)
        return frac * (pnl / risk_price)

    for bar in bars:
        bar_open_t = bar.open_time.astimezone(UTC)
        bar_close_t = bar.close_time.astimezone(UTC)

        if not filled:
            # Check expiry before processing bars strictly past expiry
            if bar_open_t >= entry_expiry_at:
                return ScalpSettlementOutcome(
                    status=ScalpOutcomeStatus.EXPIRED_UNFILLED,
                    filled=False,
                    exit_time_utc=entry_expiry_at,
                    is_final=True,
                )

            # Cancel if TP1 is touched before fill (even during human delay)
            tp1_touched = (bar.high >= spec.tp1) if is_long else (bar.low <= spec.tp1)
            # Only eligible to fill once bar_close_t > order_active_at
            can_fill_now = bar_close_t > order_active_at

            if can_fill_now:
                traded_through = (
                    (bar.low <= spec.entry_limit - spec.tick_size)
                    if is_long
                    else (bar.high >= spec.entry_limit + spec.tick_size)
                )
                if traded_through:
                    filled = True
                    fill_time = max(bar_open_t, order_active_at)
                    time_stop_at = fill_time + timedelta(minutes=spec.max_hold_minutes)
                    bars_held = 1
                    # Conservative intrabar on fill bar: check SL first
                    sl_hit_now = (bar.low <= active_sl) if is_long else (bar.high >= active_sl)
                    if sl_hit_now:
                        gross = _gross_in_r(active_sl, 1.0)
                        cost_r = _cost_in_r(spec.sl_taker_bps, 1.0)
                        return ScalpSettlementOutcome(
                            status=ScalpOutcomeStatus.SL_HIT,
                            filled=True,
                            fill_time_utc=fill_time,
                            fill_price=spec.entry_limit,
                            exit_time_utc=bar_close_t,
                            exit_price=active_sl,
                            gross_r=gross,
                            cost_r=cost_r,
                            net_r=gross - cost_r,
                            bars_held_1m=bars_held,
                            is_final=True,
                        )
                    continue

            if tp1_touched:
                return ScalpSettlementOutcome(
                    status=ScalpOutcomeStatus.CANCELLED_TP1_BEFORE_FILL,
                    filled=False,
                    exit_time_utc=bar_close_t,
                    is_final=True,
                )
            continue

        # Already filled: evaluate intrabar SL BEFORE TP (conservative)
        bars_held += 1
        rem_frac = 1.0 - spec.tp1_close_frac if tp1_hit else 1.0

        # 1. Gap through SL fills at bar.open
        gap_through_sl = (bar.open <= active_sl) if is_long else (bar.open >= active_sl)
        sl_hit_now = gap_through_sl or (
            (bar.low <= active_sl) if is_long else (bar.high >= active_sl)
        )
        if sl_hit_now:
            exit_px = bar.open if gap_through_sl else active_sl
            if tp1_hit:
                gross = _gross_in_r(spec.tp1, spec.tp1_close_frac) + _gross_in_r(exit_px, rem_frac)
                cost_r = _cost_in_r(spec.tp_maker_bps, spec.tp1_close_frac) + _cost_in_r(
                    spec.sl_taker_bps, rem_frac
                )
                return ScalpSettlementOutcome(
                    status=ScalpOutcomeStatus.TP1_THEN_BE,
                    filled=True,
                    fill_time_utc=fill_time,
                    fill_price=spec.entry_limit,
                    tp1_hit=True,
                    tp1_time_utc=tp1_time,
                    exit_time_utc=bar_close_t,
                    exit_price=exit_px,
                    gross_r=gross,
                    cost_r=cost_r,
                    net_r=gross - cost_r,
                    bars_held_1m=bars_held,
                    is_final=True,
                )
            gross = _gross_in_r(exit_px, 1.0)
            cost_r = _cost_in_r(spec.sl_taker_bps, 1.0)
            return ScalpSettlementOutcome(
                status=ScalpOutcomeStatus.SL_HIT,
                filled=True,
                fill_time_utc=fill_time,
                fill_price=spec.entry_limit,
                exit_time_utc=bar_close_t,
                exit_price=exit_px,
                gross_r=gross,
                cost_r=cost_r,
                net_r=gross - cost_r,
                bars_held_1m=bars_held,
                is_final=True,
            )

        # 2. Check TP1 if not yet hit
        if not tp1_hit:
            tp1_reached = (bar.high >= spec.tp1) if is_long else (bar.low <= spec.tp1)
            if tp1_reached:
                tp1_hit = True
                tp1_time = bar_close_t
                active_sl = be_stop
                rem_frac = 1.0 - spec.tp1_close_frac

        # 3. Check TP2 (only if TP1 is hit)
        if tp1_hit:
            tp2_reached = (bar.high >= spec.tp2) if is_long else (bar.low <= spec.tp2)
            if tp2_reached:
                gross = _gross_in_r(spec.tp1, spec.tp1_close_frac) + _gross_in_r(spec.tp2, rem_frac)
                cost_r = _cost_in_r(spec.tp_maker_bps, spec.tp1_close_frac) + _cost_in_r(
                    spec.tp_maker_bps, rem_frac
                )
                return ScalpSettlementOutcome(
                    status=ScalpOutcomeStatus.TP2_HIT,
                    filled=True,
                    fill_time_utc=fill_time,
                    fill_price=spec.entry_limit,
                    tp1_hit=True,
                    tp1_time_utc=tp1_time,
                    exit_time_utc=bar_close_t,
                    exit_price=spec.tp2,
                    gross_r=gross,
                    cost_r=cost_r,
                    net_r=gross - cost_r,
                    bars_held_1m=bars_held,
                    is_final=True,
                )

        # 4. Time stop check (90 min after fill -> close remaining at market taker)
        if time_stop_at is not None and bar_close_t >= time_stop_at:
            exit_px = bar.close
            if tp1_hit:
                gross = _gross_in_r(spec.tp1, spec.tp1_close_frac) + _gross_in_r(exit_px, rem_frac)
                cost_r = _cost_in_r(spec.tp_maker_bps, spec.tp1_close_frac) + _cost_in_r(
                    spec.sl_taker_bps, rem_frac
                )
                return ScalpSettlementOutcome(
                    status=ScalpOutcomeStatus.TP1_THEN_TIME_STOP,
                    filled=True,
                    fill_time_utc=fill_time,
                    fill_price=spec.entry_limit,
                    tp1_hit=True,
                    tp1_time_utc=tp1_time,
                    exit_time_utc=bar_close_t,
                    exit_price=exit_px,
                    gross_r=gross,
                    cost_r=cost_r,
                    net_r=gross - cost_r,
                    bars_held_1m=bars_held,
                    is_final=True,
                )
            gross = _gross_in_r(exit_px, 1.0)
            cost_r = _cost_in_r(spec.sl_taker_bps, 1.0)
            return ScalpSettlementOutcome(
                status=ScalpOutcomeStatus.TIME_STOP,
                filled=True,
                fill_time_utc=fill_time,
                fill_price=spec.entry_limit,
                exit_time_utc=bar_close_t,
                exit_price=exit_px,
                gross_r=gross,
                cost_r=cost_r,
                net_r=gross - cost_r,
                bars_held_1m=bars_held,
                is_final=True,
            )

    # Not finalized yet by available bars
    if not filled:
        if cutoff is not None and cutoff >= entry_expiry_at:
            return ScalpSettlementOutcome(
                status=ScalpOutcomeStatus.EXPIRED_UNFILLED,
                filled=False,
                exit_time_utc=entry_expiry_at,
                is_final=True,
            )
        return ScalpSettlementOutcome(
            status=ScalpOutcomeStatus.PENDING,
            filled=False,
            is_final=False,
        )

    return ScalpSettlementOutcome(
        status=ScalpOutcomeStatus.OPEN_FILLED,
        filled=True,
        fill_time_utc=fill_time,
        fill_price=spec.entry_limit,
        tp1_hit=tp1_hit,
        tp1_time_utc=tp1_time,
        bars_held_1m=bars_held,
        is_final=False,
    )
