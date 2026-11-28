"""SC-06: Blocking filters for `scalp_v1` (SCALP-SPEC-v1 §6).

Every filter returns an explicit `FilterReasonCode` when triggered, forcing `NO_TRADE`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.scalp.config import MacroEventWindow, ScalpFilterConfig


class FilterReasonCode(StrEnum):
    """Explicit machine-readable reason codes for every blocking filter (§6)."""

    FUNDING_WINDOW = "FILTER_FUNDING_WINDOW"
    MACRO_EVENT = "FILTER_MACRO_EVENT"
    LIQUIDATION_CASCADE = "FILTER_LIQUIDATION_CASCADE"
    CROWDED_FUNDING = "FILTER_CROWDED_FUNDING"
    NEWS_BURST = "FILTER_NEWS_BURST"
    DAILY_STOP = "FILTER_DAILY_STOP"
    LOSS_COOLDOWN = "FILTER_LOSS_COOLDOWN"
    MAX_CONCURRENCY = "FILTER_MAX_CONCURRENCY"


class ScalpFilterInputs(BaseModel):
    """Point-in-time inputs needed to evaluate all 8 blocking filters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_time_utc: datetime
    canonical_symbol: str
    direction: Literal["long", "short"] | None = None
    # Scheduled macro events
    macro_events: tuple[MacroEventWindow, ...] = ()
    # Liquidation cascade: recent 5m max(long_liq, short_liq) in last `liq_cascade_block_bars` bars
    recent_5m_liq_usd: tuple[float, ...] = ()
    liq_p99_30d_usd: float | None = None
    # Crowded funding: latest signed 8h funding rate (decimal, e.g. 0.0006 = 0.06%)
    funding_rate_8h: float | None = None
    # News burst: articles mentioning symbol in last 30m vs 30d P99 threshold
    news_count_30m: int = Field(default=0, ge=0)
    news_p99_30d_threshold: float | None = None
    # Trader risk-state filters (taken trades only)
    realized_r_today_utc: float = 0.0
    last_sl_time_for_symbol_utc: datetime | None = None
    open_taken_trades_count: int = Field(default=0, ge=0)


class FilterHit(BaseModel):
    """Single triggered filter with reason code and human-readable explanation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: FilterReasonCode
    detail: str


class ScalpFilterResult(BaseModel):
    """Aggregated filter evaluation result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocked: bool
    hits: tuple[FilterHit, ...] = ()

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(hit.code.value for hit in self.hits)


def is_in_funding_window(ts_utc: datetime, window_minutes: int = 10) -> bool:
    """Return True if `ts_utc` is within ±`window_minutes` of 00:00, 08:00, or 16:00 UTC."""
    if window_minutes <= 0:
        return False
    ts = ts_utc.astimezone(UTC)
    day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    window = timedelta(minutes=window_minutes)
    # Check yesterday 16:00, today 00/08/16, tomorrow 00:00 to handle midnight wrap cleanly
    candidates = (
        day_start - timedelta(hours=8),
        day_start,
        day_start + timedelta(hours=8),
        day_start + timedelta(hours=16),
        day_start + timedelta(hours=24),
    )
    return any(abs(ts - anchor) <= window for anchor in candidates)


def evaluate_scalp_filters(
    inputs: ScalpFilterInputs,
    *,
    config: ScalpFilterConfig | None = None,
) -> ScalpFilterResult:
    """Evaluate all 8 blocking filters from SCALP-SPEC-v1 §6."""
    cfg = config or ScalpFilterConfig()
    ts = inputs.decision_time_utc.astimezone(UTC)
    hits: list[FilterHit] = []

    # 1. Funding window (±10m around 00:00, 08:00, 16:00 UTC)
    if is_in_funding_window(ts, cfg.funding_window_minutes):
        hits.append(
            FilterHit(
                code=FilterReasonCode.FUNDING_WINDOW,
                detail=f"Within ±{cfg.funding_window_minutes}m of 8h funding settlement",
            )
        )

    # 2. Scheduled macro events (-15m / +30m from config/event_calendar.yaml)
    for event in inputs.macro_events:
        if event.blocks(
            ts,
            default_pre_minutes=cfg.macro_pre_minutes,
            default_post_minutes=cfg.macro_post_minutes,
        ):
            hits.append(
                FilterHit(
                    code=FilterReasonCode.MACRO_EVENT,
                    detail=f"Inside scheduled macro event window: {event.name}",
                )
            )
            break

    # 3. Liquidation cascade (5m liq >= 30d P99 within last 3 bars)
    if inputs.liq_p99_30d_usd is not None and inputs.liq_p99_30d_usd > 0.0:
        window_bars = inputs.recent_5m_liq_usd[-cfg.liq_cascade_block_bars :]
        if any(val >= inputs.liq_p99_30d_usd for val in window_bars):
            peak = max(window_bars)
            hits.append(
                FilterHit(
                    code=FilterReasonCode.LIQUIDATION_CASCADE,
                    detail=(
                        f"5m liquidation {peak:,.0f} USD >= 30d P99 "
                        f"{inputs.liq_p99_30d_usd:,.0f} USD "
                        f"(block {cfg.liq_cascade_block_bars} bars)"
                    ),
                )
            )

    # 4. Crowded funding (|funding 8h| >= 0.05% blocks entries in crowded direction)
    if inputs.funding_rate_8h is not None and inputs.direction is not None:
        fr = inputs.funding_rate_8h
        if fr >= cfg.crowded_funding_abs_8h and inputs.direction == "long":
            hits.append(
                FilterHit(
                    code=FilterReasonCode.CROWDED_FUNDING,
                    detail=f"Crowded positive funding {fr * 100:.3f}% blocks LONG",
                )
            )
        elif fr <= -cfg.crowded_funding_abs_8h and inputs.direction == "short":
            hits.append(
                FilterHit(
                    code=FilterReasonCode.CROWDED_FUNDING,
                    detail=f"Crowded negative funding {fr * 100:.3f}% blocks SHORT",
                )
            )

    # 5. News burst (>= N articles in last 30m where N = 30d P99)
    if (
        inputs.news_p99_30d_threshold is not None
        and inputs.news_p99_30d_threshold > 0.0
        and inputs.news_count_30m >= inputs.news_p99_30d_threshold
    ):
        hits.append(
            FilterHit(
                code=FilterReasonCode.NEWS_BURST,
                detail=(
                    f"News burst: {inputs.news_count_30m} articles in "
                    f"{cfg.news_burst_window_minutes}m >= P99 ({inputs.news_p99_30d_threshold:.1f})"
                ),
            )
        )

    # 6. Daily stop (realized <= -3R today UTC on taken trades)
    if inputs.realized_r_today_utc <= cfg.daily_stop_r:
        hits.append(
            FilterHit(
                code=FilterReasonCode.DAILY_STOP,
                detail=(
                    f"Daily stop reached: realized {inputs.realized_r_today_utc:.2f}R "
                    f"<= {cfg.daily_stop_r:.2f}R today (UTC)"
                ),
            )
        )

    # 7. Loss cooldown (after SL on symbol -> symbol blocked 30 min)
    if inputs.last_sl_time_for_symbol_utc is not None and cfg.loss_cooldown_minutes > 0:
        last_sl = inputs.last_sl_time_for_symbol_utc.astimezone(UTC)
        elapsed = ts - last_sl
        cooldown = timedelta(minutes=cfg.loss_cooldown_minutes)
        if timedelta(0) <= elapsed < cooldown:
            rem_min = (cooldown - elapsed).total_seconds() / 60.0
            hits.append(
                FilterHit(
                    code=FilterReasonCode.LOSS_COOLDOWN,
                    detail=(
                        f"Symbol {inputs.canonical_symbol} in {cfg.loss_cooldown_minutes}m "
                        f"SL cooldown ({rem_min:.1f}m remaining)"
                    ),
                )
            )

    # 8. Max concurrency (>= 2 open taken trades -> no new alerts)
    if inputs.open_taken_trades_count >= cfg.max_open_taken_trades:
        hits.append(
            FilterHit(
                code=FilterReasonCode.MAX_CONCURRENCY,
                detail=(
                    f"Max concurrency reached: {inputs.open_taken_trades_count} open taken trades "
                    f">= {cfg.max_open_taken_trades}"
                ),
            )
        )

    return ScalpFilterResult(blocked=bool(hits), hits=tuple(hits))


def compute_percentile_threshold(values: Sequence[float], percentile: float = 99.0) -> float | None:
    """Compute empirical percentile threshold from non-empty historical values."""
    clean = sorted(float(v) for v in values if v >= 0.0)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (percentile / 100.0) * (len(clean) - 1)
    lo = int(rank)
    hi = min(len(clean) - 1, lo + 1)
    frac = rank - lo
    return clean[lo] + frac * (clean[hi] - clean[lo])
