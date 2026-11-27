"""UTC-anchored OHLCV timeframe resampling and native-bar parity verification (D-03)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.market_data.base import DEFAULT_FRESHNESS_GRACE, OhlcvBatch
from crypto_chatbot.market_data.quality import is_candle_stale
from crypto_chatbot.schemas.market import (
    MarketCandle,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
    UtcDatetime,
)

RESAMPLE_MISMATCH_BPS_THRESHOLD = 5.0


def resample_candles(
    base_candles: Sequence[MarketCandle],
    *,
    target_timeframe: Timeframe,
    as_of: datetime | None = None,
    additional_flags: Sequence[QualityFlag] = (),
) -> OhlcvBatch:
    """Resample closed UTC candles upward (e.g. `1h -> 4h` / `1h -> 1d`).

    Strict invariants:
    - Anchored cleanly to UTC epoch multiples (00:00, 04:00, ...; 00:00Z daily).
    - Emits ONLY complete buckets where every sub-bar `[bucket_open + k*base_dur]` exists.
    - Enforces `bucket_close <= as_of` so forming higher-timeframe bars are never emitted.
    """
    if not base_candles:
        raise ValueError("base_candles cannot be empty")
    first = base_candles[0]
    base_tf = first.timeframe
    base_secs = int(base_tf.duration.total_seconds())
    target_secs = int(target_timeframe.duration.total_seconds())
    if target_secs <= base_secs or target_secs % base_secs != 0:
        raise ValueError(
            f"target_timeframe ({target_timeframe.value}) must be a strict multiple "
            f"of base timeframe ({base_tf.value})"
        )
    expected_per_bucket = target_secs // base_secs
    latest_received = max(c.received_time.astimezone(UTC) for c in base_candles)
    effective_as_of = (as_of or latest_received).astimezone(UTC)

    by_open: dict[datetime, MarketCandle] = {}
    for c in base_candles:
        if c.timeframe is not base_tf:
            raise ValueError("all base_candles must share the same timeframe")
        if not c.is_closed or c.close_time > effective_as_of:
            continue
        by_open[c.open_time.astimezone(UTC)] = c

    buckets: dict[datetime, list[MarketCandle]] = {}
    for open_t in sorted(by_open):
        epoch = int(open_t.timestamp())
        bucket_epoch = (epoch // target_secs) * target_secs
        bucket_open = datetime.fromtimestamp(bucket_epoch, tz=UTC)
        buckets.setdefault(bucket_open, []).append(by_open[open_t])

    resampled: list[MarketCandle] = []
    for bucket_open in sorted(buckets):
        bucket_close = bucket_open + target_timeframe.duration
        if bucket_close > effective_as_of:
            continue
        group = buckets[bucket_open]
        if len(group) != expected_per_bucket:
            continue
        expected_times = [bucket_open + k * base_tf.duration for k in range(expected_per_bucket)]
        if [c.open_time.astimezone(UTC) for c in group] != expected_times:
            continue

        b_first = group[0]
        b_last = group[-1]
        base_vol_sum = sum(c.base_volume for c in group)
        raw_contracts_sum = (
            sum(c.raw_contracts for c in group if c.raw_contracts is not None)
            if all(c.raw_contracts is not None for c in group)
            else None
        )
        quote_vol_sum = (
            sum(c.quote_volume for c in group if c.quote_volume is not None)
            if all(c.quote_volume is not None for c in group)
            else None
        )
        taker_sum = (
            sum(c.taker_buy_base_volume for c in group if c.taker_buy_base_volume is not None)
            if all(c.taker_buy_base_volume is not None for c in group)
            else None
        )
        if taker_sum is not None and taker_sum > base_vol_sum:
            taker_sum = base_vol_sum

        bar_flags: list[QualityFlag] = list(additional_flags)
        for c in group:
            for fl in c.quality_flags:
                if (
                    fl in {QualityFlag.BACKFILL, QualityFlag.QUOTE_VOL_ESTIMATED}
                    and fl not in bar_flags
                ):
                    bar_flags.append(fl)

        resampled.append(
            MarketCandle(
                provider=f"{b_first.provider}_resampled",
                venue=b_first.venue,
                canonical_symbol=b_first.canonical_symbol,
                provider_symbol=b_first.provider_symbol,
                market_type=b_first.market_type,
                timeframe=target_timeframe,
                open_time=bucket_open,
                close_time=bucket_close,
                received_time=latest_received,
                available_time=bucket_close,
                open=b_first.open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=b_last.close,
                raw_contracts=raw_contracts_sum,
                base_volume=base_vol_sum,
                quote_volume=quote_vol_sum,
                taker_buy_base_volume=taker_sum,
                base_unit=b_first.base_unit,
                quote_currency=b_first.quote_currency,
                is_closed=True,
                quality_flags=tuple(bar_flags),
            )
        )

    batch_flags: list[QualityFlag] = list(additional_flags)
    if resampled and is_candle_stale(
        resampled[-1].close_time,
        timeframe=target_timeframe,
        now=effective_as_of,
        freshness_grace=DEFAULT_FRESHNESS_GRACE,
    ):
        batch_flags.append(QualityFlag.STALE)

    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider=f"{first.provider}_resampled",
            venue=first.venue,
            market_type=first.market_type,
            base_unit=first.base_unit,
            quote_currency=first.quote_currency,
        ),
        canonical_symbol=first.canonical_symbol,
        provider_symbol=first.provider_symbol,
        timeframe=target_timeframe,
        received_time=latest_received,
        candles=tuple(resampled),
        quality_flags=tuple(dict.fromkeys(batch_flags)),
    )


class TimeframeParitySummary(BaseModel):
    """Comparison summary of resampled vs native Gate bars on overlapping timestamps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_timeframe: Timeframe
    base_timeframe: Timeframe
    matched_bars: int = Field(ge=0)
    max_close_diff_bps: float = Field(ge=0.0)
    mean_close_diff_bps: float = Field(ge=0.0)
    max_volume_rel_diff_pct: float = Field(ge=0.0)
    mismatch_flagged: bool
    quality_flags: tuple[QualityFlag, ...] = ()
    worst_bar_open_time: UtcDatetime | None = None


class ResampleParityReport(BaseModel):
    """Complete D-03 parity audit report artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: UtcDatetime
    venue: str
    canonical_symbol: str
    base_timeframe: Timeframe
    threshold_bps: float = RESAMPLE_MISMATCH_BPS_THRESHOLD
    targets: tuple[TimeframeParitySummary, ...]


def compare_resampled_with_native(
    resampled_candles: Sequence[MarketCandle],
    native_candles: Sequence[MarketCandle],
    *,
    base_timeframe: Timeframe,
    target_timeframe: Timeframe,
    max_bars: int = 200,
    mismatch_threshold_bps: float = RESAMPLE_MISMATCH_BPS_THRESHOLD,
) -> TimeframeParitySummary:
    """Compare resampled vs native bars over the last `max_bars` overlapping timestamps."""
    res_map = {c.open_time.astimezone(UTC): c for c in resampled_candles}
    nat_map = {c.open_time.astimezone(UTC): c for c in native_candles}
    overlap = sorted(set(res_map) & set(nat_map))[-max_bars:]
    if not overlap:
        return TimeframeParitySummary(
            target_timeframe=target_timeframe,
            base_timeframe=base_timeframe,
            matched_bars=0,
            max_close_diff_bps=0.0,
            mean_close_diff_bps=0.0,
            max_volume_rel_diff_pct=0.0,
            mismatch_flagged=True,
            quality_flags=(QualityFlag.RESAMPLE_MISMATCH,),
        )

    close_diffs_bps: list[float] = []
    vol_diffs_pct: list[float] = []
    worst_t: datetime | None = None
    worst_bps = -1.0
    for ts in overlap:
        rc = res_map[ts]
        nc = nat_map[ts]
        bps = abs(rc.close - nc.close) / nc.close * 10_000.0
        close_diffs_bps.append(bps)
        if bps > worst_bps:
            worst_bps = bps
            worst_t = ts
        if nc.base_volume > 0.0:
            vol_diffs_pct.append(abs(rc.base_volume - nc.base_volume) / nc.base_volume * 100.0)
        elif rc.base_volume == 0.0:
            vol_diffs_pct.append(0.0)

    max_bps = max(close_diffs_bps)
    mean_bps = sum(close_diffs_bps) / len(close_diffs_bps)
    max_vol_diff = max(vol_diffs_pct) if vol_diffs_pct else 0.0
    mismatch = max_bps > mismatch_threshold_bps
    flags = (QualityFlag.RESAMPLE_MISMATCH,) if mismatch else ()
    return TimeframeParitySummary(
        target_timeframe=target_timeframe,
        base_timeframe=base_timeframe,
        matched_bars=len(overlap),
        max_close_diff_bps=round(max_bps, 6),
        mean_close_diff_bps=round(mean_bps, 6),
        max_volume_rel_diff_pct=round(max_vol_diff, 6),
        mismatch_flagged=mismatch,
        quality_flags=flags,
        worst_bar_open_time=worst_t,
    )
