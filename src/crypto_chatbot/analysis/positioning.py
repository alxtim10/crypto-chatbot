"""Causal positioning sentiment panel derived from Gate `contract_stats` (S-01, FORWARD_ONLY)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from crypto_chatbot.schemas.analysis import PositioningPanel
from crypto_chatbot.schemas.derivatives import ContractStatsObservation

DEFAULT_POSITIONING_WARMUP_DAYS = 7.0
DEFAULT_POSITIONING_WINDOW_DAYS = 30


def _percentile_and_zscore(
    values: Sequence[float],
    latest: float | None,
) -> tuple[float | None, float | None]:
    if latest is None or len(values) < 2:
        return None, None
    count_le = sum(1 for v in values if v <= latest)
    pct = (count_le / len(values)) * 100.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    z = ((latest - mean) / std) if std > 1e-12 else 0.0
    return round(pct, 2), round(z, 3)


def _oi_24h_delta_pct(
    window_obs: Sequence[ContractStatsObservation],
    latest: ContractStatsObservation,
) -> float | None:
    if latest.open_interest_usd is None or latest.open_interest_usd <= 0.0:
        return None
    target_t = latest.observed_time - timedelta(hours=24)
    tolerance = timedelta(hours=2)
    best: ContractStatsObservation | None = None
    best_dist: timedelta | None = None
    for obs in window_obs:
        if obs.open_interest_usd is None or obs.open_interest_usd <= 0.0:
            continue
        dist = abs(obs.observed_time - target_t)
        if dist <= tolerance and (best_dist is None or dist < best_dist):
            best = obs
            best_dist = dist
    if best is None or best.open_interest_usd is None or best.open_interest_usd <= 0.0:
        return None
    return round(
        (latest.open_interest_usd - best.open_interest_usd) / best.open_interest_usd * 100.0,
        3,
    )


def _liquidation_24h_metrics(
    window_obs: Sequence[ContractStatsObservation],
    latest_time: datetime,
) -> tuple[float | None, float | None, float | None]:
    cutoff_24h = latest_time - timedelta(hours=24)
    recent = [obs for obs in window_obs if obs.observed_time > cutoff_24h]
    long_vals = [obs.long_liq_usd for obs in recent if obs.long_liq_usd is not None]
    short_vals = [obs.short_liq_usd for obs in recent if obs.short_liq_usd is not None]
    if not long_vals and not short_vals:
        return None, None, None
    long_sum = sum(long_vals)
    short_sum = sum(short_vals)
    total = long_sum + short_sum
    skew = ((long_sum - short_sum) / total) if total > 0.0 else None
    return (
        round(long_sum, 2),
        round(short_sum, 2),
        round(skew, 4) if skew is not None else None,
    )


def compute_positioning_panel(
    observations: Sequence[ContractStatsObservation],
    *,
    canonical_symbol: str,
    as_of: datetime,
    venue: str = "gateio",
    use_available_time: bool = True,
    warmup_days: float = DEFAULT_POSITIONING_WARMUP_DAYS,
    window_days: int = DEFAULT_POSITIONING_WINDOW_DAYS,
) -> PositioningPanel:
    """Build strictly causal `[FORWARD_ONLY]` positioning sentiment panel from `contract_stats`."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of_utc = as_of.astimezone(UTC)
    window_start = as_of_utc - timedelta(days=window_days)

    eligible = [
        obs
        for obs in observations
        if (obs.available_time if use_available_time else obs.observed_time) <= as_of_utc
        and obs.observed_time >= window_start
    ]
    eligible.sort(key=lambda o: o.observed_time)

    if not eligible:
        return PositioningPanel(
            parity_mode="FORWARD_ONLY",
            status="UNAVAILABLE",
            venue=venue,
            canonical_symbol=canonical_symbol,
            as_of=as_of_utc,
            history_days=0.0,
            observation_count=0,
            summary_label="tidak_tersedia",
            note="FORWARD_ONLY: belum ada histori contract_stats Gate untuk simbol ini.",
        )

    latest = eligible[-1]
    history_days = max(
        0.0,
        (latest.observed_time - eligible[0].observed_time).total_seconds() / 86_400.0,
    )
    oi_delta_pct = _oi_24h_delta_pct(eligible, latest)
    long_liq_24h, short_liq_24h, liq_skew = _liquidation_24h_metrics(eligible, latest.observed_time)

    if history_days + 1e-9 < warmup_days:
        return PositioningPanel(
            parity_mode="FORWARD_ONLY",
            status="WARMING_UP",
            venue=latest.venue,
            canonical_symbol=canonical_symbol,
            as_of=as_of_utc,
            history_days=round(history_days, 2),
            observation_count=len(eligible),
            top_lsr_account=latest.top_lsr_account,
            lsr_taker=latest.lsr_taker,
            lsr_account=latest.lsr_account,
            top_lsr_size=latest.top_lsr_size,
            open_interest_usd=latest.open_interest_usd,
            oi_24h_delta_pct=oi_delta_pct,
            long_liq_24h_usd=long_liq_24h,
            short_liq_24h_usd=short_liq_24h,
            liq_skew_24h=liq_skew,
            summary_label="warming_up",
            note=(
                f"FORWARD_ONLY: pemanasan historis ({history_days:.1f}/{warmup_days:.0f} hari); "
                "persentil/z-score 30d menunggu >= 7 hari rekaman contract_stats."
            ),
        )

    top_series = [o.top_lsr_account for o in eligible if o.top_lsr_account is not None]
    taker_series = [o.lsr_taker for o in eligible if o.lsr_taker is not None]
    top_pct, top_z = _percentile_and_zscore(top_series, latest.top_lsr_account)
    taker_pct, taker_z = _percentile_and_zscore(taker_series, latest.lsr_taker)

    if top_pct is not None and top_pct >= 80.0:
        label = "crowded_long"
    elif top_pct is not None and top_pct <= 20.0:
        label = "crowded_short"
    else:
        label = "seimbang"

    return PositioningPanel(
        parity_mode="FORWARD_ONLY",
        status="READY",
        venue=latest.venue,
        canonical_symbol=canonical_symbol,
        as_of=as_of_utc,
        history_days=round(history_days, 2),
        observation_count=len(eligible),
        top_lsr_account=latest.top_lsr_account,
        lsr_taker=latest.lsr_taker,
        lsr_account=latest.lsr_account,
        top_lsr_size=latest.top_lsr_size,
        open_interest_usd=latest.open_interest_usd,
        oi_24h_delta_pct=oi_delta_pct,
        long_liq_24h_usd=long_liq_24h,
        short_liq_24h_usd=short_liq_24h,
        liq_skew_24h=liq_skew,
        top_lsr_account_pct_30d=top_pct,
        top_lsr_account_z_30d=top_z,
        lsr_taker_pct_30d=taker_pct,
        lsr_taker_z_30d=taker_z,
        summary_label=label,
        note=(
            f"FORWARD_ONLY ({len(eligible)} observasi / {history_days:.1f} hari): "
            "pengukuran positioning historis Gate (bukan probabilitas arah)."
        ),
    )
