"""Deterministic derivatives, order-book liquidity, and cross-source checks (Phase 9)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.market_data.derivatives_quality import (
    CROSS_SOURCE_MAX_TIME_DIFF,
    PRICE_MISMATCH_BLOCK_BPS,
    PRICE_MISMATCH_DEGRADE_BPS,
    check_cross_source_price,
)
from crypto_chatbot.schemas.analysis import DerivativesMetrics
from crypto_chatbot.schemas.derivatives import (
    CrossSourcePriceCheck,
    DerivativesContext,
    LiquidationSide,
    OrderBookDepthBand,
)
from crypto_chatbot.schemas.market import QualityFlag, Timeframe

DERIVATIVES_PARAMETERS: dict[str, int | float | str] = {
    "funding_norm_interval_hours": 8,
    "orderbook_depth_band_tight_bps": 10,
    "orderbook_depth_band_wide_bps": 50,
    "price_mismatch_degrade_bps": PRICE_MISMATCH_DEGRADE_BPS,
    "price_mismatch_block_bps": PRICE_MISMATCH_BLOCK_BPS,
    "cross_source_max_time_diff_seconds": int(CROSS_SOURCE_MAX_TIME_DIFF.total_seconds()),
}


def _require_utc(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    return dt.astimezone(UTC)


class DerivativesAnalysisResult(BaseModel):
    """Deterministic output of analyzing `DerivativesContext`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: DerivativesMetrics | None = None
    quality_flags: tuple[QualityFlag, ...] = ()
    limitations: tuple[str, ...] = ()
    cross_source_checks: tuple[CrossSourcePriceCheck, ...] = ()


def compute_derivatives_metrics(
    context: DerivativesContext,
    *,
    spot_symbol: str,
    timeframe: Timeframe,
    as_of: datetime,
    use_available_time: bool = False,
) -> DerivativesAnalysisResult:
    """Compute causal derivatives, liquidity, and cross-source price-check metrics."""
    del spot_symbol
    cutoff = _require_utc(as_of, "as_of")
    flags: list[QualityFlag] = []
    limitations: list[str] = []

    # 1. Perpetual funding rate & basis
    funding_venue: str | None = None
    funding_symbol: str | None = None
    funding_rate: float | None = None
    funding_interval_hours: float | None = None
    funding_rate_per_8h: float | None = None
    funding_rate_annualized: float | None = None
    next_funding_time: datetime | None = None
    mark_price: float | None = None
    index_price: float | None = None
    mark_index_basis_bps: float | None = None
    funding_obs = None

    f_batch = context.funding
    if f_batch is not None:
        raw_obs = f_batch.observation
        blocked_by_time = (
            raw_obs is not None and use_available_time and raw_obs.available_time > cutoff
        )
        funding_bad = (
            raw_obs is None
            or f_batch.has_provider_error
            or QualityFlag.STALE in f_batch.quality_flags
            or QualityFlag.DERIVATIVES_UNAVAILABLE in f_batch.quality_flags
            or QualityFlag.SYMBOL_MISMATCH in f_batch.quality_flags
            or blocked_by_time
        )
        if raw_obs is None or funding_bad:
            if QualityFlag.SYMBOL_MISMATCH in f_batch.quality_flags:
                flags.append(QualityFlag.SYMBOL_MISMATCH)
            if QualityFlag.STALE in f_batch.quality_flags:
                flags.append(QualityFlag.STALE)
            flags.append(QualityFlag.DERIVATIVES_UNAVAILABLE)
            limitations.append(
                f"Data funding rate ({f_batch.canonical_symbol} @ {f_batch.venue}) "
                "tidak tersedia atau tidak memenuhi syarat; nilai tidak direkayasa."
            )
        else:
            funding_obs = raw_obs
            funding_venue = raw_obs.venue
            funding_symbol = raw_obs.canonical_symbol
            funding_rate = raw_obs.funding_rate
            funding_interval_hours = raw_obs.funding_interval_hours
            funding_rate_per_8h = raw_obs.rate_per_8h
            funding_rate_annualized = raw_obs.annualized_rate
            next_funding_time = raw_obs.next_funding_time
            mark_price = raw_obs.mark_price
            index_price = raw_obs.index_price
            if mark_price is not None and index_price is not None and index_price > 0:
                mark_index_basis_bps = (mark_price - index_price) / index_price * 10_000.0

    # 2. Perpetual open interest & causal change
    oi_venue: str | None = None
    oi_symbol: str | None = None
    oi_base: float | None = None
    oi_base_unit: str | None = None
    oi_quote: float | None = None
    oi_quote_currency: str | None = None
    oi_change_pct: float | None = None
    oi_change_window: str | None = None

    oi_batch = context.open_interest
    if oi_batch is not None:
        latest_oi = oi_batch.latest
        oi_blocked_by_time = (
            latest_oi is not None and use_available_time and latest_oi.available_time > cutoff
        )
        oi_bad = (
            latest_oi is None
            or oi_batch.has_provider_error
            or QualityFlag.UNIT_MISMATCH in oi_batch.quality_flags
            or QualityFlag.DERIVATIVES_UNAVAILABLE in oi_batch.quality_flags
            or QualityFlag.SYMBOL_MISMATCH in oi_batch.quality_flags
            or oi_blocked_by_time
        )
        if latest_oi is None or oi_bad:
            if QualityFlag.UNIT_MISMATCH in oi_batch.quality_flags:
                flags.append(QualityFlag.UNIT_MISMATCH)
            if QualityFlag.SYMBOL_MISMATCH in oi_batch.quality_flags:
                flags.append(QualityFlag.SYMBOL_MISMATCH)
            flags.append(QualityFlag.DERIVATIVES_UNAVAILABLE)
            limitations.append(
                f"Data open interest ({oi_batch.canonical_symbol} @ {oi_batch.venue}) "
                "tidak tersedia atau gagal validasi unit; nilai tidak direkayasa."
            )
        else:
            oi_venue = latest_oi.venue
            oi_symbol = latest_oi.canonical_symbol
            oi_base = latest_oi.open_interest_base
            oi_base_unit = latest_oi.base_unit
            oi_quote = latest_oi.open_interest_quote
            if oi_quote is None and oi_base is not None and mark_price is not None:
                oi_quote = oi_base * mark_price
            oi_quote_currency = latest_oi.quote_currency

            causal_history = [
                obs
                for obs in oi_batch.history
                if obs.observed_time < latest_oi.observed_time
                and (not use_available_time or obs.available_time <= cutoff)
            ]
            causal_history.sort(key=lambda obs: obs.observed_time)
            if causal_history:
                target_time = latest_oi.observed_time - timeframe.duration
                prior_candidates = [
                    obs for obs in causal_history if obs.observed_time <= target_time
                ]
                baseline = prior_candidates[-1] if prior_candidates else causal_history[0]
                if latest_oi.open_interest_quote is not None and (
                    baseline.open_interest_quote is not None
                ):
                    curr_val: float | None = latest_oi.open_interest_quote
                    prev_val: float | None = baseline.open_interest_quote
                else:
                    curr_val = latest_oi.open_interest_base
                    prev_val = baseline.open_interest_base
                if curr_val is not None and prev_val is not None and prev_val > 0:
                    oi_change_pct = (curr_val - prev_val) / prev_val
                    oi_change_window = timeframe.value

    # 3. Order book spread, depth bands, and imbalance
    ob_venue: str | None = None
    ob_symbol: str | None = None
    ob_market_type = None
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread_abs: float | None = None
    spread_bps: float | None = None
    depth_10_bid: float | None = None
    depth_10_ask: float | None = None
    imb_10: float | None = None
    depth_50_bid: float | None = None
    depth_50_ask: float | None = None
    imb_50: float | None = None
    resync_count = 0
    ob_snap = None

    ob_batch = context.order_book
    if ob_batch is not None:
        resync_count = ob_batch.resync_count
        raw_book = ob_batch.snapshot
        ob_blocked_by_time = (
            raw_book is not None and use_available_time and raw_book.available_time > cutoff
        )
        ob_bad = (
            raw_book is None
            or ob_batch.has_provider_error
            or QualityFlag.ORDERBOOK_UNAVAILABLE in ob_batch.quality_flags
            or QualityFlag.SYMBOL_MISMATCH in ob_batch.quality_flags
            or ob_blocked_by_time
        )
        if QualityFlag.SEQUENCE_GAP in ob_batch.quality_flags:
            flags.append(QualityFlag.SEQUENCE_GAP)
        if raw_book is None or ob_bad:
            if QualityFlag.SYMBOL_MISMATCH in ob_batch.quality_flags:
                flags.append(QualityFlag.SYMBOL_MISMATCH)
            flags.append(QualityFlag.ORDERBOOK_UNAVAILABLE)
            limitations.append(
                f"Data order book ({ob_batch.canonical_symbol} @ {ob_batch.venue}) "
                "tidak tersedia atau mengalami kegagalan sinkronisasi; metrik spread/kedalaman "
                "tidak direkayasa."
            )
        else:
            ob_snap = raw_book
            resync_count = max(resync_count, raw_book.resync_count)
            ob_venue = raw_book.venue
            ob_symbol = raw_book.canonical_symbol
            ob_market_type = raw_book.market_type
            best_bid = raw_book.best_bid
            best_ask = raw_book.best_ask
            mid_price = raw_book.mid_price
            spread_abs = raw_book.spread_abs
            spread_bps = raw_book.spread_bps

            bands_by_bps: dict[int, OrderBookDepthBand] = {
                band.band_bps: band for band in raw_book.depth_bands
            }
            if 10 in bands_by_bps:
                b10 = bands_by_bps[10]
                depth_10_bid = b10.bid_depth_quote
                depth_10_ask = b10.ask_depth_quote
                imb_10 = b10.imbalance
            if 50 in bands_by_bps:
                b50 = bands_by_bps[50]
                depth_50_bid = b50.bid_depth_quote
                depth_50_ask = b50.ask_depth_quote
                imb_50 = b50.imbalance

            if QualityFlag.SEQUENCE_GAP in ob_batch.quality_flags or resync_count > 0:
                limitations.append(
                    f"Order book mengalami celah urutan (sequence gap) dan telah diresinkronisasi "
                    f"ulang ({resync_count}x) dari snapshot penuh."
                )

    # 4. Forced liquidation events (when reliable source is available)
    liq_available = False
    liq_count: int | None = None
    liq_long_quote: float | None = None
    liq_short_quote: float | None = None

    liq_batch = context.liquidations
    if liq_batch is not None:
        if not liq_batch.source_available or liq_batch.has_provider_error:
            liq_available = False
            limitations.append(
                "Data likuidasi publik tidak tersedia dari sumber REST publik venue ini; "
                "jumlah dan nominal likuidasi tidak direkayasa sebagai nol."
            )
        else:
            liq_available = True
            causal_events = [
                ev
                for ev in liq_batch.events
                if not use_available_time or ev.available_time <= cutoff
            ]
            liq_count = len(causal_events)
            liq_long_quote = sum(
                ev.quote_notional
                for ev in causal_events
                if ev.side is LiquidationSide.LONG_LIQUIDATED
            )
            liq_short_quote = sum(
                ev.quote_notional
                for ev in causal_events
                if ev.side is LiquidationSide.SHORT_LIQUIDATED
            )

    # 5. Cross-source price checks
    checks: list[CrossSourcePriceCheck] = []
    if ob_snap is not None and funding_obs is not None and funding_obs.index_price is not None:
        checks.append(
            check_cross_source_price(
                source_a=f"{ob_snap.venue}:{ob_snap.canonical_symbol}:orderbook_mid",
                price_a=ob_snap.mid_price,
                time_a=ob_snap.observed_time,
                source_b=f"{funding_obs.venue}:{funding_obs.canonical_symbol}:index_price",
                price_b=funding_obs.index_price,
                time_b=funding_obs.observed_time,
            )
        )
    if ob_snap is not None:
        for ref in context.reference_prices:
            if use_available_time and ref.available_time > cutoff:
                continue
            checks.append(
                check_cross_source_price(
                    source_a=f"{ob_snap.venue}:{ob_snap.canonical_symbol}:orderbook_mid",
                    price_a=ob_snap.mid_price,
                    time_a=ob_snap.observed_time,
                    source_b=f"{ref.venue}:{ref.canonical_symbol}:{ref.provider}",
                    price_b=ref.price,
                    time_b=ref.observed_time,
                )
            )
    if (
        funding_obs is not None
        and funding_obs.mark_price is not None
        and funding_obs.index_price is not None
    ):
        checks.append(
            check_cross_source_price(
                source_a=f"{funding_obs.venue}:{funding_obs.canonical_symbol}:mark_price",
                price_a=funding_obs.mark_price,
                time_a=funding_obs.observed_time,
                source_b=f"{funding_obs.venue}:{funding_obs.canonical_symbol}:index_price",
                price_b=funding_obs.index_price,
                time_b=funding_obs.observed_time,
            )
        )

    comparable = [c for c in checks if c.status != "non_comparable_time"]
    cross_dev_bps: float | None = None
    cross_pair: str | None = None
    if comparable:
        worst = max(comparable, key=lambda c: c.deviation_bps)
        cross_dev_bps = worst.deviation_bps
        cross_pair = f"{worst.source_a} vs {worst.source_b}"
        if any(c.status == "block" for c in comparable):
            flags.append(QualityFlag.PRICE_MISMATCH)
            flags.append(QualityFlag.SEVERE_PRICE_MISMATCH)
            limitations.append(
                f"Terdeteksi deviasi harga lintas sumber berat (>= {PRICE_MISMATCH_BLOCK_BPS:.0f} "
                f"bps pada {cross_pair}); analisis kondisi terkini ditahan (INSUFFICIENT_DATA)."
            )
        elif any(c.status == "degrade" for c in comparable):
            flags.append(QualityFlag.PRICE_MISMATCH)
            limitations.append(
                f"Terdeteksi deviasi harga lintas sumber moderat (>= "
                f"{PRICE_MISMATCH_DEGRADE_BPS:.0f} bps pada {cross_pair}); status diturunkan "
                "ke DEGRADED."
            )

    deriv_venue = (funding_venue or oi_venue or ob_venue or "satu venue").capitalize()
    limitations.append(
        f"Data derivatif dan order book berasal dari satu venue ({deriv_venue}) dan tidak "
        "merepresentasikan likuiditas seluruh pasar kripto."
    )

    metrics = DerivativesMetrics(
        funding_venue=funding_venue,
        funding_symbol=funding_symbol,
        funding_rate=funding_rate,
        funding_interval_hours=funding_interval_hours,
        funding_rate_per_8h=funding_rate_per_8h,
        funding_rate_annualized=funding_rate_annualized,
        next_funding_time=next_funding_time,
        mark_price=mark_price,
        index_price=index_price,
        mark_index_basis_bps=mark_index_basis_bps,
        open_interest_venue=oi_venue,
        open_interest_symbol=oi_symbol,
        open_interest_base=oi_base,
        open_interest_base_unit=oi_base_unit,
        open_interest_quote=oi_quote,
        open_interest_quote_currency=oi_quote_currency,
        open_interest_change_pct=oi_change_pct,
        open_interest_change_window=oi_change_window,
        orderbook_venue=ob_venue,
        orderbook_symbol=ob_symbol,
        orderbook_market_type=ob_market_type,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread_abs=spread_abs,
        spread_bps=spread_bps,
        depth_10bps_bid_quote=depth_10_bid,
        depth_10bps_ask_quote=depth_10_ask,
        imbalance_10bps=imb_10,
        depth_50bps_bid_quote=depth_50_bid,
        depth_50bps_ask_quote=depth_50_ask,
        imbalance_50bps=imb_50,
        orderbook_resync_count=resync_count,
        liquidation_source_available=liq_available,
        liquidation_count=liq_count,
        liquidation_long_quote=liq_long_quote,
        liquidation_short_quote=liq_short_quote,
        cross_source_deviation_bps=cross_dev_bps,
        cross_source_pair=cross_pair,
    )

    return DerivativesAnalysisResult(
        metrics=metrics,
        quality_flags=tuple(dict.fromkeys(flags)),
        limitations=tuple(dict.fromkeys(limitations)),
        cross_source_checks=tuple(checks),
    )
