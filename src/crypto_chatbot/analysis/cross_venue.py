"""Cross-venue derivatives comparison: dispersion and basis spreads."""

from __future__ import annotations

from typing import Literal

from crypto_chatbot.schemas.derivatives import (
    CrossVenueDerivativesSummary,
    DerivativesContext,
)


def compute_cross_venue_derivatives(
    primary_context: DerivativesContext | None,
    secondary_context: DerivativesContext | None,
    *,
    primary_venue: str = "binance",
    secondary_venue: str = "bybit",
) -> CrossVenueDerivativesSummary | None:
    """Compare funding rates and mark prices across primary and secondary venues.

    Never fails if secondary venue is missing (degrades gracefully to 'partial' or None).
    """
    if primary_context is None and secondary_context is None:
        return None

    venues: list[str] = []
    if primary_context is not None:
        venues.append(primary_venue)
    if secondary_context is not None:
        venues.append(secondary_venue)

    if not venues:
        return None

    if len(venues) == 1:
        return CrossVenueDerivativesSummary(
            venues=tuple(venues),
            primary_venue=primary_venue,
            funding_dispersion_bps=None,
            mark_price_spread_bps=None,
            coverage_status="partial",
        )

    # Both venues present
    primary_funding = (
        primary_context.funding.observation.rate_per_8h
        if primary_context and primary_context.funding and primary_context.funding.observation
        else None
    )
    secondary_funding = (
        secondary_context.funding.observation.rate_per_8h
        if secondary_context and secondary_context.funding and secondary_context.funding.observation
        else None
    )

    funding_dispersion_bps: float | None = None
    if primary_funding is not None and secondary_funding is not None:
        funding_dispersion_bps = abs(primary_funding - secondary_funding) * 10_000.0

    primary_mark = (
        primary_context.funding.observation.mark_price
        if primary_context and primary_context.funding and primary_context.funding.observation
        else None
    )
    secondary_mark = (
        secondary_context.funding.observation.mark_price
        if secondary_context and secondary_context.funding and secondary_context.funding.observation
        else None
    )

    mark_price_spread_bps: float | None = None
    if primary_mark is not None and secondary_mark is not None and primary_mark > 0.0:
        mark_price_spread_bps = (abs(primary_mark - secondary_mark) / primary_mark) * 10_000.0

    coverage: Literal["full", "partial", "unavailable"] = (
        "full"
        if (funding_dispersion_bps is not None or mark_price_spread_bps is not None)
        else "partial"
    )

    return CrossVenueDerivativesSummary(
        venues=tuple(venues),
        primary_venue=primary_venue,
        funding_dispersion_bps=funding_dispersion_bps,
        mark_price_spread_bps=mark_price_spread_bps,
        coverage_status=coverage,
    )
