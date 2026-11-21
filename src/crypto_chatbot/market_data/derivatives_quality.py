"""Normalization and quality checks for funding rates, OI units, liquidations, and prices."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from crypto_chatbot.market_data.symbols import SymbolMapping
from crypto_chatbot.schemas.derivatives import (
    FUNDING_COMPARISON_INTERVAL_HOURS,
    HOURS_PER_YEAR,
    CrossSourcePriceCheck,
    FundingRateBatch,
    FundingRateObservation,
    LiquidationBatch,
    LiquidationEvent,
    LiquidationSide,
    OpenInterestBatch,
    OpenInterestObservation,
)
from crypto_chatbot.schemas.market import MarketType, QualityFlag, Timeframe

DEFAULT_OI_UNIT_TOLERANCE = 0.10
PRICE_MISMATCH_DEGRADE_BPS = 50.0
PRICE_MISMATCH_BLOCK_BPS = 200.0
CROSS_SOURCE_MAX_TIME_DIFF = timedelta(minutes=2)
DEFAULT_FUNDING_MAX_AGE = timedelta(minutes=15)


def _require_utc(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    return dt.astimezone(UTC)


def _to_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            val = float(stripped)
        except ValueError:
            return None
        return val if math.isfinite(val) else None
    return None


def _parse_timestamp_utc(value: Any) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    return None


def _parse_interval_hours(raw_interval: Any) -> float | None:
    if isinstance(raw_interval, (int, float)) and not isinstance(raw_interval, bool):
        val = float(raw_interval)
        return val if math.isfinite(val) and val > 0 else None
    if isinstance(raw_interval, str):
        cleaned = raw_interval.strip().lower()
        if cleaned.endswith("h"):
            cleaned = cleaned[:-1]
        val_parsed = _to_finite_float(cleaned)
        if val_parsed is not None and val_parsed > 0:
            return val_parsed
    return None


def normalize_funding_rate_value(
    funding_rate: float,
    funding_interval_hours: float,
) -> tuple[float, float]:
    """Convert a per-interval funding rate into `(rate_per_8h, annualized_rate)`."""
    if not math.isfinite(funding_rate):
        raise ValueError("funding_rate must be finite")
    if not math.isfinite(funding_interval_hours) or funding_interval_hours <= 0:
        raise ValueError("funding_interval_hours must be finite and > 0")
    rate_per_8h = funding_rate * (FUNDING_COMPARISON_INTERVAL_HOURS / funding_interval_hours)
    annualized_rate = funding_rate * (HOURS_PER_YEAR / funding_interval_hours)
    return rate_per_8h, annualized_rate


def compare_funding_rates(
    first: FundingRateObservation,
    second: FundingRateObservation,
) -> float:
    """Compare two normalized funding observations on the unified 8h horizon."""
    return first.rate_per_8h - second.rate_per_8h


def normalize_funding_rate_payload(
    raw_funding: Mapping[str, Any],
    *,
    symbol: SymbolMapping,
    received_time: datetime,
    funding_interval_hours: float | None = None,
    provider: str = "ccxt",
    now: datetime | None = None,
    freshness_max_age: timedelta = DEFAULT_FUNDING_MAX_AGE,
    extra_flags: Sequence[QualityFlag] = (),
) -> FundingRateBatch:
    """Normalize a raw CCXT funding-rate payload with interval comparability & freshness checks."""
    if symbol.market_type is not MarketType.PERPETUAL:
        raise ValueError(
            f"funding rate requires perpetual symbol, got market_type={symbol.market_type.value}"
        )
    received_utc = _require_utc(received_time, "received_time")
    check_utc = _require_utc(now, "now") if now is not None else received_utc
    if freshness_max_age < timedelta(0):
        raise ValueError("freshness_max_age must be non-negative")

    info = raw_funding.get("info")
    info_map = info if isinstance(info, Mapping) else {}

    funding_rate = _to_finite_float(raw_funding.get("fundingRate"))
    if funding_rate is None:
        funding_rate = _to_finite_float(info_map.get("lastFundingRate"))

    resolved_interval = funding_interval_hours
    if resolved_interval is None:
        resolved_interval = _parse_interval_hours(raw_funding.get("interval"))
    if resolved_interval is None:
        resolved_interval = _parse_interval_hours(info_map.get("fundingIntervalHours"))

    if funding_rate is None or resolved_interval is None or resolved_interval <= 0:
        err_flags = tuple(dict.fromkeys([*extra_flags, QualityFlag.DERIVATIVES_UNAVAILABLE]))
        return FundingRateBatch(
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=symbol.provider_symbol,
            venue=symbol.venue,
            received_time=received_utc,
            observation=None,
            quality_flags=err_flags,
            error_message="Funding rate payload missing finite rate or positive funding interval.",
        )

    observed_utc = (
        _parse_timestamp_utc(raw_funding.get("timestamp"))
        or _parse_timestamp_utc(info_map.get("time"))
        or _parse_timestamp_utc(raw_funding.get("datetime"))
        or received_utc
    )
    next_funding_utc = _parse_timestamp_utc(raw_funding.get("fundingTimestamp")) or (
        _parse_timestamp_utc(info_map.get("nextFundingTime"))
    )
    available_utc = max(observed_utc, received_utc)

    mark_price = _to_finite_float(raw_funding.get("markPrice")) or _to_finite_float(
        info_map.get("markPrice")
    )
    if mark_price is not None and mark_price <= 0:
        mark_price = None
    index_price = _to_finite_float(raw_funding.get("indexPrice")) or _to_finite_float(
        info_map.get("indexPrice")
    )
    if index_price is not None and index_price <= 0:
        index_price = None

    rate_per_8h, annualized_rate = normalize_funding_rate_value(funding_rate, resolved_interval)

    batch_flags: list[QualityFlag] = list(extra_flags)
    if (check_utc - observed_utc) > freshness_max_age:
        batch_flags.append(QualityFlag.STALE)
        batch_flags.append(QualityFlag.DERIVATIVES_UNAVAILABLE)

    unique_flags = tuple(dict.fromkeys(batch_flags))
    raw_snapshot = {
        "symbol": raw_funding.get("symbol"),
        "timestamp": raw_funding.get("timestamp"),
        "fundingTimestamp": raw_funding.get("fundingTimestamp"),
        "fundingRate": funding_rate,
        "fundingIntervalHours": resolved_interval,
        "markPrice": mark_price,
        "indexPrice": index_price,
    }
    observation = FundingRateObservation(
        provider=provider,
        venue=symbol.venue,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        market_type=MarketType.PERPETUAL,
        observed_time=observed_utc,
        next_funding_time=next_funding_utc,
        received_time=received_utc,
        available_time=available_utc,
        funding_rate=funding_rate,
        funding_interval_hours=resolved_interval,
        rate_per_8h=rate_per_8h,
        annualized_rate=annualized_rate,
        mark_price=mark_price,
        index_price=index_price,
        quality_flags=unique_flags,
        raw_payload=raw_snapshot,
    )
    return FundingRateBatch(
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        venue=symbol.venue,
        received_time=received_utc,
        observation=observation,
        quality_flags=unique_flags,
    )


def normalize_open_interest_unit(
    *,
    contracts: float,
    raw_unit: str,
    contract_size: float = 1.0,
    quote_value: float | None = None,
    reference_price: float | None = None,
    is_inverse: bool = False,
    unit_tolerance: float = DEFAULT_OI_UNIT_TOLERANCE,
) -> tuple[float | None, float | None, tuple[QualityFlag, ...]]:
    """Normalize venue open-interest units into `(base_quantity, quote_notional, flags)`."""
    if not math.isfinite(contracts) or contracts < 0:
        raise ValueError("contracts must be finite and >= 0")
    if not math.isfinite(contract_size) or contract_size <= 0:
        raise ValueError("contract_size must be finite and > 0")
    if quote_value is not None and (not math.isfinite(quote_value) or quote_value < 0):
        raise ValueError("quote_value must be finite and >= 0")
    if reference_price is not None and (not math.isfinite(reference_price) or reference_price <= 0):
        raise ValueError("reference_price must be finite and > 0")
    if not math.isfinite(unit_tolerance) or unit_tolerance < 0:
        raise ValueError("unit_tolerance must be non-negative")

    unit_clean = raw_unit.strip().lower()
    final_base: float | None = None
    final_quote: float | None = None

    if is_inverse:
        derived_quote = contracts * contract_size
        if quote_value is not None:
            scale = max(derived_quote, quote_value, 1e-12)
            if max(derived_quote, quote_value) > 0 and (
                abs(derived_quote - quote_value) / scale > unit_tolerance
            ):
                return None, None, (QualityFlag.UNIT_MISMATCH,)
            final_quote = quote_value
        else:
            final_quote = derived_quote
        final_base = (final_quote / reference_price) if reference_price is not None else None
        return final_base, final_quote, ()

    if unit_clean == "quote":
        final_quote = quote_value if quote_value is not None else contracts
        if quote_value is not None and max(contracts, quote_value) > 0:
            scale = max(contracts, quote_value, 1e-12)
            if abs(contracts - quote_value) / scale > unit_tolerance:
                return None, None, (QualityFlag.UNIT_MISMATCH,)
        final_base = (final_quote / reference_price) if reference_price is not None else None
        return final_base, final_quote, ()

    final_base = contracts * contract_size
    if quote_value is not None and reference_price is not None:
        expected_quote = final_base * reference_price
        scale = max(expected_quote, quote_value, 1e-12)
        if max(expected_quote, quote_value) > 0 and (
            abs(expected_quote - quote_value) / scale > unit_tolerance
        ):
            return None, None, (QualityFlag.UNIT_MISMATCH,)
        final_quote = quote_value
    elif quote_value is not None:
        final_quote = quote_value
    elif reference_price is not None:
        final_quote = final_base * reference_price
    return final_base, final_quote, ()


def _parse_single_oi_entry(
    raw_entry: Mapping[str, Any],
    *,
    symbol: SymbolMapping,
    period: str,
    received_utc: datetime,
    provider: str,
    contract_size: float,
    raw_unit: str,
    is_inverse: bool,
    reference_price: float | None,
    unit_tolerance: float,
    extra_flags: Sequence[QualityFlag],
) -> tuple[OpenInterestObservation | None, tuple[QualityFlag, ...]]:
    info = raw_entry.get("info")
    info_map = info if isinstance(info, Mapping) else {}

    contracts = _to_finite_float(raw_entry.get("openInterestAmount"))
    if contracts is None:
        contracts = _to_finite_float(info_map.get("sumOpenInterest"))
    if contracts is None:
        contracts = _to_finite_float(info_map.get("openInterest"))
    if contracts is None or contracts < 0:
        return None, (QualityFlag.DERIVATIVES_UNAVAILABLE,)

    quote_value = _to_finite_float(raw_entry.get("openInterestValue"))
    if quote_value is None:
        quote_value = _to_finite_float(info_map.get("sumOpenInterestValue"))
    if quote_value is None:
        quote_value = _to_finite_float(info_map.get("sumOpenInterestUsd"))
    if quote_value is not None and quote_value < 0:
        return None, (QualityFlag.DERIVATIVES_UNAVAILABLE,)

    entry_ref_price = _to_finite_float(raw_entry.get("referencePrice")) or reference_price
    if entry_ref_price is not None and entry_ref_price <= 0:
        entry_ref_price = None

    observed_utc = (
        _parse_timestamp_utc(raw_entry.get("timestamp"))
        or _parse_timestamp_utc(info_map.get("timestamp"))
        or _parse_timestamp_utc(info_map.get("time"))
        or received_utc
    )
    available_utc = max(observed_utc, received_utc)

    base_val, quote_val, unit_flags = normalize_open_interest_unit(
        contracts=contracts,
        raw_unit=raw_unit,
        contract_size=contract_size,
        quote_value=quote_value,
        reference_price=entry_ref_price,
        is_inverse=is_inverse,
        unit_tolerance=unit_tolerance,
    )
    if QualityFlag.UNIT_MISMATCH in unit_flags:
        return None, (QualityFlag.UNIT_MISMATCH, QualityFlag.DERIVATIVES_UNAVAILABLE)

    obs_flags = tuple(dict.fromkeys([*extra_flags, *unit_flags]))
    obs = OpenInterestObservation(
        provider=provider,
        venue=symbol.venue,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        market_type=MarketType.PERPETUAL,
        period=period,
        observed_time=observed_utc,
        received_time=received_utc,
        available_time=available_utc,
        open_interest_contracts=contracts,
        raw_unit=raw_unit,
        contract_size=contract_size,
        open_interest_base=base_val,
        base_unit=symbol.base,
        open_interest_quote=quote_val,
        quote_currency=symbol.quote,
        quality_flags=obs_flags,
        raw_payload={
            "timestamp": raw_entry.get("timestamp"),
            "openInterestAmount": contracts,
            "openInterestValue": quote_value,
            "raw_unit": raw_unit,
            "contract_size": contract_size,
        },
    )
    return obs, obs_flags


def normalize_open_interest_payload(
    raw_latest: Mapping[str, Any] | None,
    raw_history: Sequence[Mapping[str, Any]],
    *,
    symbol: SymbolMapping,
    timeframe: Timeframe,
    received_time: datetime,
    contract_size: float = 1.0,
    raw_unit: str | None = None,
    is_inverse: bool = False,
    reference_price: float | None = None,
    unit_tolerance: float = DEFAULT_OI_UNIT_TOLERANCE,
    provider: str = "ccxt",
    extra_flags: Sequence[QualityFlag] = (),
) -> OpenInterestBatch:
    """Normalize current + historical open-interest payloads with unit checks."""
    if symbol.market_type is not MarketType.PERPETUAL:
        raise ValueError(
            f"open interest requires perpetual symbol, got market_type={symbol.market_type.value}"
        )
    received_utc = _require_utc(received_time, "received_time")
    resolved_unit = raw_unit or ("inverse_contract" if is_inverse else "contracts")

    flags: list[QualityFlag] = list(extra_flags)
    latest_obs: OpenInterestObservation | None = None
    error_message: str | None = None

    if raw_latest is not None:
        latest_obs, latest_flags = _parse_single_oi_entry(
            raw_latest,
            symbol=symbol,
            period="live",
            received_utc=received_utc,
            provider=provider,
            contract_size=contract_size,
            raw_unit=resolved_unit,
            is_inverse=is_inverse,
            reference_price=reference_price,
            unit_tolerance=unit_tolerance,
            extra_flags=extra_flags,
        )
        flags.extend(latest_flags)
        if latest_obs is None and QualityFlag.UNIT_MISMATCH in latest_flags:
            error_message = "Open-interest base and quote units diverge beyond tolerance."
    else:
        flags.append(QualityFlag.DERIVATIVES_UNAVAILABLE)
        error_message = "Open-interest payload unavailable."

    by_time: dict[datetime, OpenInterestObservation] = {}
    for item in raw_history:
        obs, item_flags = _parse_single_oi_entry(
            item,
            symbol=symbol,
            period=timeframe.value,
            received_utc=received_utc,
            provider=provider,
            contract_size=contract_size,
            raw_unit=resolved_unit,
            is_inverse=is_inverse,
            reference_price=reference_price,
            unit_tolerance=unit_tolerance,
            extra_flags=extra_flags,
        )
        if QualityFlag.UNIT_MISMATCH in item_flags:
            flags.extend(item_flags)
            latest_obs = None
            by_time.clear()
            error_message = "Open-interest history units diverge beyond tolerance."
            break
        if obs is not None:
            by_time[obs.observed_time] = obs

    sorted_history = tuple(by_time[t] for t in sorted(by_time))
    if latest_obs is None and not error_message:
        flags.append(QualityFlag.DERIVATIVES_UNAVAILABLE)
        error_message = "Latest open-interest observation missing valid numeric values."

    unique_flags = tuple(dict.fromkeys(flags))
    return OpenInterestBatch(
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        venue=symbol.venue,
        timeframe=timeframe,
        received_time=received_utc,
        latest=latest_obs,
        history=sorted_history,
        quality_flags=unique_flags,
        error_message=error_message,
    )


def normalize_liquidation_events(
    raw_events: Sequence[Mapping[str, Any]],
    *,
    symbol: SymbolMapping,
    received_time: datetime,
    provider: str = "ccxt",
    source_available: bool = True,
    extra_flags: Sequence[QualityFlag] = (),
) -> LiquidationBatch:
    """Normalize reliable liquidation events without ever fabricating zeros when absent."""
    if symbol.market_type is not MarketType.PERPETUAL:
        raise ValueError(
            f"liquidations require perpetual symbol, got market_type={symbol.market_type.value}"
        )
    received_utc = _require_utc(received_time, "received_time")
    unique_flags = tuple(dict.fromkeys(extra_flags))
    if not source_available:
        return LiquidationBatch(
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=symbol.provider_symbol,
            venue=symbol.venue,
            received_time=received_utc,
            source_available=False,
            events=(),
            quality_flags=unique_flags,
            error_message=(
                "No reliable public liquidation source available; counts not fabricated."
            ),
        )

    events: list[LiquidationEvent] = []
    seen_keys: set[str] = set()
    for raw in raw_events:
        price = _to_finite_float(raw.get("price"))
        qty = _to_finite_float(raw.get("base_quantity"))
        if qty is None:
            qty = _to_finite_float(raw.get("amount"))
        if qty is None:
            qty = _to_finite_float(raw.get("origQty"))
        if price is None or price <= 0 or qty is None or qty <= 0:
            continue

        raw_side = str(raw.get("side", "")).strip().lower()
        if raw_side in {"sell", "long", "long_liquidated"}:
            side = LiquidationSide.LONG_LIQUIDATED
        elif raw_side in {"buy", "short", "short_liquidated"}:
            side = LiquidationSide.SHORT_LIQUIDATED
        else:
            continue

        event_time = (
            _parse_timestamp_utc(raw.get("timestamp"))
            or _parse_timestamp_utc(raw.get("time"))
            or _parse_timestamp_utc(raw.get("event_time"))
            or received_utc
        )
        available_utc = max(event_time, received_utc)

        notional = _to_finite_float(raw.get("quote_notional")) or _to_finite_float(raw.get("cost"))
        if notional is None or notional <= 0:
            notional = price * qty

        raw_key = raw.get("id") or raw.get("event_key")
        if isinstance(raw_key, str) and raw_key.strip():
            event_key = raw_key.strip()[:64]
        else:
            digest_input = (
                f"{symbol.venue}|{symbol.canonical_symbol}|{event_time.isoformat()}|"
                f"{side.value}|{price:.12g}|{qty:.12g}"
            )
            event_key = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:32]

        if event_key in seen_keys:
            continue
        seen_keys.add(event_key)

        events.append(
            LiquidationEvent(
                provider=provider,
                venue=symbol.venue,
                canonical_symbol=symbol.canonical_symbol,
                provider_symbol=symbol.provider_symbol,
                market_type=MarketType.PERPETUAL,
                event_time=event_time,
                received_time=received_utc,
                available_time=available_utc,
                side=side,
                price=price,
                base_quantity=qty,
                quote_notional=notional,
                base_unit=symbol.base,
                quote_currency=symbol.quote,
                event_key=event_key,
                raw_payload=dict(raw),
            )
        )

    events.sort(key=lambda ev: (ev.event_time, ev.event_key))
    return LiquidationBatch(
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        venue=symbol.venue,
        received_time=received_utc,
        source_available=True,
        events=tuple(events),
        quality_flags=unique_flags,
    )


def check_cross_source_price(
    *,
    source_a: str,
    price_a: float,
    time_a: datetime,
    source_b: str,
    price_b: float,
    time_b: datetime,
    degrade_bps: float = PRICE_MISMATCH_DEGRADE_BPS,
    block_bps: float = PRICE_MISMATCH_BLOCK_BPS,
    max_time_diff: timedelta = CROSS_SOURCE_MAX_TIME_DIFF,
) -> CrossSourcePriceCheck:
    """Compare two price sources strictly when observations are contemporaneous."""
    if not math.isfinite(price_a) or price_a <= 0 or not math.isfinite(price_b) or price_b <= 0:
        raise ValueError("prices for cross-source check must be finite and > 0")
    if degrade_bps <= 0 or block_bps < degrade_bps:
        raise ValueError("require 0 < degrade_bps <= block_bps")
    time_a_utc = _require_utc(time_a, "time_a")
    time_b_utc = _require_utc(time_b, "time_b")

    diff_seconds = abs((time_a_utc - time_b_utc).total_seconds())
    deviation_bps = abs(price_a - price_b) / price_b * 10_000.0

    status: Literal["ok", "degrade", "block", "non_comparable_time"]
    if diff_seconds > max_time_diff.total_seconds():
        status = "non_comparable_time"
    elif deviation_bps >= block_bps:
        status = "block"
    elif deviation_bps >= degrade_bps:
        status = "degrade"
    else:
        status = "ok"

    return CrossSourcePriceCheck(
        source_a=source_a,
        price_a=price_a,
        time_a=time_a_utc,
        source_b=source_b,
        price_b=price_b,
        time_b=time_b_utc,
        time_diff_seconds=diff_seconds,
        deviation_bps=deviation_bps,
        status=status,
    )
