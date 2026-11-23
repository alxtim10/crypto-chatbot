"""Deterministic candle normalization, gap/duplicate detection, and freshness checks."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from crypto_chatbot.market_data.base import (
    DEFAULT_BACKFILL_AVAILABILITY_LAG,
    DEFAULT_FRESHNESS_GRACE,
    MIN_DEGRADED_HISTORY_CANDLES,
    MIN_FULL_HISTORY_CANDLES,
    OhlcvBatch,
)
from crypto_chatbot.market_data.symbols import SymbolMapping
from crypto_chatbot.schemas.market import (
    MarketCandle,
    ProviderMetadata,
    QualityFlag,
    QualityStatus,
    Timeframe,
)

_INSUFFICIENT_FLAGS: frozenset[QualityFlag] = frozenset(
    {
        QualityFlag.STALE,
        QualityFlag.INVALID_OHLC,
        QualityFlag.SYMBOL_MISMATCH,
        QualityFlag.GAP,
        QualityFlag.PROVIDER_ERROR,
        QualityFlag.SEVERE_PRICE_MISMATCH,
    }
)

_DEGRADED_FLAGS: frozenset[QualityFlag] = frozenset(
    {
        QualityFlag.SHORT_HISTORY,
        QualityFlag.ZERO_VOLUME,
        QualityFlag.FLAT_PRICE,
        QualityFlag.PRICE_MISMATCH,
        QualityFlag.DERIVATIVES_UNAVAILABLE,
        QualityFlag.ORDERBOOK_UNAVAILABLE,
        QualityFlag.UNIT_MISMATCH,
    }
)


def _require_utc(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    return dt.astimezone(UTC)


def _to_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def is_candle_stale(
    latest_close_time: datetime,
    timeframe: Timeframe,
    now: datetime,
    *,
    freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
) -> bool:
    """Return True when `now - latest_close_time > timeframe + grace` (PROJECT_PLAN §5.3)."""
    close_utc = _require_utc(latest_close_time, "latest_close_time")
    now_utc = _require_utc(now, "now")
    return (now_utc - close_utc) > (timeframe.duration + freshness_grace)


def determine_quality_status(
    candle_count: int,
    quality_flags: Sequence[QualityFlag],
) -> QualityStatus:
    """Map candle history length and flags to READY / DEGRADED / INSUFFICIENT_DATA (§5.7)."""
    flag_set = set(quality_flags)
    if (flag_set & _INSUFFICIENT_FLAGS) or candle_count < MIN_DEGRADED_HISTORY_CANDLES:
        return QualityStatus.INSUFFICIENT_DATA
    if (flag_set & _DEGRADED_FLAGS) or candle_count < MIN_FULL_HISTORY_CANDLES:
        return QualityStatus.DEGRADED
    return QualityStatus.READY


def _candle_sort_key(candle: MarketCandle) -> tuple[datetime, float, float, float, float, float]:
    return (
        candle.open_time,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.base_volume,
    )


def _ohlcv_equal(first: MarketCandle, second: MarketCandle) -> bool:
    return (
        first.open == second.open
        and first.high == second.high
        and first.low == second.low
        and first.close == second.close
        and first.base_volume == second.base_volume
        and first.quote_volume == second.quote_volume
        and first.taker_buy_base_volume == second.taker_buy_base_volume
    )


def _parse_raw_row(
    row: Sequence[Any],
    *,
    symbol: SymbolMapping,
    timeframe: Timeframe,
    provider: str,
    received_utc: datetime,
    is_backfill: bool,
    backfill_lag: timedelta,
) -> tuple[MarketCandle | None, bool, bool]:
    """Parse one raw CCXT row. Returns `(candle, is_excluded_open, is_invalid)`."""
    if len(row) < 6:
        return None, False, True

    ts_ms = _to_finite_float(row[0])
    open_price = _to_finite_float(row[1])
    high_price = _to_finite_float(row[2])
    low_price = _to_finite_float(row[3])
    close_price = _to_finite_float(row[4])
    base_volume = _to_finite_float(row[5])
    quote_volume: float | None = None
    if len(row) >= 7 and row[6] is not None:
        quote_volume = _to_finite_float(row[6])
        if quote_volume is None:
            return None, False, True

    taker_buy_base_volume: float | None = None
    if len(row) >= 10 and row[9] is not None:
        raw_tb = _to_finite_float(row[9])
        if raw_tb is not None and base_volume is not None:
            taker_buy_base_volume = min(max(0.0, raw_tb), base_volume)

    if (
        ts_ms is None
        or open_price is None
        or high_price is None
        or low_price is None
        or close_price is None
        or base_volume is None
    ):
        return None, False, True

    try:
        open_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None, False, True

    close_time = open_time + timeframe.duration
    if close_time > received_utc:
        return None, True, False

    available_time = close_time + backfill_lag if is_backfill else max(close_time, received_utc)

    try:
        candle = MarketCandle(
            provider=provider,
            venue=symbol.venue,
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=symbol.provider_symbol,
            market_type=symbol.market_type,
            timeframe=timeframe,
            open_time=open_time,
            close_time=close_time,
            received_time=received_utc,
            available_time=available_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            base_volume=base_volume,
            quote_volume=quote_volume,
            taker_buy_base_volume=taker_buy_base_volume,
            base_unit=symbol.base,
            quote_currency=symbol.quote,
            is_closed=True,
        )
    except ValidationError:
        return None, False, True

    return candle, False, False


def _deduplicate_candles(
    parsed: list[MarketCandle],
) -> tuple[list[MarketCandle], int, bool]:
    """Deduplicate by `open_time` deterministically; flag conflicting duplicates."""
    grouped: dict[datetime, list[MarketCandle]] = {}
    for candle in parsed:
        grouped.setdefault(candle.open_time, []).append(candle)

    deduplicated: list[MarketCandle] = []
    duplicate_count = 0
    has_conflict = False

    for open_time in sorted(grouped):
        group = sorted(grouped[open_time], key=_candle_sort_key)
        if len(group) > 1:
            duplicate_count += len(group) - 1
            first = group[0]
            if any(not _ohlcv_equal(first, other) for other in group[1:]):
                has_conflict = True
        deduplicated.append(group[-1])

    return deduplicated, duplicate_count, has_conflict


def normalize_ohlcv_rows(
    raw_rows: Sequence[Sequence[Any]],
    *,
    symbol: SymbolMapping,
    timeframe: Timeframe,
    received_time: datetime,
    provider: str = "ccxt",
    limit: int | None = None,
    now: datetime | None = None,
    is_backfill: bool = False,
    backfill_lag: timedelta = DEFAULT_BACKFILL_AVAILABILITY_LAG,
    freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
    extra_flags: Sequence[QualityFlag] = (),
) -> OhlcvBatch:
    """Validate and normalize raw CCXT OHLCV rows into a closed-candle `OhlcvBatch`.

    Never interpolates missing intervals and never repairs invalid OHLC values.
    """
    received_utc = _require_utc(received_time, "received_time")
    check_time = _require_utc(now, "now") if now is not None else received_utc
    if backfill_lag < timedelta(0):
        raise ValueError("backfill_lag must be non-negative")
    if freshness_grace < timedelta(0):
        raise ValueError("freshness_grace must be non-negative")

    metadata = ProviderMetadata(
        provider=provider,
        venue=symbol.venue,
        market_type=symbol.market_type,
        base_unit=symbol.base,
        quote_currency=symbol.quote,
    )

    flags: list[QualityFlag] = list(extra_flags)
    parsed: list[MarketCandle] = []
    excluded_open_count = 0
    invalid_row_count = 0

    for row in raw_rows:
        candle, is_excluded_open, is_invalid = _parse_raw_row(
            row,
            symbol=symbol,
            timeframe=timeframe,
            provider=provider,
            received_utc=received_utc,
            is_backfill=is_backfill,
            backfill_lag=backfill_lag,
        )
        if is_excluded_open:
            excluded_open_count += 1
        elif is_invalid or candle is None:
            invalid_row_count += 1
        else:
            parsed.append(candle)

    if invalid_row_count > 0:
        flags.append(QualityFlag.INVALID_OHLC)

    deduplicated, duplicate_count, has_conflict = _deduplicate_candles(parsed)
    if duplicate_count > 0:
        flags.append(QualityFlag.DUPLICATE)
    if has_conflict:
        flags.append(QualityFlag.INVALID_OHLC)

    if limit is not None and limit > 0 and len(deduplicated) > limit:
        deduplicated = deduplicated[-limit:]

    gap_intervals: list[tuple[datetime, datetime]] = []
    for prev, curr in itertools.pairwise(deduplicated):
        expected_open = prev.open_time + timeframe.duration
        if curr.open_time > expected_open:
            gap_intervals.append((expected_open, curr.open_time))
            flags.append(QualityFlag.GAP)
        elif curr.open_time < expected_open:
            flags.append(QualityFlag.INVALID_OHLC)

    if len(deduplicated) < MIN_FULL_HISTORY_CANDLES:
        flags.append(QualityFlag.SHORT_HISTORY)

    if not is_backfill and (
        not deduplicated
        or is_candle_stale(
            deduplicated[-1].close_time,
            timeframe,
            check_time,
            freshness_grace=freshness_grace,
        )
    ):
        flags.append(QualityFlag.STALE)

    unique_flags = tuple(dict.fromkeys(flags))

    return OhlcvBatch(
        metadata=metadata,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        timeframe=timeframe,
        received_time=received_utc,
        candles=tuple(deduplicated),
        quality_flags=unique_flags,
        gap_intervals=tuple(gap_intervals),
        duplicate_count=duplicate_count,
        invalid_row_count=invalid_row_count,
        excluded_open_count=excluded_open_count,
    )
