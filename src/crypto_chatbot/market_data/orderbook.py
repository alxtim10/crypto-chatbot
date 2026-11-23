"""Order-book normalization, depth/imbalance metrics, and sequence-gap resynchronization."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from crypto_chatbot.market_data.symbols import SymbolMapping
from crypto_chatbot.schemas.derivatives import (
    OrderBookBatch,
    OrderBookDepthBand,
    OrderBookDiff,
    OrderBookLevel,
    OrderBookSnapshot,
)
from crypto_chatbot.schemas.market import QualityFlag

DEFAULT_DEPTH_BANDS_BPS: tuple[int, ...] = (10, 50)
MAX_RAW_BOOK_LEVELS = 25


def _require_utc(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    return dt.astimezone(UTC)


def _to_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _parse_book_levels(
    raw_levels: Sequence[Any],
    *,
    reverse: bool,
) -> tuple[OrderBookLevel, ...]:
    merged: dict[float, float] = {}
    for entry in raw_levels:
        if isinstance(entry, OrderBookLevel):
            price = entry.price
            qty = entry.quantity
        elif (
            isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) >= 2
        ):
            parsed_price = _to_finite_float(entry[0])
            parsed_qty = _to_finite_float(entry[1])
            if parsed_price is None or parsed_qty is None:
                raise ValueError("order book contains non-finite price or quantity")
            price = parsed_price
            qty = parsed_qty
        else:
            raise ValueError("malformed order book level")

        if price <= 0:
            raise ValueError("order book level price must be > 0")
        if qty == 0:
            continue
        if qty < 0:
            raise ValueError("order book level quantity must be >= 0")
        merged[price] = merged.get(price, 0.0) + qty

    sorted_prices = sorted(merged, reverse=reverse)
    return tuple(OrderBookLevel(price=p, quantity=merged[p]) for p in sorted_prices)


def extract_sequence_id(raw_book: Mapping[str, Any]) -> int | None:
    """Extract sequence ID (`nonce` or `info.lastUpdateId`) from a CCXT order-book dict."""
    nonce = raw_book.get("nonce")
    if isinstance(nonce, int) and not isinstance(nonce, bool) and nonce >= 1:
        return nonce
    info = raw_book.get("info")
    if isinstance(info, Mapping):
        for key in ("lastUpdateId", "u"):
            raw_id = info.get(key)
            if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id >= 1:
                return raw_id
            if isinstance(raw_id, str) and raw_id.isdigit() and int(raw_id) >= 1:
                return int(raw_id)
    return None


def compute_orderbook_metrics(
    bids: Sequence[OrderBookLevel | tuple[float, float] | Sequence[float]],
    asks: Sequence[OrderBookLevel | tuple[float, float] | Sequence[float]],
    *,
    depth_bands_bps: Sequence[int] = DEFAULT_DEPTH_BANDS_BPS,
) -> tuple[
    tuple[OrderBookLevel, ...],
    tuple[OrderBookLevel, ...],
    float,
    float,
    float,
    float,
    float,
    tuple[OrderBookDepthBand, ...],
]:
    """Validate book levels and compute spread, mid price, depth bands, and imbalance."""
    sorted_bids = _parse_book_levels(bids, reverse=True)
    sorted_asks = _parse_book_levels(asks, reverse=False)
    if not sorted_bids or not sorted_asks:
        raise ValueError("order book must contain at least one bid and one ask level")

    best_bid = sorted_bids[0].price
    best_ask = sorted_asks[0].price
    if best_bid >= best_ask:
        raise ValueError(
            f"crossed or locked order book: best_bid={best_bid} >= best_ask={best_ask}"
        )

    mid_price = (best_bid + best_ask) / 2.0
    spread_abs = best_ask - best_bid
    spread_bps = (spread_abs / mid_price) * 10_000.0

    bands: list[OrderBookDepthBand] = []
    for band_bps in depth_bands_bps:
        if band_bps <= 0:
            raise ValueError("depth band bps must be > 0")
        frac = band_bps / 10_000.0
        bid_floor = mid_price * (1.0 - frac)
        ask_ceil = mid_price * (1.0 + frac)

        bid_base = 0.0
        bid_quote = 0.0
        for lvl in sorted_bids:
            if lvl.price < bid_floor:
                break
            bid_base += lvl.quantity
            bid_quote += lvl.price * lvl.quantity

        ask_base = 0.0
        ask_quote = 0.0
        for lvl in sorted_asks:
            if lvl.price > ask_ceil:
                break
            ask_base += lvl.quantity
            ask_quote += lvl.price * lvl.quantity

        total_quote = bid_quote + ask_quote
        imbalance: float | None = (
            (bid_quote - ask_quote) / total_quote if total_quote > 0.0 else None
        )
        bands.append(
            OrderBookDepthBand(
                band_bps=band_bps,
                bid_depth_base=bid_base,
                ask_depth_base=ask_base,
                bid_depth_quote=bid_quote,
                ask_depth_quote=ask_quote,
                imbalance=imbalance,
            )
        )

    return (
        sorted_bids,
        sorted_asks,
        best_bid,
        best_ask,
        mid_price,
        spread_abs,
        spread_bps,
        tuple(bands),
    )


def normalize_order_book_payload(
    raw_book: Mapping[str, Any],
    *,
    symbol: SymbolMapping,
    received_time: datetime,
    provider: str = "ccxt",
    depth_bands_bps: Sequence[int] = DEFAULT_DEPTH_BANDS_BPS,
    resync_count: int = 0,
    max_raw_levels: int = MAX_RAW_BOOK_LEVELS,
    extra_flags: Sequence[QualityFlag] = (),
) -> OrderBookBatch:
    """Normalize a raw CCXT order-book dictionary into an `OrderBookBatch`."""
    received_utc = _require_utc(received_time, "received_time")
    raw_bids = raw_book.get("bids")
    raw_asks = raw_book.get("asks")
    if not isinstance(raw_bids, Sequence) or not isinstance(raw_asks, Sequence):
        flags = tuple(dict.fromkeys([*extra_flags, QualityFlag.ORDERBOOK_UNAVAILABLE]))
        return OrderBookBatch(
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=symbol.provider_symbol,
            venue=symbol.venue,
            market_type=symbol.market_type,
            received_time=received_utc,
            snapshot=None,
            resync_count=resync_count,
            quality_flags=flags,
            error_message="Order book payload missing bids or asks.",
        )

    ts_ms = _to_finite_float(raw_book.get("timestamp"))
    if ts_ms is not None:
        try:
            observed_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            observed_utc = received_utc
    else:
        observed_utc = received_utc
    available_utc = max(observed_utc, received_utc)

    try:
        (
            sorted_bids,
            sorted_asks,
            best_bid,
            best_ask,
            mid_price,
            spread_abs,
            spread_bps,
            bands,
        ) = compute_orderbook_metrics(
            raw_bids,
            raw_asks,
            depth_bands_bps=depth_bands_bps,
        )
    except ValueError as exc:
        flags = tuple(dict.fromkeys([*extra_flags, QualityFlag.ORDERBOOK_UNAVAILABLE]))
        return OrderBookBatch(
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=symbol.provider_symbol,
            venue=symbol.venue,
            market_type=symbol.market_type,
            received_time=received_utc,
            snapshot=None,
            resync_count=resync_count,
            quality_flags=flags,
            error_message=f"Invalid order book payload: {exc}",
        )

    sequence_id = extract_sequence_id(raw_book)
    unique_flags = tuple(dict.fromkeys(extra_flags))
    bounded_raw = {
        "timestamp": raw_book.get("timestamp"),
        "nonce": sequence_id,
        "bids": [[lvl.price, lvl.quantity] for lvl in sorted_bids[:max_raw_levels]],
        "asks": [[lvl.price, lvl.quantity] for lvl in sorted_asks[:max_raw_levels]],
    }
    snapshot = OrderBookSnapshot(
        provider=provider,
        venue=symbol.venue,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        market_type=symbol.market_type,
        observed_time=observed_utc,
        received_time=received_utc,
        available_time=available_utc,
        sequence_id=sequence_id,
        bids=sorted_bids,
        asks=sorted_asks,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        spread_abs=spread_abs,
        spread_bps=spread_bps,
        depth_bands=bands,
        base_unit=symbol.base,
        quote_currency=symbol.quote,
        resync_count=resync_count,
        quality_flags=unique_flags,
        raw_payload=bounded_raw,
    )
    return OrderBookBatch(
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.provider_symbol,
        venue=symbol.venue,
        market_type=symbol.market_type,
        received_time=received_utc,
        snapshot=snapshot,
        resync_count=resync_count,
        quality_flags=unique_flags,
    )


class OrderBookSynchronizer:
    """Stateful L2 order-book sequence checker that triggers snapshot resync on gaps."""

    def __init__(self) -> None:
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self.last_update_id: int | None = None
        self.observed_time: datetime | None = None
        self.synchronized: bool = False
        self.resync_count: int = 0
        self.had_sequence_gap: bool = False
        self._has_applied_first_diff: bool = False

    def load_snapshot(
        self,
        *,
        sequence_id: int,
        bids: Sequence[OrderBookLevel | tuple[float, float] | Sequence[float]],
        asks: Sequence[OrderBookLevel | tuple[float, float] | Sequence[float]],
        observed_time: datetime,
    ) -> None:
        """Initialize or replace the local order book from a full REST snapshot."""
        if sequence_id < 1:
            raise ValueError("sequence_id must be >= 1")
        sorted_bids, sorted_asks, *_ = compute_orderbook_metrics(bids, asks)
        self._bids = {lvl.price: lvl.quantity for lvl in sorted_bids}
        self._asks = {lvl.price: lvl.quantity for lvl in sorted_asks}
        self.last_update_id = sequence_id
        self.observed_time = _require_utc(observed_time, "observed_time")
        self.synchronized = True
        self._has_applied_first_diff = False

    def apply_diff(self, diff: OrderBookDiff) -> bool:
        """Apply an incremental depth diff. Returns False when a sequence gap requires resync."""
        if not self.synchronized or self.last_update_id is None:
            self.had_sequence_gap = True
            return False

        # Drop stale diffs whose final update ID is not newer than our snapshot.
        if diff.final_update_id <= self.last_update_id:
            return True

        gap_detected = False
        if diff.first_update_id > self.last_update_id + 1:
            gap_detected = True
        elif diff.prev_final_update_id is not None:
            if self._has_applied_first_diff:
                if diff.prev_final_update_id != self.last_update_id:
                    gap_detected = True
            else:
                bridges_initial = (
                    diff.first_update_id <= self.last_update_id <= diff.final_update_id
                    and diff.prev_final_update_id <= self.last_update_id
                )
                if diff.prev_final_update_id != self.last_update_id and not bridges_initial:
                    gap_detected = True

        if gap_detected:
            self.synchronized = False
            self.had_sequence_gap = True
            return False

        for side_map, updates in ((self._bids, diff.bids), (self._asks, diff.asks)):
            for price, qty in updates:
                if qty == 0.0:
                    side_map.pop(price, None)
                else:
                    side_map[price] = qty

        if not self._bids or not self._asks or max(self._bids) >= min(self._asks):
            self.synchronized = False
            self.had_sequence_gap = True
            return False

        self.last_update_id = diff.final_update_id
        self.observed_time = diff.event_time
        self._has_applied_first_diff = True
        return True

    def current_raw_payload(self) -> dict[str, Any]:
        """Serialize the current synchronized local book into a CCXT-like payload."""
        if not self.synchronized or self.observed_time is None:
            raise RuntimeError("order book synchronizer is not synchronized")
        bids = [[p, self._bids[p]] for p in sorted(self._bids, reverse=True)]
        asks = [[p, self._asks[p]] for p in sorted(self._asks)]
        return {
            "timestamp": int(self.observed_time.timestamp() * 1000),
            "nonce": self.last_update_id,
            "bids": bids,
            "asks": asks,
        }

    async def apply_diff_or_resync(
        self,
        diff: OrderBookDiff,
        *,
        resync_fetch: Callable[[], Awaitable[Mapping[str, Any]]],
        symbol: SymbolMapping,
        received_time: datetime,
        provider: str = "ccxt",
        depth_bands_bps: Sequence[int] = DEFAULT_DEPTH_BANDS_BPS,
        max_resync_attempts: int = 2,
    ) -> OrderBookBatch:
        """Apply `diff`, or trigger full-snapshot resynchronization if a sequence gap occurs."""
        received_utc = _require_utc(received_time, "received_time")
        if max_resync_attempts < 1:
            raise ValueError("max_resync_attempts must be >= 1")

        if self.apply_diff(diff):
            flags = (QualityFlag.SEQUENCE_GAP,) if self.had_sequence_gap else ()
            return normalize_order_book_payload(
                self.current_raw_payload(),
                symbol=symbol,
                received_time=received_utc,
                provider=provider,
                depth_bands_bps=depth_bands_bps,
                resync_count=self.resync_count,
                extra_flags=flags,
            )

        prev_update_id = self.last_update_id
        last_error: str | None = None
        for _ in range(max_resync_attempts):
            self.resync_count += 1
            try:
                raw_snapshot = await resync_fetch()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            batch = normalize_order_book_payload(
                raw_snapshot,
                symbol=symbol,
                received_time=received_utc,
                provider=provider,
                depth_bands_bps=depth_bands_bps,
                resync_count=self.resync_count,
                extra_flags=(QualityFlag.SEQUENCE_GAP,),
            )
            snap = batch.snapshot
            if snap is None:
                last_error = batch.error_message or "invalid snapshot during resync"
                continue
            if (
                snap.sequence_id is not None
                and prev_update_id is not None
                and snap.sequence_id <= prev_update_id
            ):
                last_error = (
                    f"stale snapshot sequence_id={snap.sequence_id} <= {prev_update_id} "
                    "during resync"
                )
                continue

            if snap.sequence_id is not None:
                self.load_snapshot(
                    sequence_id=snap.sequence_id,
                    bids=snap.bids,
                    asks=snap.asks,
                    observed_time=snap.observed_time,
                )
            else:
                self._bids = {lvl.price: lvl.quantity for lvl in snap.bids}
                self._asks = {lvl.price: lvl.quantity for lvl in snap.asks}
                self.last_update_id = None
                self.observed_time = snap.observed_time
                self.synchronized = True
                self._has_applied_first_diff = False
            return batch

        return OrderBookBatch(
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=symbol.provider_symbol,
            venue=symbol.venue,
            market_type=symbol.market_type,
            received_time=received_utc,
            snapshot=None,
            resync_count=self.resync_count,
            quality_flags=(
                QualityFlag.SEQUENCE_GAP,
                QualityFlag.PROVIDER_ERROR,
                QualityFlag.ORDERBOOK_UNAVAILABLE,
            ),
            error_message=(
                f"Order-book resynchronization failed after {max_resync_attempts} attempt(s)"
                + (f": {last_error}" if last_error else "")
            ),
        )
