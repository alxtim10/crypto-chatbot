"""Direct Gate.io USDT perpetual candlesticks client & contract-unit normalizer (D-02)."""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_FRESHNESS_GRACE,
    FuturesTickerBatch,
    FuturesTickerSnapshot,
    OhlcvBatch,
)
from crypto_chatbot.market_data.gate_contract_stats import DEFAULT_GATE_API_URL
from crypto_chatbot.market_data.quality import is_candle_stale
from crypto_chatbot.market_data.symbols import SymbolError, SymbolMapping, resolve_symbol
from crypto_chatbot.market_data.venue_health import VenueHealthCache, is_venue_block_error
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
)

logger = logging.getLogger(__name__)

GATE_CANDLESTICK_PAGE_LIMIT = 1000
DEFAULT_CONTRACTS_CACHE_TTL_SECONDS = 3600.0
_GATE_1000_PREFIX_BASES: frozenset[str] = frozenset({"PEPE", "SHIB", "FLOKI", "BONK", "LUNC"})

_GATE_TIMEFRAMES: dict[Timeframe, str] = {
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}


class _RetryableGateHttpError(RuntimeError):
    """Transient HTTP 429/5xx error eligible for bounded retry."""


def _parse_decimal(value: Any, *, field_name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} missing")
    try:
        dec = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"non-finite {field_name}: {value!r}")
    return dec


def parse_contract_multipliers(raw_contracts: Sequence[dict[str, Any]]) -> dict[str, Decimal]:
    """Extract `{contract_name: quanto_multiplier}` from `/futures/usdt/contracts` payload."""
    multipliers: dict[str, Decimal] = {}
    for item in raw_contracts:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_mult = item.get("quanto_multiplier")
        if raw_mult is None:
            continue
        mult = _parse_decimal(raw_mult, field_name=f"{name}.quanto_multiplier")
        if mult <= 0:
            raise ValueError(f"{name}.quanto_multiplier must be positive, got {mult}")
        multipliers[name.strip().upper()] = mult
    return multipliers


def to_gate_perp_contract_candidates(mapping: SymbolMapping) -> tuple[str, ...]:
    """Return Gate contract IDs to attempt (handling `1000PEPE_USDT` convention cleanly)."""
    base = mapping.base.upper()
    primary = f"{base}_{mapping.quote.upper()}"
    if base in _GATE_1000_PREFIX_BASES and not base.startswith("1000"):
        return (primary, f"1000{primary}")
    return (primary,)


def _normalize_gate_candlesticks_from_rows(
    raw_candles: Sequence[dict[str, Any]],
    *,
    symbol: SymbolMapping,
    provider_symbol: str,
    timeframe: Timeframe,
    quanto_multiplier: Decimal | float | str,
    received_time: datetime,
    venue: str = "gate",
    provider_name: str = "gate_rest",
    as_of: datetime | None = None,
    freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
    additional_flags: Sequence[QualityFlag] = (),
) -> OhlcvBatch:
    """Normalize Gate candlesticks (`v` in contracts, optional `sum` in USDT) cleanly."""
    if received_time.tzinfo is None:
        raise ValueError("received_time must be timezone-aware")
    mult = _parse_decimal(quanto_multiplier, field_name="quanto_multiplier")
    if mult <= 0:
        raise ValueError("quanto_multiplier must be positive")

    received_utc = received_time.astimezone(UTC)
    effective_as_of = (as_of or received_time).astimezone(UTC)
    flags: list[QualityFlag] = list(additional_flags)

    parsed_by_open: dict[datetime, MarketCandle] = {}
    saw_duplicate = False
    saw_invalid_ohlc = False
    saw_quote_estimated = False

    for row in raw_candles:
        if not isinstance(row, dict):
            saw_invalid_ohlc = True
            continue
        raw_t = row.get("t")
        if raw_t is None or isinstance(raw_t, bool):
            saw_invalid_ohlc = True
            continue
        try:
            open_time = datetime.fromtimestamp(int(raw_t), tz=UTC)
            close_time = open_time + timeframe.duration
            # Closed-candle invariant: skip still-forming bars whose close_time > as_of.
            if close_time > effective_as_of:
                continue

            o_price = float(_parse_decimal(row.get("o"), field_name="o"))
            h_price = float(_parse_decimal(row.get("h"), field_name="h"))
            l_price = float(_parse_decimal(row.get("l"), field_name="l"))
            c_price = float(_parse_decimal(row.get("c"), field_name="c"))
            contracts_dec = _parse_decimal(row.get("v"), field_name="v")
        except (TypeError, ValueError):
            saw_invalid_ohlc = True
            continue

        if (
            min(o_price, h_price, l_price, c_price) <= 0.0
            or l_price > h_price
            or h_price < max(o_price, c_price)
            or l_price > min(o_price, c_price)
            or contracts_dec < 0
        ):
            saw_invalid_ohlc = True
            continue

        raw_contracts = float(contracts_dec)
        base_volume_dec = contracts_dec * mult
        base_volume = float(base_volume_dec)

        raw_sum = row.get("sum")
        quote_volume: float
        row_flags: list[QualityFlag] = list(additional_flags)
        if raw_sum is not None and not isinstance(raw_sum, bool) and str(raw_sum).strip() != "":
            try:
                parsed_sum = float(_parse_decimal(raw_sum, field_name="sum"))
                if parsed_sum < 0.0 or not math.isfinite(parsed_sum):
                    raise ValueError("negative or non-finite sum")
                quote_volume = parsed_sum
            except ValueError:
                quote_volume = base_volume * c_price
                row_flags.append(QualityFlag.QUOTE_VOL_ESTIMATED)
                saw_quote_estimated = True
        else:
            quote_volume = base_volume * c_price
            row_flags.append(QualityFlag.QUOTE_VOL_ESTIMATED)
            saw_quote_estimated = True

        candle = MarketCandle(
            provider=provider_name,
            venue=venue,
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=provider_symbol,
            market_type=symbol.market_type,
            timeframe=timeframe,
            open_time=open_time,
            close_time=close_time,
            received_time=received_utc,
            available_time=close_time,
            open=o_price,
            high=h_price,
            low=l_price,
            close=c_price,
            raw_contracts=raw_contracts,
            base_volume=base_volume,
            quote_volume=quote_volume,
            base_unit=symbol.base,
            quote_currency=symbol.quote,
            is_closed=True,
            quality_flags=tuple(dict.fromkeys(row_flags)),
        )
        if open_time in parsed_by_open:
            saw_duplicate = True
        parsed_by_open[open_time] = candle

    candles = tuple(parsed_by_open[ts] for ts in sorted(parsed_by_open))
    if saw_duplicate:
        flags.append(QualityFlag.DUPLICATE)
    if saw_invalid_ohlc:
        flags.append(QualityFlag.INVALID_OHLC)
    if saw_quote_estimated:
        flags.append(QualityFlag.QUOTE_VOL_ESTIMATED)

    gaps: list[tuple[datetime, datetime]] = []
    if len(candles) >= 2:
        step = timeframe.duration
        for prev, curr in itertools.pairwise(candles):
            if curr.open_time - prev.open_time > step:
                gaps.append((prev.open_time + step, curr.open_time))
    if gaps:
        flags.append(QualityFlag.GAP)

    if candles:
        latest = candles[-1]
        if is_candle_stale(
            latest.close_time,
            timeframe=timeframe,
            now=effective_as_of,
            freshness_grace=freshness_grace,
        ):
            flags.append(QualityFlag.STALE)
        if all(c.open == c.high == c.low == c.close for c in candles):
            flags.append(QualityFlag.FLAT_PRICE)
        if all(c.base_volume == 0.0 for c in candles):
            flags.append(QualityFlag.ZERO_VOLUME)

    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider=provider_name,
            venue=venue,
            market_type=symbol.market_type,
            base_unit=symbol.base,
            quote_currency=symbol.quote,
        ),
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=provider_symbol,
        timeframe=timeframe,
        received_time=received_utc,
        candles=candles,
        quality_flags=tuple(dict.fromkeys(flags)),
        gap_intervals=tuple(gaps),
    )


def normalize_gate_candlesticks(
    raw_candles: Sequence[dict[str, Any]],
    *,
    symbol: SymbolMapping,
    provider_symbol: str,
    timeframe: Timeframe,
    quanto_multiplier: Decimal | float | str,
    received_time: datetime,
    venue: str = "gate",
    provider_name: str = "gate_rest",
    as_of: datetime | None = None,
    freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
    additional_flags: Sequence[QualityFlag] = (),
) -> OhlcvBatch:
    """Public entry point for D-02 Gate.io perpetual candlestick unit normalization."""
    return _normalize_gate_candlesticks_from_rows(
        raw_candles,
        symbol=symbol,
        provider_symbol=provider_symbol,
        timeframe=timeframe,
        quanto_multiplier=quanto_multiplier,
        received_time=received_time,
        venue=venue,
        provider_name=provider_name,
        as_of=as_of,
        freshness_grace=freshness_grace,
        additional_flags=additional_flags,
    )


class GateFuturesRestClient:
    """Async public REST client for Gate `/futures/usdt/{contracts,candlesticks,tickers}`."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_GATE_API_URL,
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        contracts_cache_ttl_seconds: float = DEFAULT_CONTRACTS_CACHE_TTL_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        mono_clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )
        self._max_attempts = max(1, max_attempts)
        self._contracts_ttl = max(0.0, float(contracts_cache_ttl_seconds))
        self._mono_clock = mono_clock or time.monotonic
        self._multipliers_cache: tuple[float, dict[str, Decimal]] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        retrying = AsyncRetrying(
            retry=retry_if_exception(
                lambda exc: (
                    isinstance(exc, (httpx.TransportError, _RetryableGateHttpError))
                    and not is_venue_block_error(exc)
                )
            ),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.4, min=0.2, max=2.0),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.get(path, params=params)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise _RetryableGateHttpError(
                        f"Gate HTTP {response.status_code}: {response.text[:160]}"
                    )
                response.raise_for_status()
                return response.json()
        raise RuntimeError("unreachable")  # pragma: no cover

    async def fetch_contracts(self, *, force_refresh: bool = False) -> dict[str, Decimal]:
        """Return cached `{contract_name: quanto_multiplier}` mapping."""
        now_m = self._mono_clock()
        if (
            not force_refresh
            and self._multipliers_cache is not None
            and (now_m - self._multipliers_cache[0]) < self._contracts_ttl
        ):
            return self._multipliers_cache[1]
        payload = await self._get_json("/futures/usdt/contracts")
        if not isinstance(payload, list):
            raise ValueError("unexpected Gate /contracts payload type")
        parsed = parse_contract_multipliers([item for item in payload if isinstance(item, dict)])
        self._multipliers_cache = (now_m, parsed)
        return parsed

    async def get_quanto_multiplier(self, contract: str) -> Decimal:
        multipliers = await self.fetch_contracts()
        key = contract.strip().upper()
        if key in multipliers:
            return multipliers[key]
        # Single-contract fallback endpoint if contract wasn't in cached page
        payload = await self._get_json(f"/futures/usdt/contracts/{key}")
        if not isinstance(payload, dict) or "quanto_multiplier" not in payload:
            raise SymbolError(f"Gate contract '{key}' not found or missing quanto_multiplier")
        mult = _parse_decimal(payload["quanto_multiplier"], field_name=f"{key}.quanto_multiplier")
        if mult <= 0:
            raise ValueError(f"{key}.quanto_multiplier must be positive")
        updated = dict(multipliers)
        updated[key] = mult
        self._multipliers_cache = (self._mono_clock(), updated)
        return mult

    async def fetch_candlesticks(
        self,
        contract: str,
        *,
        interval: str,
        limit: int | None = None,
        from_time: int | None = None,
        to_time: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"contract": contract, "interval": interval}
        if from_time is not None:
            params["from"] = int(from_time)
        if to_time is not None:
            params["to"] = int(to_time)
        if limit is not None and from_time is None and to_time is None:
            params["limit"] = min(max(1, int(limit)), GATE_CANDLESTICK_PAGE_LIMIT)
        payload = await self._get_json("/futures/usdt/candlesticks", params=params)
        if not isinstance(payload, list):
            raise ValueError("unexpected Gate /candlesticks payload type")
        return [item for item in payload if isinstance(item, dict)]

    async def fetch_tickers(self) -> list[dict[str, Any]]:
        payload = await self._get_json("/futures/usdt/tickers")
        if not isinstance(payload, list):
            raise ValueError("unexpected Gate /tickers payload type")
        return [item for item in payload if isinstance(item, dict)]


class GateFuturesMarketDataProvider:
    """Direct Gate.io USDT perpetual MarketData/History/Ticker provider."""

    venue = "gate"

    def __init__(
        self,
        *,
        client: GateFuturesRestClient | None = None,
        health_cache: VenueHealthCache | None = None,
        freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client or GateFuturesRestClient()
        self._owns_client = client is None
        self._health_cache = health_cache
        self._freshness_grace = freshness_grace
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    def _error_batch(
        self,
        mapping: SymbolMapping,
        timeframe: Timeframe,
        *,
        error_message: str,
        flags: tuple[QualityFlag, ...] = (QualityFlag.PROVIDER_ERROR,),
    ) -> OhlcvBatch:
        return OhlcvBatch(
            metadata=ProviderMetadata(
                provider="gate_rest",
                venue=self.venue,
                market_type=mapping.market_type,
                base_unit=mapping.base,
                quote_currency=mapping.quote,
            ),
            canonical_symbol=mapping.canonical_symbol,
            provider_symbol=f"GATE:{mapping.base}USDT.P",
            timeframe=timeframe,
            received_time=self._clock(),
            candles=(),
            quality_flags=flags,
            error_message=error_message,
        )

    async def _resolve_contract_and_multiplier(
        self,
        mapping: SymbolMapping,
    ) -> tuple[SymbolMapping, str, Decimal]:
        multipliers = await self._client.fetch_contracts()
        for candidate in to_gate_perp_contract_candidates(mapping):
            if candidate in multipliers:
                resolved_base = candidate.split("_", 1)[0]
                resolved_mapping = (
                    mapping
                    if resolved_base == mapping.base
                    else resolve_symbol(resolved_base, market_type=MarketType.PERPETUAL)
                )
                return resolved_mapping, candidate, multipliers[candidate]
        primary_contract = f"{mapping.base.upper()}_{mapping.quote.upper()}"
        mult = await self._client.get_quanto_multiplier(primary_contract)
        return mapping, primary_contract, mult

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        mapping = resolve_symbol(symbol, market_type=MarketType.PERPETUAL)
        if self._health_cache is not None and not self._health_cache.is_available(self.venue):
            reason = self._health_cache.blocked_reason(self.venue) or "cooldown active"
            return self._error_batch(mapping, timeframe, error_message=f"gate skipped ({reason})")

        interval = _GATE_TIMEFRAMES.get(timeframe)
        if interval is None:
            return self._error_batch(
                mapping,
                timeframe,
                error_message=f"Gate perpetual REST does not support timeframe {timeframe.value}",
            )

        try:
            resolved_mapping, contract_id, mult = await self._resolve_contract_and_multiplier(
                mapping
            )
            # Request limit+1 because Gate returns any currently forming open candle at the tail.
            fetch_limit = min(max(1, limit) + 1, GATE_CANDLESTICK_PAGE_LIMIT)
            raw_rows = await self._client.fetch_candlesticks(
                contract_id,
                interval=interval,
                limit=fetch_limit,
            )
            received = self._clock()
            batch = normalize_gate_candlesticks(
                raw_rows,
                symbol=resolved_mapping,
                provider_symbol=f"GATE:{resolved_mapping.base}{resolved_mapping.quote}.P",
                timeframe=timeframe,
                quanto_multiplier=mult,
                received_time=received,
                venue=self.venue,
                provider_name="gate_rest",
                as_of=received,
                freshness_grace=self._freshness_grace,
            )
            if len(batch.candles) > limit:
                batch = batch.model_copy(update={"candles": batch.candles[-limit:]})
            return batch
        except SymbolError as exc:
            return self._error_batch(mapping, timeframe, error_message=str(exc))
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            if self._health_cache is not None and is_venue_block_error(exc):
                self._health_cache.mark_unavailable(self.venue, reason=msg)
            logger.warning(
                "Gate candlesticks fetch failed for %s %s: %s",
                symbol,
                timeframe.value,
                msg,
            )
            return self._error_batch(mapping, timeframe, error_message=msg)

    async def get_closed_ohlcv_history(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        if limit <= GATE_CANDLESTICK_PAGE_LIMIT - 1:
            return await self.get_closed_ohlcv(symbol, timeframe, limit=limit)

        mapping = resolve_symbol(symbol, market_type=MarketType.PERPETUAL)
        if self._health_cache is not None and not self._health_cache.is_available(self.venue):
            reason = self._health_cache.blocked_reason(self.venue) or "cooldown active"
            return self._error_batch(mapping, timeframe, error_message=f"gate skipped ({reason})")

        interval = _GATE_TIMEFRAMES.get(timeframe)
        if interval is None:
            return self._error_batch(
                mapping,
                timeframe,
                error_message=f"Gate perpetual REST does not support timeframe {timeframe.value}",
            )

        try:
            resolved_mapping, contract_id, mult = await self._resolve_contract_and_multiplier(
                mapping
            )
            step_seconds = int(timeframe.duration.total_seconds())
            now = self._clock().astimezone(UTC)
            end_epoch = int(now.timestamp())
            # Walk backward in pages of up to 900 bars using [from, to].
            page_bars = 900
            by_t: dict[int, dict[str, Any]] = {}
            cursor_end = end_epoch
            while len(by_t) < limit + 2:
                cursor_start = max(0, cursor_end - page_bars * step_seconds)
                page = await self._client.fetch_candlesticks(
                    contract_id,
                    interval=interval,
                    from_time=cursor_start,
                    to_time=cursor_end,
                )
                if not page:
                    break
                earliest_seen: int | None = None
                for row in page:
                    if "t" in row:
                        t_val = int(row["t"])
                        by_t[t_val] = row
                        earliest_seen = (
                            t_val if earliest_seen is None else min(earliest_seen, t_val)
                        )
                if (earliest_seen is None or earliest_seen <= cursor_start) and len(page) < 2:
                    break
                next_end = (
                    earliest_seen - step_seconds if earliest_seen is not None else cursor_start - 1
                )
                if next_end >= cursor_end or next_end < 0:
                    break
                cursor_end = next_end

            ordered_rows = [by_t[k] for k in sorted(by_t)]
            received = self._clock()
            batch = normalize_gate_candlesticks(
                ordered_rows,
                symbol=resolved_mapping,
                provider_symbol=f"GATE:{resolved_mapping.base}{resolved_mapping.quote}.P",
                timeframe=timeframe,
                quanto_multiplier=mult,
                received_time=received,
                venue=self.venue,
                provider_name="gate_rest",
                as_of=received,
                freshness_grace=self._freshness_grace,
            )
            if len(batch.candles) > limit:
                batch = batch.model_copy(update={"candles": batch.candles[-limit:]})
            return batch
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            if self._health_cache is not None and is_venue_block_error(exc):
                self._health_cache.mark_unavailable(self.venue, reason=msg)
            return self._error_batch(mapping, timeframe, error_message=msg)

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        now = self._clock().astimezone(UTC)
        if self._health_cache is not None and not self._health_cache.is_available(self.venue):
            reason = self._health_cache.blocked_reason(self.venue) or "cooldown active"
            return FuturesTickerBatch(
                venue=self.venue,
                received_time=now,
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=f"gate skipped ({reason})",
            )
        try:
            raw_tickers = await self._client.fetch_tickers()
            snapshots: list[FuturesTickerSnapshot] = []
            for item in raw_tickers:
                contract = str(item.get("contract", "")).strip().upper()
                if not contract.endswith("_USDT"):
                    continue
                base = contract[:-5]
                if not base or "_" in base:
                    continue
                try:
                    last_price = float(item.get("last") or 0.0)
                    pct = float(item.get("change_percentage") or 0.0)
                    raw_quote_vol = (
                        item.get("volume_24h_settle") or item.get("volume_24h_usd") or 0.0
                    )
                    quote_vol = float(raw_quote_vol)
                except (TypeError, ValueError):
                    continue
                if last_price <= 0.0 or not (
                    math.isfinite(last_price) and math.isfinite(quote_vol)
                ):
                    continue
                try:
                    mapping = resolve_symbol(base, market_type=MarketType.PERPETUAL)
                except SymbolError:
                    continue
                snapshots.append(
                    FuturesTickerSnapshot(
                        canonical_symbol=mapping.canonical_symbol,
                        provider_symbol=f"GATE:{mapping.base}USDT.P",
                        venue=self.venue,
                        last_price=last_price,
                        change_24h_pct=pct if math.isfinite(pct) else 0.0,
                        quote_volume_24h_usd=max(0.0, quote_vol),
                        observed_time=now,
                    )
                )
            return FuturesTickerBatch(
                venue=self.venue,
                received_time=now,
                tickers=tuple(snapshots),
            )
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            if self._health_cache is not None and is_venue_block_error(exc):
                self._health_cache.mark_unavailable(self.venue, reason=msg)
            return FuturesTickerBatch(
                venue=self.venue,
                received_time=now,
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=msg,
            )
