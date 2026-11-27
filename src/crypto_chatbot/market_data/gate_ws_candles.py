"""Gate.io futures live WebSocket closed-candle subscriber with REST gap-fill (D-09)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.market_data.gate_candles import (
    GateFuturesMarketDataProvider,
    GateFuturesRestClient,
)
from crypto_chatbot.market_data.symbols import SymbolError, SymbolMapping, resolve_symbol
from crypto_chatbot.schemas.market import MarketCandle, MarketType, QualityFlag, Timeframe
from crypto_chatbot.storage.repositories import MarketRepository

logger = logging.getLogger(__name__)

DEFAULT_GATE_WS_URL = "wss://fx-ws.gateio.ws/v4/ws/usdt"

_GATE_WS_TIMEFRAMES: dict[str, Timeframe] = {
    "1m": Timeframe.M1,
    "5m": Timeframe.M5,
    "15m": Timeframe.M15,
    "1h": Timeframe.H1,
    "4h": Timeframe.H4,
    "1d": Timeframe.D1,
}


def build_gate_ws_subscribe_message(
    contract: str,
    interval: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a verified Gate `futures.candlesticks` WebSocket subscribe frame."""
    ts = int((now or datetime.now(UTC)).astimezone(UTC).timestamp())
    return {
        "time": ts,
        "channel": "futures.candlesticks",
        "event": "subscribe",
        "payload": [interval.strip().lower(), contract.strip().upper()],
    }


def parse_gate_ws_candlestick_entry(
    entry: Mapping[str, Any],
    *,
    symbol: SymbolMapping,
    quanto_multiplier: Decimal | float | str,
    received_time: datetime,
    venue: str = "gate",
    provider_name: str = "gate_ws",
) -> tuple[MarketCandle, bool] | None:
    """Parse one `futures.candlesticks` update item into `(MarketCandle, is_window_closed)`."""
    n_raw = str(entry.get("n", "")).strip()
    if "_" not in n_raw:
        return None
    interval_raw, _contract_raw = n_raw.split("_", 1)
    tf = _GATE_WS_TIMEFRAMES.get(interval_raw.lower())
    if tf is None:
        return None

    try:
        open_epoch = int(entry["t"])
        open_p = float(entry["o"])
        high_p = float(entry["h"])
        low_p = float(entry["l"])
        close_p = float(entry["c"])
        contracts_val = float(entry.get("v", 0.0))
    except (KeyError, TypeError, ValueError):
        return None

    if open_epoch <= 0 or not all(
        math.isfinite(val) for val in (open_p, high_p, low_p, close_p, contracts_val)
    ):
        return None
    if min(open_p, high_p, low_p, close_p) <= 0.0 or contracts_val < 0.0:
        return None

    mult = Decimal(str(quanto_multiplier))
    if not mult.is_finite() or mult <= 0:
        raise ValueError(f"invalid quanto_multiplier: {quanto_multiplier}")

    base_vol = float(Decimal(str(contracts_val)) * mult)
    raw_a = entry.get("a")
    flags: tuple[QualityFlag, ...] = ()
    quote_vol: float
    if raw_a is not None and str(raw_a).strip() != "":
        try:
            parsed_a = float(raw_a)
            if math.isfinite(parsed_a) and parsed_a >= 0.0:
                quote_vol = parsed_a
            else:
                quote_vol = base_vol * close_p
                flags = (QualityFlag.QUOTE_VOL_ESTIMATED,)
        except (TypeError, ValueError):
            quote_vol = base_vol * close_p
            flags = (QualityFlag.QUOTE_VOL_ESTIMATED,)
    else:
        quote_vol = base_vol * close_p
        flags = (QualityFlag.QUOTE_VOL_ESTIMATED,)

    open_time = datetime.fromtimestamp(open_epoch, tz=UTC)
    close_time = open_time + tf.duration
    received_utc = (
        received_time.astimezone(UTC)
        if received_time.tzinfo is not None
        else received_time.replace(tzinfo=UTC)
    )
    window_closed = bool(entry.get("w", False))
    provider_symbol = f"GATE:{symbol.base.upper()}{symbol.quote.upper()}.P"

    candle = MarketCandle(
        provider=provider_name,
        venue=venue,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=provider_symbol,
        market_type=MarketType.PERPETUAL,
        timeframe=tf,
        open_time=open_time,
        close_time=close_time,
        received_time=received_utc,
        available_time=close_time,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        raw_contracts=contracts_val,
        base_volume=base_vol,
        quote_volume=quote_vol,
        taker_buy_base_volume=None,
        base_unit=symbol.base,
        quote_currency=symbol.quote,
        is_closed=window_closed,
        quality_flags=flags,
    )
    return candle, window_closed


class GateClosedCandleStreamAssembler:
    """Buffer forming Gate WS bars and emit closed candles strictly once on close or rollover."""

    def __init__(self, *, max_dedup_per_stream: int = 512) -> None:
        self._pending: dict[tuple[str, Timeframe], MarketCandle] = {}
        self._emitted_opens: dict[tuple[str, Timeframe], deque[datetime]] = {}
        self._max_dedup = max(16, int(max_dedup_per_stream))

    def _mark_emitted(self, stream_key: tuple[str, Timeframe], open_time: datetime) -> bool:
        seen = self._emitted_opens.setdefault(stream_key, deque(maxlen=self._max_dedup))
        if open_time in seen:
            return False
        seen.append(open_time)
        return True

    def ingest_payload(
        self,
        payload: Mapping[str, Any],
        *,
        multipliers: Mapping[str, Decimal],
        received_time: datetime,
    ) -> list[MarketCandle]:
        """Process one decoded Gate WS frame and return newly closed `MarketCandle` items."""
        if payload.get("channel") != "futures.candlesticks" or payload.get("event") != "update":
            return []
        result = payload.get("result")
        if isinstance(result, Mapping):
            items: Sequence[Any] = [result]
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            items = result
        else:
            return []

        emitted: list[MarketCandle] = []
        for raw_item in items:
            if not isinstance(raw_item, Mapping):
                continue
            n_raw = str(raw_item.get("n", "")).strip()
            if "_" not in n_raw:
                continue
            _interval, contract_raw = n_raw.split("_", 1)
            contract_id = contract_raw.strip().upper()
            mult = multipliers.get(contract_id)
            if mult is None:
                continue
            base_sym = contract_id[:-5] if contract_id.endswith("_USDT") else contract_id
            try:
                symbol_map = resolve_symbol(base_sym, market_type=MarketType.PERPETUAL)
            except SymbolError:
                continue

            parsed = parse_gate_ws_candlestick_entry(
                raw_item,
                symbol=symbol_map,
                quanto_multiplier=mult,
                received_time=received_time,
            )
            if parsed is None:
                continue
            candle, window_closed = parsed
            stream_key = (contract_id, candle.timeframe)
            pending = self._pending.get(stream_key)

            # Rollover: strictly newer open_time supersedes an unclosed buffered candle.
            if pending is not None and candle.open_time > pending.open_time:
                if self._mark_emitted(stream_key, pending.open_time):
                    effective_rx = max(
                        received_time.astimezone(UTC)
                        if received_time.tzinfo is not None
                        else received_time.replace(tzinfo=UTC),
                        pending.close_time,
                    )
                    emitted.append(
                        pending.model_copy(
                            update={"is_closed": True, "received_time": effective_rx}
                        )
                    )
                self._pending.pop(stream_key, None)
                pending = None

            # Ignore out-of-order stale updates for already superseded bars.
            if pending is not None and candle.open_time < pending.open_time:
                continue

            if window_closed:
                self._pending.pop(stream_key, None)
                if self._mark_emitted(stream_key, candle.open_time):
                    emitted.append(candle)
            else:
                self._pending[stream_key] = candle

        return emitted


class GateWsCandlesRunner:
    """Managed async WebSocket live candle client with REST gap-fill and reconnect backoff."""

    def __init__(
        self,
        *,
        symbols: Sequence[str] = ("BTC", "ETH", "SOL"),
        timeframes: Sequence[str] = ("1m", "5m", "15m", "1h"),
        ws_url: str = DEFAULT_GATE_WS_URL,
        rest_client: GateFuturesRestClient | None = None,
        market_provider: GateFuturesMarketDataProvider | None = None,
        repository: MarketRepository | None = None,
        on_candles: Callable[[Sequence[MarketCandle]], Awaitable[None] | None] | None = None,
        ws_connect: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        gap_fill_limit: int = 50,
    ) -> None:
        self._symbols = tuple(s.strip().upper() for s in symbols if s.strip())
        self._intervals = tuple(
            tf.strip().lower() for tf in timeframes if tf.strip().lower() in _GATE_WS_TIMEFRAMES
        )
        self._ws_url = ws_url
        self._owns_rest = rest_client is None
        self._rest = rest_client or GateFuturesRestClient()
        self._owns_provider = market_provider is None
        self._provider = market_provider or GateFuturesMarketDataProvider(client=self._rest)
        self._repository = repository
        self._on_candles = on_candles
        self._ws_connect = ws_connect
        self._clock = clock or (lambda: datetime.now(UTC))
        self._delay_initial = max(0.05, float(reconnect_initial_delay))
        self._delay_max = max(self._delay_initial, float(reconnect_max_delay))
        self._gap_fill_limit = max(2, int(gap_fill_limit))

        self._assembler = GateClosedCandleStreamAssembler()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._reconnect_count = 0

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    async def _backfill_gaps(self) -> list[MarketCandle]:
        """Fetch recent closed REST candles to close any gap before/after WS reconnect."""
        recovered: list[MarketCandle] = []
        for sym in self._symbols:
            for interval in self._intervals:
                tf = _GATE_WS_TIMEFRAMES[interval]
                batch: OhlcvBatch = await self._provider.get_closed_ohlcv(
                    sym,
                    tf,
                    limit=self._gap_fill_limit,
                )
                if batch.candles:
                    recovered.extend(batch.candles)
        if recovered:
            await self._deliver(recovered)
        return recovered

    async def _deliver(self, candles: Sequence[MarketCandle]) -> None:
        if not candles:
            return
        if self._repository is not None:
            self._repository.save_candles(candles)
        if self._on_candles is not None:
            res = self._on_candles(candles)
            if res is not None:
                await res

    async def _resolve_contracts_and_multipliers(
        self,
    ) -> tuple[list[tuple[str, str]], dict[str, Decimal]]:
        all_mults = await self._rest.fetch_contracts()
        pairs: list[tuple[str, str]] = []
        needed_mults: dict[str, Decimal] = {}
        for sym in self._symbols:
            try:
                mapping = resolve_symbol(sym, market_type=MarketType.PERPETUAL)
            except SymbolError:
                continue
            contract = f"{mapping.base.upper()}_{mapping.quote.upper()}"
            mult = all_mults.get(contract)
            if mult is None:
                try:
                    mult = await self._rest.get_quanto_multiplier(contract)
                except Exception as exc:
                    logger.warning("Skipping WS subscription for %s: %s", contract, exc)
                    continue
            needed_mults[contract] = mult
            for interval in self._intervals:
                pairs.append((contract, interval))
        return pairs, needed_mults

    async def run_once(self) -> None:
        """Run one WS session (gap-fill, subscribe, and ingest frames until closed)."""
        pairs, multipliers = await self._resolve_contracts_and_multipliers()
        if not pairs:
            return
        await self._backfill_gaps()

        connect_fn = self._ws_connect
        if connect_fn is None:
            import websockets

            connect_fn = websockets.connect

        async with connect_fn(self._ws_url, ping_interval=20, ping_timeout=10) as ws:
            now = self._clock()
            for contract, interval in pairs:
                msg = build_gate_ws_subscribe_message(contract, interval, now=now)
                await ws.send(json.dumps(msg))

            async for raw_frame in ws:
                if self._stop_event.is_set():
                    break
                if isinstance(raw_frame, bytes):
                    raw_frame = raw_frame.decode("utf-8")
                try:
                    decoded = json.loads(raw_frame)
                except (TypeError, ValueError):
                    continue
                if not isinstance(decoded, Mapping):
                    continue
                closed = self._assembler.ingest_payload(
                    decoded,
                    multipliers=multipliers,
                    received_time=self._clock(),
                )
                if closed:
                    await self._deliver(closed)

    async def _run_loop(self) -> None:
        delay = self._delay_initial
        while not self._stop_event.is_set():
            try:
                await self.run_once()
                delay = self._delay_initial
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._reconnect_count += 1
                logger.warning("Gate WS session ended (%s); reconnecting in %.1fs", exc, delay)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                delay = min(self._delay_max, delay * 2.0)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="gate-ws-candles")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._owns_provider:
            await self._provider.close()
        if self._owns_rest:
            await self._rest.close()
