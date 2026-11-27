"""Derive per-bar Gate.io perpetual taker buy/sell base volume from trades (D-05, PRD F-20).

Verified Gate.io sign & unit convention (`GET /futures/usdt/trades` & bulk `trades` CSV):
- `size > 0`: taker buy (buyer lifted ask), base_volume = `size * quanto_multiplier`.
- `size < 0`: taker sell (seller hit bid), base_volume = `(-size) * quanto_multiplier`.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from crypto_chatbot.market_data.gate_contract_stats import DEFAULT_GATE_API_URL
from crypto_chatbot.market_data.gate_history_ingest import GateArchiveTrade
from crypto_chatbot.schemas.market import MarketCandle, Timeframe


@dataclass(frozen=True, slots=True)
class TradeTick:
    """Normalized execution tick used to derive per-bar taker buy/sell volume."""

    timestamp: datetime
    price: float
    signed_contracts: float
    trade_id: int | None = None

    @property
    def is_taker_buy(self) -> bool:
        return self.signed_contracts > 0.0

    @property
    def abs_contracts(self) -> float:
        return abs(self.signed_contracts)


@dataclass(frozen=True, slots=True)
class TakerBarStats:
    """Aggregated taker buy/sell volume within a single candle bucket."""

    open_time: datetime
    taker_buy_contracts: float
    taker_sell_contracts: float
    taker_buy_base_volume: float
    taker_sell_base_volume: float
    trade_count: int

    @property
    def total_base_volume(self) -> float:
        return self.taker_buy_base_volume + self.taker_sell_base_volume


@dataclass(frozen=True, slots=True)
class TakerBarCrossCheck:
    """Per-bar reconciliation between summed trade volume and candlestick volume."""

    open_time: datetime
    candle_base_volume: float
    trade_total_base_volume: float
    abs_diff_pct: float
    within_tolerance: bool


@dataclass(frozen=True, slots=True)
class TakerFlowEnrichmentResult:
    """Result of enriching closed candles with trade-derived `taker_buy_base_volume`."""

    candles: tuple[MarketCandle, ...]
    bars_matched: int
    bars_missing_trades: int
    cross_checks: tuple[TakerBarCrossCheck, ...]
    max_abs_diff_pct: float
    all_within_tolerance: bool


def parse_rest_trades_payload(raw_trades: Sequence[dict[str, Any]]) -> tuple[TradeTick, ...]:
    """Parse Gate REST `/futures/usdt/trades` items into `TradeTick` records."""
    parsed: list[TradeTick] = []
    for item in raw_trades:
        if not isinstance(item, dict):
            continue
        raw_time = item.get("create_time") or item.get("create_time_ms")
        raw_size = item.get("size")
        raw_price = item.get("price")
        if raw_time is None or raw_size is None or raw_price is None:
            continue
        ts = datetime.fromtimestamp(float(raw_time), tz=UTC)
        trade_id = int(item["id"]) if item.get("id") is not None else None
        parsed.append(
            TradeTick(
                timestamp=ts,
                price=float(raw_price),
                signed_contracts=float(raw_size),
                trade_id=trade_id,
            )
        )
    return tuple(parsed)


def _floor_bar_open(ts: datetime, step: timedelta) -> datetime:
    step_seconds = int(step.total_seconds())
    epoch = int(ts.astimezone(UTC).timestamp())
    return datetime.fromtimestamp((epoch // step_seconds) * step_seconds, tz=UTC)


def aggregate_trades_to_bars(
    trades: Iterable[TradeTick | GateArchiveTrade],
    *,
    timeframe: Timeframe,
    quanto_multiplier: Decimal | float | str,
) -> dict[datetime, TakerBarStats]:
    """Bucket execution trades into `[open_time, close_time)` taker buy/sell base volumes."""
    mult = Decimal(str(quanto_multiplier))
    if mult <= 0:
        raise ValueError("quanto_multiplier must be positive")
    step = timeframe.duration

    buy_by_bar: dict[datetime, Decimal] = {}
    sell_by_bar: dict[datetime, Decimal] = {}
    counts_by_bar: dict[datetime, int] = {}

    for tr in trades:
        bar_open = _floor_bar_open(tr.timestamp, step)
        contracts_dec = Decimal(str(tr.signed_contracts))
        if contracts_dec > 0:
            buy_by_bar[bar_open] = buy_by_bar.get(bar_open, Decimal("0")) + contracts_dec
        elif contracts_dec < 0:
            sell_by_bar[bar_open] = sell_by_bar.get(bar_open, Decimal("0")) + (-contracts_dec)
        counts_by_bar[bar_open] = counts_by_bar.get(bar_open, 0) + 1

    out: dict[datetime, TakerBarStats] = {}
    for bar_open in sorted(counts_by_bar):
        buy_c = buy_by_bar.get(bar_open, Decimal("0"))
        sell_c = sell_by_bar.get(bar_open, Decimal("0"))
        out[bar_open] = TakerBarStats(
            open_time=bar_open,
            taker_buy_contracts=float(buy_c),
            taker_sell_contracts=float(sell_c),
            taker_buy_base_volume=float(buy_c * mult),
            taker_sell_base_volume=float(sell_c * mult),
            trade_count=counts_by_bar[bar_open],
        )
    return out


def enrich_candles_with_taker_flow(
    candles: Sequence[MarketCandle],
    *,
    trades_or_bars: Iterable[TradeTick | GateArchiveTrade] | Mapping[datetime, TakerBarStats],
    quanto_multiplier: Decimal | float | str,
    max_deviation_pct: float = 0.5,
) -> TakerFlowEnrichmentResult:
    """Attach `taker_buy_base_volume` from trades to closed candles and cross-check totals."""
    if not candles:
        return TakerFlowEnrichmentResult(
            candles=(),
            bars_matched=0,
            bars_missing_trades=0,
            cross_checks=(),
            max_abs_diff_pct=0.0,
            all_within_tolerance=True,
        )

    if isinstance(trades_or_bars, Mapping):
        bars = dict(trades_or_bars)
    else:
        bars = aggregate_trades_to_bars(
            trades_or_bars,
            timeframe=candles[0].timeframe,
            quanto_multiplier=quanto_multiplier,
        )

    enriched: list[MarketCandle] = []
    checks: list[TakerBarCrossCheck] = []
    matched = 0
    missing = 0
    max_diff_pct = 0.0
    all_ok = True

    for candle in candles:
        stats = bars.get(candle.open_time)
        if stats is None:
            enriched.append(candle)
            missing += 1
            continue

        matched += 1
        trade_total = stats.total_base_volume
        if candle.base_volume > 0.0:
            diff_pct = abs(trade_total - candle.base_volume) / candle.base_volume * 100.0
        elif trade_total == 0.0:
            diff_pct = 0.0
        else:
            diff_pct = 100.0
        within = diff_pct <= max_deviation_pct
        max_diff_pct = max(max_diff_pct, diff_pct)
        if not within:
            all_ok = False
        checks.append(
            TakerBarCrossCheck(
                open_time=candle.open_time,
                candle_base_volume=candle.base_volume,
                trade_total_base_volume=trade_total,
                abs_diff_pct=round(diff_pct, 6),
                within_tolerance=within,
            )
        )
        # Keep MarketCandle invariant `taker_buy_base_volume <= base_volume + 1e-6`
        safe_buy = min(stats.taker_buy_base_volume, candle.base_volume)
        enriched.append(candle.model_copy(update={"taker_buy_base_volume": safe_buy}))

    return TakerFlowEnrichmentResult(
        candles=tuple(enriched),
        bars_matched=matched,
        bars_missing_trades=missing,
        cross_checks=tuple(checks),
        max_abs_diff_pct=round(max_diff_pct, 6),
        all_within_tolerance=all_ok,
    )


def aggregate_gatedata_trades_archives(
    archive_paths: Sequence[Path],
    *,
    timeframe: Timeframe,
    quanto_multiplier: Decimal | float | str,
) -> dict[datetime, TakerBarStats]:
    """Stream one or more `.csv.gz` Gate trades archives into bar taker stats."""
    mult = Decimal(str(quanto_multiplier))
    if mult <= 0:
        raise ValueError("quanto_multiplier must be positive")
    step = timeframe.duration
    buy_by_bar: dict[datetime, Decimal] = {}
    sell_by_bar: dict[datetime, Decimal] = {}
    counts_by_bar: dict[datetime, int] = {}

    for path in archive_paths:
        with gzip.open(path, "rt", encoding="utf-8") as gz:
            for row in csv.reader(gz):
                if not row or len(row) < 4:
                    continue
                ts = datetime.fromtimestamp(float(row[0].strip()), tz=UTC)
                bar_open = _floor_bar_open(ts, step)
                signed_contracts = Decimal(row[3].strip())
                if signed_contracts > 0:
                    buy_by_bar[bar_open] = buy_by_bar.get(bar_open, Decimal("0")) + signed_contracts
                elif signed_contracts < 0:
                    sell_by_bar[bar_open] = sell_by_bar.get(bar_open, Decimal("0")) + (
                        -signed_contracts
                    )
                counts_by_bar[bar_open] = counts_by_bar.get(bar_open, 0) + 1

    out: dict[datetime, TakerBarStats] = {}
    for bar_open in sorted(counts_by_bar):
        buy_c = buy_by_bar.get(bar_open, Decimal("0"))
        sell_c = sell_by_bar.get(bar_open, Decimal("0"))
        out[bar_open] = TakerBarStats(
            open_time=bar_open,
            taker_buy_contracts=float(buy_c),
            taker_sell_contracts=float(sell_c),
            taker_buy_base_volume=float(buy_c * mult),
            taker_sell_base_volume=float(sell_c * mult),
            trade_count=counts_by_bar[bar_open],
        )
    return out


class GateTradesClient:
    """Async public REST client for Gate `/futures/usdt/trades` paginated by window."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_GATE_API_URL,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_trades_window(
        self,
        contract: str,
        *,
        start_time: datetime,
        end_time: datetime,
        page_limit: int = 1000,
        max_pages: int = 50,
    ) -> tuple[TradeTick, ...]:
        """Fetch deduplicated trades across `[start_time, end_time)` via `offset` pagination."""
        from_s = int(start_time.astimezone(UTC).timestamp())
        to_s = int(end_time.astimezone(UTC).timestamp())
        seen: dict[tuple[datetime, int | None, float, float], TradeTick] = {}
        offset = 0
        limit = min(max(1, page_limit), 1000)

        for _ in range(max_pages):
            response = await self._client.get(
                "/futures/usdt/trades",
                params={
                    "contract": contract,
                    "from": from_s,
                    "to": to_s,
                    "limit": limit,
                    "offset": offset,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            ticks = parse_rest_trades_payload([item for item in payload if isinstance(item, dict)])
            for tk in ticks:
                if start_time <= tk.timestamp < end_time:
                    seen[(tk.timestamp, tk.trade_id, tk.price, tk.signed_contracts)] = tk
            if len(payload) < limit:
                break
            offset += len(payload)

        return tuple(sorted(seen.values(), key=lambda t: (t.timestamp, t.trade_id or 0)))
