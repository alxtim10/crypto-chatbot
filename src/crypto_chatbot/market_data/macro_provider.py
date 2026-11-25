"""Macro data provider for US Dollar Index (DXY) observations (ADR 0013)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityFlag,
    Timeframe,
)

logger = logging.getLogger(__name__)

YAHOO_DXY_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
DEFAULT_DXY_TIMEOUT = 3.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def is_dxy_market_closed(as_of: datetime) -> bool:
    """Determine whether global FX markets for DXY are closed.

    Closure window: Friday ~22:00 UTC to Sunday ~22:00 UTC.
    """
    utc_dt = as_of.astimezone(UTC)
    weekday = utc_dt.weekday()  # Monday is 0, Sunday is 6
    if weekday == 4 and utc_dt.hour >= 22:  # Friday after 22:00 UTC
        return True
    if weekday == 5:  # Saturday full day
        return True
    return bool(weekday == 6 and utc_dt.hour < 22)


class DxyMarketDataProvider:
    """Fetches and normalizes closed-bar DXY candles from public Yahoo Finance Chart API."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_DXY_TIMEOUT,
    ) -> None:
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        self._cached_candles: dict[Timeframe, list[MarketCandle]] = {}

    async def close(self) -> None:
        """Close internal HTTP client if owned."""
        if self._owned_client:
            await self._client.aclose()

    async def get_dxy_candles(
        self,
        timeframe: Timeframe,
        *,
        limit: int = 100,
        now: datetime | None = None,
    ) -> tuple[MarketCandle, ...]:
        """Fetch closed DXY candles for given timeframe.

        Carries forward Friday close during weekends or API unresponsiveness.
        """
        current_time = now or datetime.now(UTC)
        interval = "1h"
        query_range = "1mo"
        if timeframe is Timeframe.M15:
            interval = "15m"
            query_range = "5d"
        elif timeframe is Timeframe.H1 or timeframe is Timeframe.H4:
            interval = "1h"
            query_range = "1mo"
        elif timeframe is Timeframe.D1:
            interval = "1d"
            query_range = "3mo"

        try:
            resp = await self._client.get(
                YAHOO_DXY_CHART_URL,
                params={"interval": interval, "range": query_range},
            )
            if resp.status_code == 200:
                if timeframe is Timeframe.H4:
                    hourly = self._parse_yahoo_chart_response(
                        resp.json(),
                        timeframe=Timeframe.H1,
                        now=current_time,
                    )
                    candles = self._resample_to_4h(hourly, now=current_time)
                else:
                    candles = self._parse_yahoo_chart_response(
                        resp.json(),
                        timeframe=timeframe,
                        now=current_time,
                    )
                if candles:
                    self._cached_candles[timeframe] = list(candles)
                    return candles[-limit:]
        except Exception as exc:
            logger.warning("Failed to fetch DXY candles from Yahoo Finance: %s", exc)

        # Fallback to cached candles for this exact timeframe
        cached = self._cached_candles.get(timeframe)
        if cached:
            return tuple(cached[-limit:])
        return ()

    @staticmethod
    def _resample_to_4h(
        hourly_candles: tuple[MarketCandle, ...],
        *,
        now: datetime,
    ) -> tuple[MarketCandle, ...]:
        """Resample closed 1h DXY candles into closed 4h UTC buckets cleanly."""
        if not hourly_candles:
            return ()
        buckets: dict[datetime, list[MarketCandle]] = {}
        for candle in hourly_candles:
            utc_open = candle.open_time.astimezone(UTC)
            bucket_hour = (utc_open.hour // 4) * 4
            bucket_open = utc_open.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
            buckets.setdefault(bucket_open, []).append(candle)

        out: list[MarketCandle] = []
        for bucket_open in sorted(buckets):
            bucket_close = bucket_open + Timeframe.H4.duration
            if bucket_close > now:
                continue
            group = sorted(buckets[bucket_open], key=lambda c: c.open_time)
            out.append(
                MarketCandle(
                    provider=group[0].provider,
                    venue=group[0].venue,
                    canonical_symbol=group[0].canonical_symbol,
                    provider_symbol=group[0].provider_symbol,
                    market_type=group[0].market_type,
                    timeframe=Timeframe.H4,
                    open_time=bucket_open,
                    close_time=bucket_close,
                    received_time=now,
                    available_time=bucket_close,
                    open=group[0].open,
                    high=max(c.high for c in group),
                    low=min(c.low for c in group),
                    close=group[-1].close,
                    base_volume=sum(c.base_volume for c in group),
                    quote_volume=None,
                    taker_buy_base_volume=None,
                    base_unit=group[0].base_unit,
                    quote_currency=group[0].quote_currency,
                    is_closed=True,
                    quality_flags=(),
                )
            )
        return tuple(out)

    def _parse_yahoo_chart_response(
        self,
        data: dict[str, Any],
        *,
        timeframe: Timeframe,
        now: datetime,
    ) -> tuple[MarketCandle, ...]:
        chart = data.get("chart", {})
        results = chart.get("result", [])
        if not results:
            return ()

        entry = results[0]
        timestamps = entry.get("timestamp", [])
        quotes = entry.get("indicators", {}).get("quote", [{}])[0]
        opens = quotes.get("open", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        closes = quotes.get("close", [])
        volumes = quotes.get("volume", [])

        candles: list[MarketCandle] = []
        is_closed_market = is_dxy_market_closed(now)

        for i, ts in enumerate(timestamps):
            if ts is None:
                continue
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            low_val = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else None

            if o is None or h is None or low_val is None or c is None:
                continue

            open_time = datetime.fromtimestamp(ts, tz=UTC)
            close_time = open_time + timeframe.duration
            if close_time > now:
                # Candle is currently forming; only closed candles are allowed
                continue

            flags: list[QualityFlag] = []
            if is_closed_market and i == len(timestamps) - 1:
                # Carried forward over closed market
                pass

            candle = MarketCandle(
                provider="yahoo",
                venue="nyb",
                canonical_symbol="DXY/USD",
                provider_symbol="DX-Y.NYB",
                market_type=MarketType.SPOT,
                timeframe=timeframe,
                open_time=open_time,
                close_time=close_time,
                received_time=now,
                available_time=close_time,
                open=float(o),
                high=float(h),
                low=float(low_val),
                close=float(c),
                base_volume=float(v) if v is not None else 0.0,
                quote_volume=None,
                taker_buy_base_volume=None,
                base_unit="DXY",
                quote_currency="USD",
                is_closed=True,
                quality_flags=tuple(flags),
            )
            candles.append(candle)

        return tuple(candles)
