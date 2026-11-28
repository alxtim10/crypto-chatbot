"""Gate.io /futures/usdt/contract_stats client, normalizer, recorder, and gap auditing (D-06)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from crypto_chatbot.market_data.symbols import SymbolMapping, resolve_default_perp_symbol
from crypto_chatbot.schemas.derivatives import ContractStatsObservation
from crypto_chatbot.schemas.market import QualityFlag, UtcDatetime
from crypto_chatbot.storage.repositories import ContractStatsRepository

logger = logging.getLogger(__name__)

DEFAULT_GATE_API_URL = "https://api.gateio.ws/api/v4"
DEFAULT_CONTRACT_STATS_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "SUI/USDT:USDT",
    "DOGE/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "ADA/USDT:USDT",
)
GATE_CONTRACT_STATS_INTERVALS: dict[str, timedelta] = {
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
}
GATE_CONTRACT_STATS_MAX_LOOKBACK = timedelta(days=179, hours=12)
DEFAULT_CONTRACT_STATS_PAGE_LIMIT = 1000
DEFAULT_POLL_LIMIT = 12


def to_gate_contract_symbol(symbol: str) -> tuple[SymbolMapping, str]:
    """Resolve user/canonical input into `(SymbolMapping, Gate contract_id like 'BTC_USDT')`."""
    cleaned = symbol.strip().upper().replace("-", "_")
    if cleaned.endswith("_USDT"):
        cleaned = cleaned[:-5]
    mapping = resolve_default_perp_symbol(cleaned)
    return mapping, f"{mapping.base}_{mapping.quote}"


def _optional_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def normalize_contract_stats_payload(
    raw_items: Sequence[dict[str, Any]],
    *,
    symbol: SymbolMapping,
    provider_symbol: str,
    interval: Literal["5m", "1h"] = "5m",
    received_time: datetime,
    venue: str = "gateio",
    as_of: datetime | None = None,
    closed_only: bool = True,
    backfill: bool = False,
) -> tuple[ContractStatsObservation, ...]:
    """Normalize Gate `/futures/usdt/contract_stats` records with strict UTC availability."""
    if interval not in GATE_CONTRACT_STATS_INTERVALS:
        raise ValueError(f"unsupported contract_stats interval: {interval}")
    if received_time.tzinfo is None:
        raise ValueError("received_time must be timezone-aware")
    step = GATE_CONTRACT_STATS_INTERVALS[interval]
    effective_as_of = as_of or received_time
    if effective_as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    effective_as_of_utc = effective_as_of.astimezone(UTC)
    received_utc = received_time.astimezone(UTC)

    by_time: dict[datetime, ContractStatsObservation] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_time = item.get("time")
        if raw_time is None:
            continue
        observed_time = datetime.fromtimestamp(int(raw_time), tz=UTC)
        bar_close_time = observed_time + step
        if closed_only and bar_close_time > effective_as_of_utc:
            continue
        # Backfilled historical rows use bar close time (with BACKFILL flag) for causal
        # backtest/ML usability; live polls record actual fetch time (>= bar_close_time).
        available_time = bar_close_time if backfill else max(bar_close_time, received_utc)

        flags: list[QualityFlag] = []
        if backfill:
            flags.append(QualityFlag.BACKFILL)

        long_liq = _optional_finite_float(item.get("long_liq_usd"))
        short_liq = _optional_finite_float(item.get("short_liq_usd"))
        if long_liq is None:
            long_liq = _optional_finite_float(item.get("long_liq_usd_new"))
        if short_liq is None:
            short_liq = _optional_finite_float(item.get("short_liq_usd_new"))

        obs = ContractStatsObservation(
            venue=venue,
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=provider_symbol,
            interval=interval,
            observed_time=observed_time,
            received_time=received_utc,
            available_time=available_time,
            lsr_taker=_optional_finite_float(item.get("lsr_taker")),
            lsr_account=_optional_finite_float(item.get("lsr_account")),
            top_lsr_account=_optional_finite_float(item.get("top_lsr_account")),
            top_lsr_size=_optional_finite_float(item.get("top_lsr_size")),
            long_liq_usd=long_liq if (long_liq is None or long_liq >= 0.0) else None,
            short_liq_usd=short_liq if (short_liq is None or short_liq >= 0.0) else None,
            open_interest=_optional_finite_float(item.get("open_interest")),
            open_interest_usd=_optional_finite_float(item.get("open_interest_usd")),
            mark_price=_optional_finite_float(item.get("mark_price")),
            last_funding_rate=_optional_finite_float(item.get("last_funding_rate")),
            long_users=_optional_non_negative_int(item.get("long_users")),
            short_users=_optional_non_negative_int(item.get("short_users")),
            quality_flags=tuple(flags),
            raw_payload=dict(item),
        )
        by_time[observed_time] = obs

    return tuple(by_time[ts] for ts in sorted(by_time))


class _RetryableGateHttpError(RuntimeError):
    """Transient HTTP error eligible for retry."""


class GateContractStatsClient:
    """Async public REST client for Gate.io `/futures/usdt/contract_stats`."""

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

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.TransportError, _RetryableGateHttpError)),
    )
    async def fetch_contract_stats(
        self,
        contract: str,
        *,
        interval: Literal["5m", "1h"] = "5m",
        from_time: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if interval not in GATE_CONTRACT_STATS_INTERVALS:
            raise ValueError(f"unsupported interval: {interval}")
        if limit < 1 or limit > DEFAULT_CONTRACT_STATS_PAGE_LIMIT:
            raise ValueError(f"limit must be in 1..{DEFAULT_CONTRACT_STATS_PAGE_LIMIT}")
        params: dict[str, Any] = {
            "contract": contract,
            "interval": interval,
            "limit": limit,
        }
        if from_time is not None:
            params["from"] = int(from_time)
        response = await self._client.get("/futures/usdt/contract_stats", params=params)
        if response.status_code in {429, 500, 502, 503, 504}:
            raise _RetryableGateHttpError(
                f"Gate HTTP {response.status_code}: {response.text[:160]}"
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(
                f"unexpected Gate contract_stats payload type: {type(payload).__name__}"
            )
        return [item for item in payload if isinstance(item, dict)]


@dataclass(frozen=True, slots=True)
class ContractStatsRunSummary:
    """Summary of one poll or backfill pass."""

    venue: str
    interval: str
    symbols_processed: int
    fetched_rows: int
    inserted_rows: int
    skipped_duplicates: int
    errors: tuple[str, ...] = ()


class ContractStatsGapRange(BaseModel):
    """Contiguous missing bar range within an expected contract_stats grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_time: UtcDatetime
    end_time: UtcDatetime
    missing_bars: int = Field(ge=1)


class SymbolContractStatsGapReport(BaseModel):
    """Coverage and missing-bar report for a single perpetual symbol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    venue: str
    canonical_symbol: str
    interval: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    expected_bars: int = Field(ge=0)
    observed_bars: int = Field(ge=0)
    missing_bars: int = Field(ge=0)
    coverage_pct: float = Field(ge=0.0, le=100.0)
    earliest_observed: UtcDatetime | None = None
    latest_observed: UtcDatetime | None = None
    gaps: tuple[ContractStatsGapRange, ...] = ()


class ContractStatsGapReport(BaseModel):
    """Aggregate coverage report across all audited symbols."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: UtcDatetime
    venue: str
    interval: str
    lookback_days: int = Field(ge=1)
    symbols: tuple[SymbolContractStatsGapReport, ...]


def _floor_to_grid(ts: datetime, step: timedelta) -> datetime:
    step_seconds = int(step.total_seconds())
    epoch = int(ts.astimezone(UTC).timestamp())
    floored = (epoch // step_seconds) * step_seconds
    return datetime.fromtimestamp(floored, tz=UTC)


def compute_symbol_gap_report(
    observed_times: Sequence[datetime],
    *,
    venue: str,
    canonical_symbol: str,
    interval: Literal["5m", "1h"],
    window_start: datetime,
    window_end: datetime,
) -> SymbolContractStatsGapReport:
    """Compute exact expected vs observed bar grid and contiguous missing intervals."""
    step = GATE_CONTRACT_STATS_INTERVALS[interval]
    step_s = int(step.total_seconds())
    grid_start = _floor_to_grid(window_start, step)
    if grid_start < window_start.astimezone(UTC):
        grid_start += step
    # Last closed bar starting at grid_end must have grid_end + step <= window_end
    grid_end = _floor_to_grid(window_end.astimezone(UTC) - step, step)
    if grid_end < grid_start:
        return SymbolContractStatsGapReport(
            venue=venue,
            canonical_symbol=canonical_symbol,
            interval=interval,
            window_start=window_start.astimezone(UTC),
            window_end=window_end.astimezone(UTC),
            expected_bars=0,
            observed_bars=0,
            missing_bars=0,
            coverage_pct=100.0,
        )

    observed_epochs = {
        int(ts.astimezone(UTC).timestamp())
        for ts in observed_times
        if grid_start <= ts.astimezone(UTC) <= grid_end
    }
    start_epoch = int(grid_start.timestamp())
    end_epoch = int(grid_end.timestamp())
    expected_bars = ((end_epoch - start_epoch) // step_s) + 1
    observed_bars = len(observed_epochs)
    missing_bars = expected_bars - observed_bars

    gaps: list[ContractStatsGapRange] = []
    current_gap_start: int | None = None
    current_gap_count = 0
    for epoch in range(start_epoch, end_epoch + 1, step_s):
        if epoch not in observed_epochs:
            if current_gap_start is None:
                current_gap_start = epoch
                current_gap_count = 1
            else:
                current_gap_count += 1
        elif current_gap_start is not None:
            gap_end = epoch - step_s
            gaps.append(
                ContractStatsGapRange(
                    start_time=datetime.fromtimestamp(current_gap_start, tz=UTC),
                    end_time=datetime.fromtimestamp(gap_end, tz=UTC),
                    missing_bars=current_gap_count,
                )
            )
            current_gap_start = None
            current_gap_count = 0

    if current_gap_start is not None:
        gaps.append(
            ContractStatsGapRange(
                start_time=datetime.fromtimestamp(current_gap_start, tz=UTC),
                end_time=datetime.fromtimestamp(end_epoch, tz=UTC),
                missing_bars=current_gap_count,
            )
        )

    sorted_obs = sorted(observed_epochs)
    coverage_pct = round((observed_bars / expected_bars) * 100.0, 2) if expected_bars else 100.0
    return SymbolContractStatsGapReport(
        venue=venue,
        canonical_symbol=canonical_symbol,
        interval=interval,
        window_start=grid_start,
        window_end=grid_end + step,
        expected_bars=expected_bars,
        observed_bars=observed_bars,
        missing_bars=missing_bars,
        coverage_pct=coverage_pct,
        earliest_observed=(datetime.fromtimestamp(sorted_obs[0], tz=UTC) if sorted_obs else None),
        latest_observed=(datetime.fromtimestamp(sorted_obs[-1], tz=UTC) if sorted_obs else None),
        gaps=tuple(gaps),
    )


class ContractStatsRecorder:
    """Polls, backfills, and audits Gate.io perpetual `contract_stats` snapshots."""

    def __init__(
        self,
        *,
        client: GateContractStatsClient,
        repository: ContractStatsRepository,
        symbols: Sequence[str] = DEFAULT_CONTRACT_STATS_UNIVERSE,
        interval: Literal["5m", "1h"] = "5m",
        venue: str = "gateio",
    ) -> None:
        self._client = client
        self._repository = repository
        self._symbols = tuple(symbols) if symbols else DEFAULT_CONTRACT_STATS_UNIVERSE
        self._interval = interval
        self._venue = venue

    async def poll_once(
        self,
        *,
        limit: int = DEFAULT_POLL_LIMIT,
        as_of: datetime | None = None,
    ) -> ContractStatsRunSummary:
        now = (as_of or datetime.now(UTC)).astimezone(UTC)
        fetched_total = 0
        inserted_total = 0
        processed = 0
        errors: list[str] = []

        for raw_symbol in self._symbols:
            try:
                mapping, contract_id = to_gate_contract_symbol(raw_symbol)
                raw_items = await self._client.fetch_contract_stats(
                    contract_id,
                    interval=self._interval,
                    limit=limit,
                )
                observations = normalize_contract_stats_payload(
                    raw_items,
                    symbol=mapping,
                    provider_symbol=contract_id,
                    interval=self._interval,
                    received_time=now,
                    venue=self._venue,
                    as_of=now,
                    closed_only=True,
                    backfill=False,
                )
                fetched_total += len(observations)
                inserted_total += self._repository.save_contract_stats(observations)
                processed += 1
            except Exception as exc:
                logger.warning("contract_stats poll failed for %s: %s", raw_symbol, exc)
                errors.append(f"{raw_symbol}: {exc}")

        return ContractStatsRunSummary(
            venue=self._venue,
            interval=self._interval,
            symbols_processed=processed,
            fetched_rows=fetched_total,
            inserted_rows=inserted_total,
            skipped_duplicates=max(0, fetched_total - inserted_total),
            errors=tuple(errors),
        )

    async def backfill_history(
        self,
        *,
        lookback_days: int = 179,
        as_of: datetime | None = None,
        page_limit: int = DEFAULT_CONTRACT_STATS_PAGE_LIMIT,
    ) -> ContractStatsRunSummary:
        if lookback_days < 1 or lookback_days > 180:
            raise ValueError("lookback_days must be within 1..180 for Gate contract_stats")
        now = (as_of or datetime.now(UTC)).astimezone(UTC)
        max_earliest = now - GATE_CONTRACT_STATS_MAX_LOOKBACK
        requested_earliest = now - timedelta(days=lookback_days)
        start_dt = max(requested_earliest, max_earliest)
        step = GATE_CONTRACT_STATS_INTERVALS[self._interval]
        step_seconds = int(step.total_seconds())
        end_epoch = int((now - step).timestamp())

        fetched_total = 0
        inserted_total = 0
        processed = 0
        errors: list[str] = []

        for raw_symbol in self._symbols:
            try:
                mapping, contract_id = to_gate_contract_symbol(raw_symbol)
                cursor_epoch = int(_floor_to_grid(start_dt, step).timestamp())
                seen_epochs: set[int] = set()
                while cursor_epoch <= end_epoch:
                    raw_items = await self._client.fetch_contract_stats(
                        contract_id,
                        interval=self._interval,
                        from_time=cursor_epoch,
                        limit=page_limit,
                    )
                    if not raw_items:
                        break
                    observations = normalize_contract_stats_payload(
                        raw_items,
                        symbol=mapping,
                        provider_symbol=contract_id,
                        interval=self._interval,
                        received_time=now,
                        venue=self._venue,
                        as_of=now,
                        closed_only=True,
                        backfill=True,
                    )
                    if observations:
                        fetched_total += len(observations)
                        inserted_total += self._repository.save_contract_stats(observations)
                    raw_epochs = [int(item["time"]) for item in raw_items if "time" in item]
                    if not raw_epochs:
                        break
                    max_epoch = max(raw_epochs)
                    if max_epoch in seen_epochs and max_epoch <= cursor_epoch:
                        break
                    seen_epochs.update(raw_epochs)
                    next_cursor = max_epoch + step_seconds
                    if next_cursor <= cursor_epoch:
                        break
                    cursor_epoch = next_cursor
                    if len(raw_items) < max(2, page_limit - 5):
                        break
                processed += 1
            except Exception as exc:
                logger.warning("contract_stats backfill failed for %s: %s", raw_symbol, exc)
                errors.append(f"{raw_symbol}: {exc}")

        return ContractStatsRunSummary(
            venue=self._venue,
            interval=self._interval,
            symbols_processed=processed,
            fetched_rows=fetched_total,
            inserted_rows=inserted_total,
            skipped_duplicates=max(0, fetched_total - inserted_total),
            errors=tuple(errors),
        )

    def generate_gap_report(
        self,
        *,
        lookback_days: int = 7,
        as_of: datetime | None = None,
    ) -> ContractStatsGapReport:
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        now = (as_of or datetime.now(UTC)).astimezone(UTC)
        window_start = now - timedelta(days=lookback_days)
        reports: list[SymbolContractStatsGapReport] = []
        for raw_symbol in self._symbols:
            mapping, _ = to_gate_contract_symbol(raw_symbol)
            observed_times = self._repository.list_contract_stats_times(
                venue=self._venue,
                canonical_symbol=mapping.canonical_symbol,
                interval=self._interval,
                start_time=window_start,
                end_time=now,
            )
            reports.append(
                compute_symbol_gap_report(
                    observed_times,
                    venue=self._venue,
                    canonical_symbol=mapping.canonical_symbol,
                    interval=self._interval,
                    window_start=window_start,
                    window_end=now,
                )
            )
        return ContractStatsGapReport(
            generated_at=now,
            venue=self._venue,
            interval=self._interval,
            lookback_days=lookback_days,
            symbols=tuple(reports),
        )


class ContractStatsScheduler:
    """Async background loop running startup backfill followed by periodic 5m polls."""

    def __init__(
        self,
        recorder: ContractStatsRecorder,
        *,
        interval_seconds: int = 300,
        backfill_days: int = 179,
        run_backfill_on_start: bool = True,
    ) -> None:
        if interval_seconds < 10:
            raise ValueError("interval_seconds must be >= 10")
        self._recorder = recorder
        self._interval_seconds = interval_seconds
        self._backfill_days = backfill_days
        self._run_backfill_on_start = run_backfill_on_start
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def request_stop(self) -> None:
        self._stop_event.set()

    async def run_loop(self) -> None:
        if self._run_backfill_on_start:
            try:
                summary = await self._recorder.backfill_history(lookback_days=self._backfill_days)
                logger.info(
                    "Gate contract_stats startup backfill: symbols=%s fetched=%s "
                    "inserted=%s dupes=%s",
                    summary.symbols_processed,
                    summary.fetched_rows,
                    summary.inserted_rows,
                    summary.skipped_duplicates,
                )
            except Exception:
                logger.exception("Gate contract_stats startup backfill failed")

        while not self._stop_event.is_set():
            try:
                summary = await self._recorder.poll_once()
                logger.info(
                    "Gate contract_stats poll: symbols=%s fetched=%s inserted=%s errors=%s",
                    summary.symbols_processed,
                    summary.fetched_rows,
                    summary.inserted_rows,
                    len(summary.errors),
                )
            except Exception:
                logger.exception("Gate contract_stats scheduled poll failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                continue

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self.run_loop(), name="gate-contract-stats-scheduler")
        return self._task

    async def stop(self) -> None:
        self.request_stop()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
