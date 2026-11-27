"""Gate.io perpetual risk limit & maintenance margin tier client (D-08)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from crypto_chatbot.market_data.symbols import SymbolError, resolve_default_perp_symbol
from crypto_chatbot.market_data.venue_health import VenueHealthCache, is_venue_block_error

logger = logging.getLogger(__name__)

DEFAULT_GATE_API_URL = "https://api.gateio.ws/api/v4"
DEFAULT_RISK_TIERS_TTL_SECONDS = 3600.0


class _RetryableRiskTiersError(RuntimeError):
    """Transient HTTP error from Gate `/futures/usdt/risk_limit_tiers`."""


class GateRiskLimitTier(BaseModel):
    """Normalized entry from Gate `GET /futures/usdt/risk_limit_tiers` (D-08)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: str
    tier: int = Field(ge=1)
    risk_limit_usdt: float = Field(gt=0.0)
    maintenance_rate: float = Field(ge=0.0, lt=1.0)
    initial_rate: float = Field(ge=0.0, le=1.0)
    leverage_max: float = Field(gt=0.0)
    deduction_usdt: float = Field(ge=0.0)


def parse_gate_risk_limit_tiers(
    raw_tiers: Sequence[Mapping[str, Any]],
    *,
    contract: str,
) -> tuple[GateRiskLimitTier, ...]:
    """Parse and sort ascending Gate risk limit / maintenance margin tiers."""
    contract_norm = contract.strip().upper()
    parsed: list[GateRiskLimitTier] = []
    for item in raw_tiers:
        if not isinstance(item, Mapping):
            continue
        try:
            tier_num = int(item["tier"])
            risk_limit = float(item["risk_limit"])
            mmr = float(item["maintenance_rate"])
            imr = float(item.get("initial_rate", 0.0))
            lev_max = float(item["leverage_max"])
            deduction = float(item.get("deduction", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if risk_limit <= 0.0 or mmr < 0.0 or mmr >= 1.0 or lev_max <= 0.0:
            continue
        parsed.append(
            GateRiskLimitTier(
                contract=contract_norm,
                tier=tier_num,
                risk_limit_usdt=risk_limit,
                maintenance_rate=mmr,
                initial_rate=max(0.0, min(1.0, imr)),
                leverage_max=lev_max,
                deduction_usdt=max(0.0, deduction),
            )
        )
    parsed.sort(key=lambda t: (t.risk_limit_usdt, t.tier))
    return tuple(parsed)


def select_risk_limit_tier(
    tiers: Sequence[GateRiskLimitTier],
    *,
    notional_usdt: float = 0.0,
) -> GateRiskLimitTier | None:
    """Return the lowest ascending tier covering `notional_usdt` (or top tier if exceeded)."""
    if not tiers:
        return None
    target = max(0.0, float(notional_usdt))
    ordered = sorted(tiers, key=lambda t: (t.risk_limit_usdt, t.tier))
    for entry in ordered:
        if target <= entry.risk_limit_usdt:
            return entry
    return ordered[-1]


def resolve_tier_mmr(
    tiers: Sequence[GateRiskLimitTier],
    *,
    notional_usdt: float = 0.0,
) -> tuple[float | None, float | None]:
    """Return `(maintenance_rate, leverage_max)` for the tier covering `notional_usdt`."""
    selected = select_risk_limit_tier(tiers, notional_usdt=notional_usdt)
    if selected is None:
        return None, None
    return selected.maintenance_rate, selected.leverage_max


def to_gate_perp_contract_name(symbol: str) -> str:
    """Convert an internal symbol (`BTC`, `BTC/USDT:USDT`, `1000PEPE`) to Gate contract name."""
    mapping = resolve_default_perp_symbol(symbol)
    return f"{mapping.base.upper()}_{mapping.quote.upper()}"


class GateRiskTiersClient:
    """Cached async client for Gate `GET /futures/usdt/risk_limit_tiers`."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_GATE_API_URL,
        timeout_seconds: float = 10.0,
        max_attempts: int = 2,
        cache_ttl_seconds: float = DEFAULT_RISK_TIERS_TTL_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        health_cache: VenueHealthCache | None = None,
        mono_clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )
        self._max_attempts = max(1, max_attempts)
        self._cache_ttl = max(0.0, float(cache_ttl_seconds))
        self._health_cache = health_cache
        self._mono_clock = mono_clock or time.monotonic
        self._cache: dict[str, tuple[float, tuple[GateRiskLimitTier, ...]]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_tiers(
        self,
        contract: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[GateRiskLimitTier, ...]:
        """Fetch and cache ascending risk limit tiers for a Gate USDT contract."""
        contract_id = contract.strip().upper()
        now_m = self._mono_clock()
        if not force_refresh and contract_id in self._cache:
            cached_ts, cached_tiers = self._cache[contract_id]
            if (now_m - cached_ts) < self._cache_ttl:
                return cached_tiers

        if self._health_cache is not None and not self._health_cache.is_available("gate"):
            raise RuntimeError("gate venue in cooldown")

        retrying = AsyncRetrying(
            retry=retry_if_exception(
                lambda exc: (
                    isinstance(exc, (httpx.TransportError, _RetryableRiskTiersError))
                    and not is_venue_block_error(exc)
                )
            ),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.3, min=0.2, max=1.5),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    resp = await self._client.get(
                        "/futures/usdt/risk_limit_tiers",
                        params={"contract": contract_id},
                    )
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        raise _RetryableRiskTiersError(
                            f"Gate risk_limit_tiers HTTP {resp.status_code}"
                        )
                    resp.raise_for_status()
                    payload = resp.json()
                    if not isinstance(payload, list):
                        raise ValueError("unexpected Gate /risk_limit_tiers payload type")
                    tiers = parse_gate_risk_limit_tiers(payload, contract=contract_id)
                    self._cache[contract_id] = (self._mono_clock(), tiers)
                    return tiers
        except Exception as exc:
            if self._health_cache is not None and is_venue_block_error(exc):
                self._health_cache.mark_unavailable("gate", reason=str(exc) or type(exc).__name__)
            raise
        raise RuntimeError("unreachable")  # pragma: no cover

    async def get_mmr_for_symbol(
        self,
        symbol: str,
        *,
        notional_usdt: float = 0.0,
    ) -> tuple[float | None, float | None]:
        """Best-effort `(maintenance_rate, leverage_max)` lookup; never raises into callers."""
        try:
            contract_id = to_gate_perp_contract_name(symbol)
        except SymbolError:
            return None, None
        try:
            tiers = await self.fetch_tiers(contract_id)
        except Exception as exc:
            logger.debug("Gate risk_limit_tiers lookup skipped for %s: %s", contract_id, exc)
            return None, None
        return resolve_tier_mmr(tiers, notional_usdt=notional_usdt)
