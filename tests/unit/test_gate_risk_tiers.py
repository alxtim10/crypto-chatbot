from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.analysis.setup import compute_leverage_ceiling
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.gate_risk_tiers import (
    GateRiskTiersClient,
    parse_gate_risk_limit_tiers,
    resolve_tier_mmr,
    select_risk_limit_tier,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.market import MarketCandle, MarketType, ProviderMetadata, Timeframe

# Verified GET /futures/usdt/risk_limit_tiers?contract=BTC_USDT payload (gate-data-audit.md §1.5)
SAMPLE_BTC_TIERS = [
    {
        "tier": 1,
        "risk_limit": "500000",
        "maintenance_rate": "0.003",
        "initial_rate": "0.005",
        "leverage_max": "200",
        "deduction": "0",
    },
    {
        "tier": 2,
        "risk_limit": "1000000",
        "maintenance_rate": "0.0035",
        "initial_rate": "0.006666",
        "leverage_max": "150.01",
        "deduction": "250",
    },
    {
        "tier": 3,
        "risk_limit": "5000000",
        "maintenance_rate": "0.015",
        "initial_rate": "0.05",
        "leverage_max": "20",
        "deduction": "5000",
    },
    {
        "tier": 4,
        "risk_limit": "10000000",
        "maintenance_rate": "0.05",
        "initial_rate": "0.10",
        "leverage_max": "10",
        "deduction": "100000",
    },
]


def test_parse_and_select_gate_risk_limit_tier() -> None:
    tiers = parse_gate_risk_limit_tiers(list(reversed(SAMPLE_BTC_TIERS)), contract="BTC_USDT")
    assert [t.tier for t in tiers] == [1, 2, 3, 4]
    assert tiers[0].risk_limit_usdt == 500_000.0
    assert tiers[0].maintenance_rate == 0.003
    assert tiers[1].leverage_max == 150.01

    t1 = select_risk_limit_tier(tiers, notional_usdt=1_000.0)
    assert t1 is not None
    assert t1.tier == 1
    assert resolve_tier_mmr(tiers, notional_usdt=500_000.0) == (0.003, 200.0)

    t2 = select_risk_limit_tier(tiers, notional_usdt=750_000.0)
    assert t2 is not None
    assert t2.tier == 2
    assert resolve_tier_mmr(tiers, notional_usdt=750_000.0) == (0.0035, 150.01)

    t4 = select_risk_limit_tier(tiers, notional_usdt=50_000_000.0)
    assert t4 is not None
    assert t4.tier == 4
    assert resolve_tier_mmr(tiers, notional_usdt=50_000_000.0) == (0.05, 10.0)


def test_leverage_ceiling_respects_mmr_and_venue_max() -> None:
    # Tight 2% stop + 0.06% cost -> uncapped denom ~ 0.0274 + mmr
    lev_tier1, assumed_f = compute_leverage_ceiling(
        0.02, 0.0006, maintenance_margin_rate=0.003, venue_max_leverage=200.0
    )
    assert assumed_f is False
    assert lev_tier1 == 20  # capped by PRD 20x safety bound

    # Low venue max leverage (e.g. high notional tier or thin altcoin) overrides 20x cap
    lev_tier4, assumed_f2 = compute_leverage_ceiling(
        0.02, 0.0006, maintenance_margin_rate=0.05, venue_max_leverage=10.0
    )
    assert assumed_f2 is False
    assert lev_tier4 == 10

    # Fallback when MMR unknown sets mmr_assumed=True and uses DEFAULT_ASSUMED_MMR (0.01)
    _, assumed_t = compute_leverage_ceiling(0.02, 0.0006, maintenance_margin_rate=None)
    assert assumed_t is True


@pytest.mark.asyncio
async def test_gate_risk_tiers_client_caches_and_degrades_gracefully() -> None:
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/api/v4/futures/usdt/risk_limit_tiers"
        if request.url.params.get("contract") == "BTC_USDT":
            return httpx.Response(200, json=SAMPLE_BTC_TIERS)
        return httpx.Response(400, json={"label": "INVALID_CONTRACT"})

    clock_val = 100.0
    client = GateRiskTiersClient(
        transport=httpx.MockTransport(_handler),
        cache_ttl_seconds=3600.0,
        mono_clock=lambda: clock_val,
    )
    try:
        mmr1, max1 = await client.get_mmr_for_symbol("BTC", notional_usdt=25_000.0)
        assert mmr1 == 0.003
        assert max1 == 200.0
        # Cached on second call
        mmr2, _ = await client.get_mmr_for_symbol("BTC/USDT:USDT", notional_usdt=900_000.0)
        assert mmr2 == 0.0035
        assert calls == 1

        # Unknown contract error returns (None, None) without raising
        mmr_err, max_err = await client.get_mmr_for_symbol("ETH")
        assert mmr_err is None
        assert max_err is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_analysis_service_uses_gate_mmr_and_clears_mmr_assumed() -> None:
    symbol = resolve_default_perp_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    received = t0 + timedelta(hours=250)
    # Construct structured trending series cleanly so setup engine reaches non-error path
    candles = tuple(
        MarketCandle(
            provider="gate_rest",
            venue="gate",
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol="GATE:BTCUSDT.P",
            market_type=MarketType.PERPETUAL,
            timeframe=Timeframe.H1,
            open_time=t0 + timedelta(hours=i),
            close_time=t0 + timedelta(hours=i + 1),
            received_time=received,
            available_time=t0 + timedelta(hours=i + 1),
            open=60000.0 + i * 10.0,
            high=60025.0 + i * 10.0,
            low=59985.0 + i * 10.0,
            close=60005.0 + i * 10.0,
            base_volume=10.0,
            quote_volume=(60005.0 + i * 10.0) * 10.0,
            taker_buy_base_volume=5.5,
            base_unit="BTC",
            quote_currency="USDT",
            is_closed=True,
        )
        for i in range(250)
    )

    class _Provider:
        async def get_closed_ohlcv(
            self, sym: str, tf: Timeframe, limit: int = DEFAULT_CANDLE_LIMIT
        ) -> OhlcvBatch:
            del sym, tf, limit
            return OhlcvBatch(
                metadata=ProviderMetadata(
                    provider="gate_rest",
                    venue="gate",
                    market_type=MarketType.PERPETUAL,
                    base_unit="BTC",
                    quote_currency="USDT",
                ),
                canonical_symbol=symbol.canonical_symbol,
                provider_symbol="GATE:BTCUSDT.P",
                timeframe=Timeframe.H1,
                received_time=received,
                candles=candles,
            )

        async def close(self) -> None:
            return None

    client = GateRiskTiersClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=SAMPLE_BTC_TIERS))
    )
    try:
        svc = AnalysisService(provider=_Provider(), risk_tiers_client=client)
        snap = await svc.analyze(
            "BTC",
            Timeframe.H1,
            limit=250,
            include_mtf=False,
            include_derivatives=False,
            persist=False,
        )
        assert snap.trade_setup is not None
        assert snap.trade_setup.mmr_assumed is False
    finally:
        await client.close()
