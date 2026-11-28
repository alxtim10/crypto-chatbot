from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from crypto_chatbot import cli
from crypto_chatbot.analysis.backtest import BacktestConfig, run_setup_backtest
from crypto_chatbot.market_data.gate_funding import (
    GateFundingHistoryClient,
    lookup_causal_funding_at,
    normalize_gate_funding_rate_payload,
    sync_gate_funding_history,
    to_backtest_funding_tuples,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol, resolve_mvp_spot_symbol
from crypto_chatbot.schemas.derivatives import (
    FundingRateObservation,
    LiquidationEvent,
    OpenInterestObservation,
    OrderBookSnapshot,
)
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"


class _MemDerivRepo:
    def __init__(self) -> None:
        self.funding: dict[tuple[str, str, datetime], FundingRateObservation] = {}

    def save_funding_observations(self, observations: Sequence[FundingRateObservation]) -> int:
        inserted = 0
        for obs in observations:
            key = (obs.venue, obs.canonical_symbol, obs.observed_time)
            if key not in self.funding:
                self.funding[key] = obs
                inserted += 1
        return inserted

    def load_funding_observations(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FundingRateObservation, ...]:
        matched = [
            obs
            for obs in self.funding.values()
            if obs.venue == venue
            and obs.canonical_symbol == canonical_symbol
            and (end_available_time is None or obs.available_time <= end_available_time)
        ]
        matched.sort(key=lambda o: o.observed_time)
        if limit is not None:
            matched = matched[-limit:]
        return tuple(matched)

    def save_open_interest_observations(
        self, observations: Sequence[OpenInterestObservation]
    ) -> int:
        del observations
        return 0

    def load_open_interest_observations(self, **kwargs: Any) -> tuple[OpenInterestObservation, ...]:
        del kwargs
        return ()

    def save_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> bool:
        del snapshot
        return False

    def load_order_book_snapshots(self, **kwargs: Any) -> tuple[OrderBookSnapshot, ...]:
        del kwargs
        return ()

    def save_liquidation_events(self, events: Sequence[LiquidationEvent]) -> int:
        del events
        return 0

    def load_liquidation_events(self, **kwargs: Any) -> tuple[LiquidationEvent, ...]:
        del kwargs
        return ()


@pytest.mark.asyncio
async def test_sync_over_90_days_btc_funding_paginated_idempotently() -> None:
    anchor = datetime(2026, 11, 27, 16, 0, tzinfo=UTC)
    # 95 days of 8h settlements = 285 records (> 3 pages of 90 each!)
    all_items: list[dict[str, Any]] = [
        {
            "t": int((anchor - timedelta(hours=8 * i)).timestamp()),
            "r": f"{0.00005 + (i % 7) * 0.00001:.7f}",
        }
        for i in range(285)
    ]

    pages_requested = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pages_requested
        pages_requested += 1
        to_val = int(request.url.params["to"])
        eligible = [item for item in all_items if int(item["t"]) <= to_val]
        # Gate /funding_rate caps pages at 90 rows descending
        return httpx.Response(200, json=eligible[:90])

    client = GateFundingHistoryClient(transport=httpx.MockTransport(handler))
    repo = _MemDerivRepo()
    try:
        first_counts = await sync_gate_funding_history(
            ["BTC"],
            client=client,
            repository=repo,
            days=90,
            as_of=anchor,
        )
        second_counts = await sync_gate_funding_history(
            ["BTC"],
            client=client,
            repository=repo,
            days=90,
            as_of=anchor,
        )
    finally:
        await client.close()

    assert pages_requested >= 4
    # 90 days * 3/day = 270..271 inclusive settlements loaded cleanly
    assert first_counts["BTC/USDT:USDT"] >= 270
    assert second_counts["BTC/USDT:USDT"] == 0
    loaded = repo.load_funding_observations(venue="gate", canonical_symbol="BTC/USDT:USDT")
    assert len(loaded) >= 270
    assert (loaded[-1].observed_time - loaded[0].observed_time) >= timedelta(days=89)


def test_point_in_time_funding_lookup_is_strictly_causal() -> None:
    symbol = resolve_default_perp_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    raw = [
        {"t": int((t0 + timedelta(hours=8 * i)).timestamp()), "r": f"{0.0001 * (i + 1):.6f}"}
        for i in range(15)
    ]
    all_obs = normalize_gate_funding_rate_payload(
        raw,
        symbol=symbol,
        provider_symbol="GATE:BTCUSDT.P",
        received_time=t0 + timedelta(days=10),
    )
    as_of = t0 + timedelta(hours=8 * 5)  # cutoff after 6th point (idx 0..5)

    repo = _MemDerivRepo()
    repo.save_funding_observations(all_obs)

    causal_prefix = repo.load_funding_observations(
        venue="gate",
        canonical_symbol="BTC/USDT:USDT",
        end_available_time=as_of,
    )
    assert len(causal_prefix) == 6
    assert all(obs.available_time <= as_of for obs in causal_prefix)

    feats_with_future = lookup_causal_funding_at(all_obs, as_of=as_of)
    feats_prefix_only = lookup_causal_funding_at(causal_prefix, as_of=as_of)
    assert feats_with_future == feats_prefix_only
    assert feats_with_future.observation_count == 6
    assert feats_with_future.latest_rate_8h == pytest.approx(0.0006)
    assert feats_with_future.zscore_30d is not None


def test_backtest_uses_causal_historical_funding_and_is_prefix_invariant() -> None:
    raw_ohlcv = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"][:750]
    spot = resolve_mvp_spot_symbol("BTC")
    perp = resolve_default_perp_symbol("BTC")
    batch = normalize_ohlcv_rows(
        raw_ohlcv,
        symbol=spot,
        timeframe=Timeframe.H4,
        received_time=datetime(2026, 11, 27, 0, 0, tzinfo=UTC),
        now=datetime(2026, 11, 27, 0, 0, tzinfo=UTC),
    )
    perp_candles: list[MarketCandle] = [
        c.model_copy(
            update={
                "provider": "gate_rest",
                "venue": "gate",
                "canonical_symbol": perp.canonical_symbol,
                "provider_symbol": "GATE:BTCUSDT.P",
                "market_type": MarketType.PERPETUAL,
            }
        )
        for c in batch.candles
    ]
    start = perp_candles[0].open_time
    end = perp_candles[-1].close_time

    # Build 8h funding settlements over the exact candle horizon plus future decoy observations
    obs_list: list[FundingRateObservation] = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur <= end + timedelta(days=5):
        rate = 0.0002 if cur <= end else 0.9999  # wild outlier strictly after sample end
        obs_list.append(
            FundingRateObservation(
                provider="gate_archive",
                venue="gate",
                canonical_symbol=perp.canonical_symbol,
                provider_symbol="GATE:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                observed_time=cur,
                received_time=cur,
                available_time=cur,
                funding_rate=rate,
                funding_interval_hours=8.0,
                rate_per_8h=rate,
                annualized_rate=rate * 3 * 365,
            )
        )
        cur += timedelta(hours=8)

    in_sample = [o for o in obs_list if o.available_time <= end]
    cfg = BacktestConfig(warmup_bars=400, max_hold_bars=24)
    rep_with_future = run_setup_backtest(
        perp_candles,
        config=cfg,
        funding_history=obs_list,
        use_available_time=True,
    )
    rep_causal_only = run_setup_backtest(
        perp_candles,
        config=cfg,
        funding_history=in_sample,
        use_available_time=True,
    )
    assert rep_with_future == rep_causal_only
    if rep_causal_only.executed_trades > 0:
        assert rep_causal_only.assumed_funding_fraction < 1.0

    pairs = to_backtest_funding_tuples(obs_list, as_of=end)
    assert all(ts <= end for ts, _ in pairs)


def test_cli_sync_gate_funding_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _DummyEngine:
        def dispose(self) -> None:
            return None

    class _FakeClient:
        async def fetch_funding_history(
            self, contract: str, *, days: int = 90, end_time: datetime | None = None
        ) -> list[dict[str, Any]]:
            del contract, days, end_time
            t0 = int(datetime(2026, 11, 20, 0, 0, tzinfo=UTC).timestamp())
            return [{"t": t0, "r": "0.0001"}]

        async def close(self) -> None:
            return None

    repo = _MemDerivRepo()
    monkeypatch.setattr(cli, "create_db_engine", lambda url: _DummyEngine())
    monkeypatch.setattr(cli, "init_db", lambda engine: None)
    monkeypatch.setattr(cli, "PostgresDerivativesRepository", lambda engine: repo)

    import crypto_chatbot.market_data.gate_funding as gf

    monkeypatch.setattr(gf, "GateFundingHistoryClient", _FakeClient)

    rc = cli.main(
        [
            "sync-gate-funding",
            "--symbols",
            "BTC",
            "--days",
            "90",
            "--database-url",
            "postgresql+psycopg://u:p@localhost/db",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["days"] == 90
    assert out["inserted"]["BTC/USDT:USDT"] == 1
