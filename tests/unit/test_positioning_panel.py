from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_chatbot.analysis.positioning import compute_positioning_panel
from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.schemas.derivatives import ContractStatsObservation
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    Timeframe,
)


def _obs(
    t: datetime,
    *,
    top_lsr_account: float = 0.45,
    lsr_taker: float = 1.25,
    open_interest_usd: float = 3_000_000_000.0,
    long_liq_usd: float = 10_000.0,
    short_liq_usd: float = 5_000.0,
) -> ContractStatsObservation:
    avail = t + timedelta(minutes=5)
    return ContractStatsObservation(
        venue="gateio",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        interval="5m",
        observed_time=t,
        received_time=avail,
        available_time=avail,
        top_lsr_account=top_lsr_account,
        lsr_taker=lsr_taker,
        lsr_account=1.4,
        top_lsr_size=0.95,
        open_interest=500_000_000.0,
        open_interest_usd=open_interest_usd,
        long_liq_usd=long_liq_usd,
        short_liq_usd=short_liq_usd,
    )


def test_positioning_panel_warms_up_under_7_days_and_excludes_future() -> None:
    anchor = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
    # Only 3 days of recorded history (< 7 days warmup threshold) + future leak trap
    obs = [
        _obs(anchor - timedelta(hours=72 - i), top_lsr_account=0.40 + 0.01 * i) for i in range(72)
    ]
    future_trap = _obs(anchor + timedelta(hours=1), top_lsr_account=9.99)

    panel = compute_positioning_panel(
        [*obs, future_trap],
        canonical_symbol="BTC/USDT:USDT",
        as_of=anchor,
        use_available_time=True,
    )
    assert panel.parity_mode == "FORWARD_ONLY"
    assert panel.status == "WARMING_UP"
    assert panel.observation_count == 72
    assert panel.top_lsr_account == pytest.approx(obs[-1].top_lsr_account)
    assert panel.top_lsr_account_pct_30d is None
    assert panel.top_lsr_account_z_30d is None
    assert panel.note is not None
    assert "pemanasan" in panel.note.lower()


def test_positioning_panel_ready_after_7_days_with_30d_percentile_zscore_oi_and_liq() -> None:
    anchor = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
    # 10 days hourly samples (span >= 7 days)
    obs: list[ContractStatsObservation] = []
    for h in range(240):
        ts = anchor - timedelta(hours=240 - h)
        top_lsr = 0.35 + (h / 240.0) * 0.40  # monotonically rising -> latest near P100
        taker = 0.8 + (h % 10) * 0.1
        oi_usd = 2_000_000_000.0 if h < 216 else 2_200_000_000.0  # +10% 24h OI jump
        obs.append(
            _obs(
                ts,
                top_lsr_account=top_lsr,
                lsr_taker=taker,
                open_interest_usd=oi_usd,
                long_liq_usd=100_000.0,
                short_liq_usd=25_000.0,
            )
        )

    panel = compute_positioning_panel(
        obs,
        canonical_symbol="BTC/USDT:USDT",
        as_of=anchor,
        use_available_time=True,
    )
    assert panel.parity_mode == "FORWARD_ONLY"
    assert panel.status == "READY"
    assert panel.summary_label == "crowded_long"
    assert panel.top_lsr_account_pct_30d is not None
    assert panel.top_lsr_account_pct_30d >= 95.0
    assert panel.top_lsr_account_z_30d is not None
    assert panel.top_lsr_account_z_30d > 1.0
    assert panel.lsr_taker_pct_30d is not None
    assert panel.oi_24h_delta_pct == pytest.approx(10.0)
    assert panel.liq_skew_24h == pytest.approx((100_000.0 - 25_000.0) / 125_000.0)


@pytest.mark.asyncio
async def test_analysis_service_and_templates_render_forward_only_positioning() -> None:
    anchor = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)

    class _Provider:
        async def get_closed_ohlcv(
            self, symbol: str, timeframe: Timeframe, limit: int = DEFAULT_CANDLE_LIMIT
        ) -> OhlcvBatch:
            del symbol
            candles: list[MarketCandle] = []
            for i in range(max(limit, 210)):
                close_t = anchor - (max(limit, 210) - 1 - i) * timeframe.duration
                open_t = close_t - timeframe.duration
                p = 60000.0 + i * 5.0
                candles.append(
                    MarketCandle(
                        provider="gate_rest",
                        venue="gate",
                        canonical_symbol="BTC/USDT:USDT",
                        provider_symbol="GATE:BTCUSDT.P",
                        market_type=MarketType.PERPETUAL,
                        timeframe=timeframe,
                        open_time=open_t,
                        close_time=close_t,
                        received_time=anchor,
                        available_time=close_t,
                        open=p,
                        high=p + 10.0,
                        low=p - 10.0,
                        close=p + 4.0,
                        base_volume=100.0,
                        base_unit="BTC",
                        quote_currency="USDT",
                        is_closed=True,
                    )
                )
            return OhlcvBatch(
                metadata=ProviderMetadata(
                    provider="gate_rest",
                    venue="gate",
                    market_type=MarketType.PERPETUAL,
                    base_unit="BTC",
                    quote_currency="USDT",
                ),
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                timeframe=timeframe,
                received_time=anchor,
                candles=tuple(candles),
            )

        async def close(self) -> None:
            return None

    class _StatsRepo:
        def __init__(self, items: Sequence[ContractStatsObservation]) -> None:
            self._items = tuple(items)

        def save_contract_stats(self, observations: Sequence[ContractStatsObservation]) -> int:
            del observations
            return 0

        def load_contract_stats(
            self,
            *,
            venue: str,
            canonical_symbol: str,
            interval: str = "5m",
            start_time: datetime | None = None,
            end_available_time: datetime | None = None,
            limit: int | None = None,
        ) -> tuple[ContractStatsObservation, ...]:
            del interval, start_time, limit
            return tuple(
                o
                for o in self._items
                if o.venue == venue
                and o.canonical_symbol == canonical_symbol
                and (end_available_time is None or o.available_time <= end_available_time)
            )

        def list_contract_stats_times(self, **kwargs: Any) -> tuple[datetime, ...]:
            del kwargs
            return ()

    stats_items = [
        _obs(anchor - timedelta(hours=240 - h), top_lsr_account=0.40 + h * 0.002)
        for h in range(240)
    ]
    service = AnalysisService(
        provider=_Provider(),
        contract_stats_repository=_StatsRepo(stats_items),
    )
    snapshot = await service.analyze("BTC", Timeframe.H4, include_mtf=False, persist=False)
    assert snapshot.positioning is not None
    assert snapshot.positioning.status == "READY"
    assert snapshot.positioning.parity_mode == "FORWARD_ONLY"

    full_txt = render_deterministic_snapshot(snapshot, html_mode=False)
    compact_txt = render_compact_snapshot(snapshot, html_mode=False)
    assert "FORWARD_ONLY" in full_txt
    assert "Top L/S:" in full_txt
    assert "Positioning (FORWARD_ONLY)" in compact_txt
