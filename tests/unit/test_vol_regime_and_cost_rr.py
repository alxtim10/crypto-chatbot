from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from crypto_chatbot.analysis.regimes import classify_volatility_regime
from crypto_chatbot.analysis.setup import _net_reward_risk
from crypto_chatbot.schemas.analysis import VolatilityRegime
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe
from crypto_chatbot.schemas.ml import CostAssumptions


def _candles_from_closes(closes: list[float]) -> list[MarketCandle]:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[MarketCandle] = []
    for i, close_p in enumerate(closes):
        open_p = closes[i - 1] if i > 0 else close_p
        high_p = max(open_p, close_p) + 1.0
        low_p = max(1.0, min(open_p, close_p) - 1.0)
        ot = t0 + i * Timeframe.H1.duration
        ct = ot + Timeframe.H1.duration
        out.append(
            MarketCandle(
                provider="ccxt",
                venue="binance",
                canonical_symbol="BTC/USDT",
                provider_symbol="BINANCE:BTCUSDT",
                market_type=MarketType.SPOT,
                timeframe=Timeframe.H1,
                open_time=ot,
                close_time=ct,
                received_time=ct + timedelta(seconds=1),
                available_time=ct + timedelta(seconds=5),
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                base_volume=100.0,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return out


def test_vol_regime_insufficient_history() -> None:
    candles = _candles_from_closes([100.0 + i * 0.1 for i in range(50)])
    result = classify_volatility_regime(candles)
    assert result.regime is None


def test_vol_regime_high_when_recent_vol_spikes() -> None:
    rng = np.random.default_rng(3)
    # Calm history then a volatile tail.
    calm = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, 140)))
    volatile = calm[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.03, 40)))
    closes = list(calm) + list(volatile)
    result = classify_volatility_regime(_candles_from_closes(closes))
    assert result.regime is VolatilityRegime.HIGH
    assert result.percentile is not None
    assert result.percentile >= 70.0


def test_vol_regime_low_when_recent_vol_drops() -> None:
    rng = np.random.default_rng(4)
    volatile = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 140)))
    calm = volatile[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.001, 40)))
    closes = list(volatile) + list(calm)
    result = classify_volatility_regime(_candles_from_closes(closes))
    assert result.regime is VolatilityRegime.LOW
    assert result.percentile is not None
    assert result.percentile <= 30.0


def test_net_reward_risk_below_gross() -> None:
    costs = CostAssumptions()
    cost_rate = costs.round_trip_cost_rate
    # entry 100, TP 104 (gross reward 4), SL 98 (gross risk 2) -> gross R:R = 2.0
    net = _net_reward_risk(entry_mid=100.0, target=104.0, stop_loss=98.0, cost_rate=cost_rate)
    assert net is not None
    gross = 4.0 / 2.0
    assert net < gross  # costs lower the net R:R


def test_net_reward_risk_zero_cost_matches_gross() -> None:
    net = _net_reward_risk(entry_mid=100.0, target=103.0, stop_loss=99.0, cost_rate=0.0)
    assert net is not None
    assert abs(net - (3.0 / 1.0)) < 1e-9


def test_net_reward_risk_handles_cost_exceeding_reward() -> None:
    # Huge cost rate drives net reward negative -> clamped to 0.0 (never negative).
    net = _net_reward_risk(entry_mid=100.0, target=100.5, stop_loss=99.5, cost_rate=0.10)
    assert net == 0.0
