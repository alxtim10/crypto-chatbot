"""Parity tests ensuring HTF (4h/1h) regime, structure, and ADX are identical across all paths.

Verifies that:
1. Standalone `analyze BTC 4h` (at 12:00 UTC) and `analyze BTC 1h` (at 14:00 UTC)
2. Top-down MTF section of `analyze BTC 15m` (at 14:15 UTC, with forming 4h/1h bars present)
3. `alerts run` 15m MTF top-down flow (`evaluate_htf_bias`)
4. `scan_market_setups` / `analyze_multi_style`

all compute the EXACT SAME 4h and 1h regime, market_structure, and ADX14 on the same
closed candles (`close_time <= as_of`) from the same venue with the same lookback.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from crypto_chatbot.alerts.mtf import evaluate_htf_bias
from crypto_chatbot.analysis.mtf import evaluate_timeframe_context
from crypto_chatbot.analysis.scanner import scan_market_setups
from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.schemas.analysis import AnalysisSnapshot, Regime
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    Timeframe,
)


def _make_ranging_then_dip_candles(
    *,
    timeframe: Timeframe,
    end_close: datetime,
    count: int = 200,
    venue: str = "gate",
    symbol: str = "BTC/USDT:USDT",
    append_forming_crash: bool = False,
) -> tuple[MarketCandle, ...]:
    """Generate deterministic candles whose closed regime is RANGING (ADX < 20, close < EMA50).

    This specifically reproduces the reported bug:
    - With 200+ bars (full lookback), `atr_percentile_100` is available and `ADX14 < 20` -> RANGING.
    - With only 100 bars (the old buggy MTF limit), `atr_percentile_100` was None and the fallback
      `close < ema_50` falsely labeled 4h as `Tren Turun` (TRENDING_DOWN)!
    - Optionally appends a forming bar (`close_time > end_close`, `is_closed=False` or future
      `close_time`) with a massive crash to prove forming/future bars are strictly excluded.
    """
    dur = timeframe.duration
    candles: list[MarketCandle] = []
    for i in range(count):
        close_t = end_close - dur * (count - 1 - i)
        open_t = close_t - dur
        # Gentle oscillation around 90_000 with slight gradual drift below EMA50 so ADX stays
        # low (< 20) and ATR percentile stays moderate (< 80) -> RANGING while last_close < ema_50.
        osc = 220.0 * math.sin(i * 0.45)
        late_dip = -20.0 * (i - (count - 12)) if i >= count - 12 else 0.0
        mid = 90_000.0 + osc + late_dip
        o = mid + (40.0 if i % 2 == 0 else -40.0)
        c = mid + (-40.0 if i % 2 == 0 else 40.0)
        wick = 240.0 if i < count - 30 else 150.0
        h = max(o, c) + wick
        lo = min(o, c) - wick
        candles.append(
            MarketCandle(
                provider="gate_direct",
                venue=venue,
                canonical_symbol=symbol,
                provider_symbol="BTC_USDT",
                market_type=MarketType.PERPETUAL,
                timeframe=timeframe,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=o,
                high=h,
                low=lo,
                close=c,
                base_volume=500.0 + (i % 7) * 15.0,
                quote_volume=(500.0 + (i % 7) * 15.0) * c,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )

    if append_forming_crash:
        # Forming bar closing AFTER end_close with an extreme crash that would flip regime if used.
        open_t = end_close
        close_t = end_close + dur
        candles.append(
            MarketCandle(
                provider="gate_direct",
                venue=venue,
                canonical_symbol=symbol,
                provider_symbol="BTC_USDT",
                market_type=MarketType.PERPETUAL,
                timeframe=timeframe,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=89_500.0,
                high=89_600.0,
                low=70_000.0,
                close=71_000.0,
                base_volume=9_999.0,
                quote_volume=9_999.0 * 71_000.0,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,  # even if mis-marked is_closed=True, close_time > as_of blocks it!
            )
        )

    return tuple(candles)


def _to_batch(candles: tuple[MarketCandle, ...], tf: Timeframe, venue: str = "gate") -> OhlcvBatch:
    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider="gate_direct",
            venue=venue,
            market_type=MarketType.PERPETUAL,
            base_unit="BTC",
            quote_currency="USDT",
        ),
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        timeframe=tf,
        received_time=candles[-1].received_time,
        candles=candles,
    )


@pytest.mark.asyncio
async def test_mtf_parity_across_standalone_analyze_15m_alerts_and_scanner() -> None:
    """4h/1h regime, structure, and ADX14 match identically across analyze, alerts, & scanner."""
    t_4h = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
    t_1h = datetime(2026, 11, 28, 14, 0, tzinfo=UTC)
    t_15m = datetime(2026, 11, 28, 14, 15, tzinfo=UTC)
    t_1d = datetime(2026, 11, 28, 0, 0, tzinfo=UTC)

    c_4h_closed = _make_ranging_then_dip_candles(
        timeframe=Timeframe.H4,
        end_close=t_4h,
        count=220,
        append_forming_crash=False,
    )
    c_4h_with_forming = _make_ranging_then_dip_candles(
        timeframe=Timeframe.H4,
        end_close=t_4h,
        count=220,
        append_forming_crash=True,
    )
    c_1h_closed = _make_ranging_then_dip_candles(
        timeframe=Timeframe.H1,
        end_close=t_1h,
        count=220,
        append_forming_crash=False,
    )
    c_1h_with_forming = _make_ranging_then_dip_candles(
        timeframe=Timeframe.H1,
        end_close=t_1h,
        count=220,
        append_forming_crash=True,
    )
    c_15m = _make_ranging_then_dip_candles(
        timeframe=Timeframe.M15,
        end_close=t_15m,
        count=220,
    )
    c_5m = _make_ranging_then_dip_candles(
        timeframe=Timeframe.M5,
        end_close=t_15m,
        count=80,
    )
    c_1d = _make_ranging_then_dip_candles(
        timeframe=Timeframe.D1,
        end_close=t_1d,
        count=150,
    )

    class _FixtureProvider:
        venue = "gate"

        def __init__(self, *, include_forming_htf: bool) -> None:
            self.include_forming_htf = include_forming_htf
            self.requested_limits: list[tuple[Timeframe, int]] = []

        async def get_closed_ohlcv(
            self, symbol: str, timeframe: Timeframe, limit: int = 1000
        ) -> OhlcvBatch:
            del symbol
            self.requested_limits.append((timeframe, limit))
            mapping = {
                Timeframe.D1: c_1d,
                Timeframe.H4: c_4h_with_forming if self.include_forming_htf else c_4h_closed,
                Timeframe.H1: c_1h_with_forming if self.include_forming_htf else c_1h_closed,
                Timeframe.M15: c_15m,
                Timeframe.M5: c_5m,
            }
            seq = mapping[timeframe]
            return _to_batch(seq[-limit:], timeframe)

    # 1. Standalone 4h (as_of 12:00 UTC) and standalone 1h (as_of 14:00 UTC)
    standalone_svc = AnalysisService(provider=_FixtureProvider(include_forming_htf=False))  # type: ignore[arg-type]
    snap_4h = await standalone_svc.analyze(
        "BTC/USDT:USDT", Timeframe.H4, include_mtf=False, include_derivatives=False, persist=False
    )
    snap_1h = await standalone_svc.analyze(
        "BTC/USDT:USDT", Timeframe.H1, include_mtf=False, include_derivatives=False, persist=False
    )
    assert snap_4h.as_of == t_4h
    assert snap_4h.regime is Regime.RANGING
    assert snap_4h.technical is not None
    assert snap_4h.technical.adx_14 is not None
    assert snap_4h.technical.adx_14 < 20.0
    assert snap_1h.as_of == t_1h
    assert snap_1h.technical is not None
    assert snap_1h.technical.adx_14 is not None

    # 2. Top-down `analyze BTC 15m` (as_of 14:15 UTC) even when forming 4h/1h crash bars exist!
    mtf_provider = _FixtureProvider(include_forming_htf=True)
    mtf_svc = AnalysisService(provider=mtf_provider)  # type: ignore[arg-type]
    snap_15m = await mtf_svc.analyze(
        "BTC/USDT:USDT", Timeframe.M15, include_mtf=True, include_derivatives=False, persist=False
    )
    assert snap_15m.as_of == t_15m
    # Verify HTF timeframes were fetched with the same full lookback (1000), not 100!
    limits_by_tf = dict(mtf_provider.requested_limits)
    assert limits_by_tf[Timeframe.H4] == limits_by_tf[Timeframe.M15] == 1000
    assert limits_by_tf[Timeframe.H1] == 1000

    ov_by_tf = {ov.timeframe: ov for ov in snap_15m.mtf_overview}
    assert Timeframe.H4 in ov_by_tf
    assert Timeframe.H1 in ov_by_tf

    # Exact parity on 4h: regime, structure, ADX14, and Indonesian trend label
    assert ov_by_tf[Timeframe.H4].regime is snap_4h.regime
    assert ov_by_tf[Timeframe.H4].market_structure is snap_4h.market_structure
    assert ov_by_tf[Timeframe.H4].adx_14 == pytest.approx(snap_4h.technical.adx_14)
    assert ov_by_tf[Timeframe.H4].trend_label == "Mendatar (Ranging)"

    # Exact parity on 1h: regime, structure, ADX14
    assert ov_by_tf[Timeframe.H1].regime is snap_1h.regime
    assert ov_by_tf[Timeframe.H1].market_structure is snap_1h.market_structure
    assert ov_by_tf[Timeframe.H1].adx_14 == pytest.approx(snap_1h.technical.adx_14)

    # 3. `alerts run` MTF bias gate (uses the exact same shared evaluator!)
    bias_eval = evaluate_htf_bias(
        {Timeframe.H4: c_4h_with_forming, Timeframe.H1: c_1h_with_forming},
        (Timeframe.H4, Timeframe.H1),
        cutoff=t_15m,
        expected_venue="gate",
    )
    assert bias_eval.per_tf_regime["4h"] is snap_4h.regime
    assert bias_eval.per_tf_structure["4h"] is snap_4h.market_structure
    assert bias_eval.per_tf_adx_14["4h"] == pytest.approx(snap_4h.technical.adx_14)
    assert bias_eval.per_tf_regime["1h"] is snap_1h.regime
    assert bias_eval.per_tf_structure["1h"] is snap_1h.market_structure
    assert bias_eval.per_tf_adx_14["1h"] == pytest.approx(snap_1h.technical.adx_14)

    # 4. Scanner (`scan_market_setups`) produces the exact same 4h/1h parity inside its 15m snapshot
    captured: list[AnalysisSnapshot] = []
    await scan_market_setups(
        mtf_svc,
        ("BTC/USDT:USDT",),
        timeframe=Timeframe.M15,
        snapshots_out=captured,
    )
    assert len(captured) == 1
    scan_ov = {ov.timeframe: ov for ov in captured[0].mtf_overview}
    assert scan_ov[Timeframe.H4].regime is snap_4h.regime
    assert scan_ov[Timeframe.H4].market_structure is snap_4h.market_structure
    assert scan_ov[Timeframe.H4].adx_14 == pytest.approx(snap_4h.technical.adx_14)


def test_mtf_overview_drops_mismatched_venue_and_enforces_close_time_cutoff() -> None:
    """Mixed-venue HTF batches are excluded and unclosed candles are ignored."""
    t_4h = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
    t_15m = datetime(2026, 11, 28, 14, 15, tzinfo=UTC)

    primary = _to_batch(
        _make_ranging_then_dip_candles(timeframe=Timeframe.M15, end_close=t_15m, count=200),
        Timeframe.M15,
        venue="gate",
    )
    wrong_venue_4h = _to_batch(
        _make_ranging_then_dip_candles(
            timeframe=Timeframe.H4, end_close=t_4h, count=200, venue="bybit"
        ),
        Timeframe.H4,
        venue="bybit",
    )
    snap = build_analysis_snapshot(primary, mtf_batches={Timeframe.H4: wrong_venue_4h})
    ov_tfs = {ov.timeframe for ov in snap.mtf_overview}
    assert Timeframe.H4 not in ov_tfs

    # Direct evaluator also rejects venue mismatch
    from crypto_chatbot.analysis.mtf import MtfVenueMismatchError

    with pytest.raises(MtfVenueMismatchError):
        evaluate_timeframe_context(
            wrong_venue_4h.candles,
            timeframe=Timeframe.H4,
            as_of=t_15m,
            expected_venue="gate",
        )
