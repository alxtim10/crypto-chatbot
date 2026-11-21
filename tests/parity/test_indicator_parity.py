from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from crypto_chatbot.analysis.indicators import (
    adx_dmi_series,
    atr_series,
    compute_technical_metrics,
    ema_series,
    macd_series,
    rsi_series,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.market import Timeframe

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"

ABS_TOL = 1e-8
REL_TOL = 1e-9


def _tv_ma(values: list[float], length: int, alpha: float) -> list[float]:
    """Independent scalar Pine Script v5 `ta.ema` / `ta.rma` reference."""
    out = [math.nan] * len(values)
    first = next((i for i, v in enumerate(values) if not math.isnan(v)), None)
    if first is None or len(values) - first < length:
        return out
    out[first + length - 1] = sum(values[first : first + length]) / float(length)
    for i in range(first + length, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _tv_ema(values: list[float], length: int) -> list[float]:
    return _tv_ma(values, length, 2.0 / float(length + 1))


def _tv_rma(values: list[float], length: int) -> list[float]:
    return _tv_ma(values, length, 1.0 / float(length))


def _tv_rsi(closes: list[float], length: int = 14) -> list[float]:
    gains = [math.nan] + [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [math.nan] + [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_g = _tv_rma(gains, length)
    avg_l = _tv_rma(losses, length)
    out = [math.nan] * len(closes)
    for i, (g, loss_v) in enumerate(zip(avg_g, avg_l, strict=True)):
        if math.isnan(g) or math.isnan(loss_v):
            continue
        if loss_v == 0.0:
            out[i] = 100.0
        elif g == 0.0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + g / loss_v)
    return out


def _tv_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    length: int = 14,
) -> list[float]:
    tr = [highs[0] - lows[0]] + [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    return _tv_rma(tr, length)


def _tv_dmi(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    length: int = 14,
) -> tuple[list[float], list[float], list[float]]:
    tr = [math.nan] + [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    pdm = [math.nan]
    mdm = [math.nan]
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        pdm.append(up if up > dn and up > 0.0 else 0.0)
        mdm.append(dn if dn > up and dn > 0.0 else 0.0)
    s_tr = _tv_rma(tr, length)
    s_p = _tv_rma(pdm, length)
    s_m = _tv_rma(mdm, length)
    pdi = [100.0 * p / t if not math.isnan(t) else math.nan for p, t in zip(s_p, s_tr, strict=True)]
    mdi = [100.0 * m / t if not math.isnan(t) else math.nan for m, t in zip(s_m, s_tr, strict=True)]
    dx = [
        100.0 * abs(p - m) / (p + m) if not math.isnan(p) else math.nan
        for p, m in zip(pdi, mdi, strict=True)
    ]
    return _tv_rma(dx, length), pdi, mdi


def _assert_series_parity(actual: np.ndarray, expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for idx, (a, e) in enumerate(zip(actual.tolist(), expected, strict=True)):
        if math.isnan(e):
            assert math.isnan(a), f"index {idx}: expected NaN, got {a}"
        else:
            assert math.isclose(a, e, rel_tol=REL_TOL, abs_tol=ABS_TOL), (
                f"index {idx}: actual={a}, expected={e}"
            )


def test_tradingview_parity_1000_candles() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    rows = fixture["raw_ohlcv"]
    assert len(rows) == 1000

    symbol = resolve_mvp_spot_symbol("BTC")
    received = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    batch = normalize_ohlcv_rows(
        rows,
        symbol=symbol,
        timeframe=Timeframe.H4,
        received_time=received,
        is_backfill=True,
    )
    assert len(batch.candles) == 1000

    highs = [c.high for c in batch.candles]
    lows = [c.low for c in batch.candles]
    closes = [c.close for c in batch.candles]

    _assert_series_parity(ema_series(closes, 20), _tv_ema(closes, 20))
    _assert_series_parity(ema_series(closes, 50), _tv_ema(closes, 50))
    _assert_series_parity(ema_series(closes, 200), _tv_ema(closes, 200))

    rsi_actual, _ = rsi_series(closes, 14)
    _assert_series_parity(rsi_actual, _tv_rsi(closes, 14))

    tv_macd = [
        a - b if not (math.isnan(a) or math.isnan(b)) else math.nan
        for a, b in zip(_tv_ema(closes, 12), _tv_ema(closes, 26), strict=True)
    ]
    tv_sig = _tv_ema(tv_macd, 9)
    tv_hist = [
        m - s if not (math.isnan(m) or math.isnan(s)) else math.nan
        for m, s in zip(tv_macd, tv_sig, strict=True)
    ]
    macd_a, sig_a, hist_a = macd_series(closes, 12, 26, 9)
    _assert_series_parity(macd_a, tv_macd)
    _assert_series_parity(sig_a, tv_sig)
    _assert_series_parity(hist_a, tv_hist)

    atr_a, _ = atr_series(highs, lows, closes, 14)
    _assert_series_parity(atr_a, _tv_atr(highs, lows, closes, 14))

    adx_a, pdi_a, mdi_a = adx_dmi_series(highs, lows, closes, 14)
    tv_adx, tv_pdi, tv_mdi = _tv_dmi(highs, lows, closes, 14)
    _assert_series_parity(adx_a, tv_adx)
    _assert_series_parity(pdi_a, tv_pdi)
    _assert_series_parity(mdi_a, tv_mdi)

    # Golden bar-999 regression guard
    metrics = compute_technical_metrics(batch.candles).metrics
    assert metrics.last_close == pytest.approx(65577.38, abs=1e-6)
    assert metrics.ema_20 == pytest.approx(63243.828834975982, abs=1e-6)
    assert metrics.ema_50 == pytest.approx(63161.267192488667, abs=1e-6)
    assert metrics.ema_200 == pytest.approx(62923.750471501262, abs=1e-6)
    assert metrics.rsi_14 == pytest.approx(65.699148476542, abs=1e-6)
    assert metrics.macd_line == pytest.approx(195.504860503075, abs=1e-6)
    assert metrics.macd_signal == pytest.approx(-87.222718332244, abs=1e-6)
    assert metrics.macd_histogram == pytest.approx(282.727578835319, abs=1e-6)
    assert metrics.atr_14 == pytest.approx(1320.461740889370, abs=1e-6)
    assert metrics.adx_14 == pytest.approx(19.129226079121, abs=1e-6)
    assert metrics.plus_di_14 == pytest.approx(26.488655793835, abs=1e-6)
    assert metrics.minus_di_14 == pytest.approx(10.942590881329, abs=1e-6)
