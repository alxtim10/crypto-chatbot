"""Point-in-time causal feature & binary target builder for offline ML."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from crypto_chatbot.analysis.failed_breakout import detect_failed_breakouts
from crypto_chatbot.analysis.indicators import (
    MIN_HISTORY_EMA_50,
    adx_dmi_series,
    atr_percentile_series,
    atr_series,
    ema_series,
    macd_series,
    relative_volume_series,
    rsi_series,
    simple_return_series,
    validate_closed_candles,
)
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.structure import detect_confirmed_swings
from crypto_chatbot.news.pipeline import aggregate_sentiment
from crypto_chatbot.schemas.analysis import (
    MarketStructure,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.market import MarketCandle
from crypto_chatbot.schemas.ml import (
    FeatureFamily,
    FeatureRow,
    MlExperimentConfig,
)
from crypto_chatbot.schemas.news import SentimentObservation

FEATURE_FAMILIES: dict[FeatureFamily, tuple[str, ...]] = {
    FeatureFamily.TECHNICAL: (
        "ret_1",
        "ret_6",
        "ret_20",
        "close_vs_ema20",
        "ema20_vs_ema50",
        "rsi_14_norm",
        "macd_hist_pct",
        "atr_pct",
        "adx_14_norm",
        "di_diff_norm",
        "rvol_20",
    ),
    FeatureFamily.STRUCTURE_PIVOT: (
        "dist_to_sh_atr",
        "dist_to_sl_atr",
        "structure_bullish",
        "structure_bearish",
        "recent_fb_bull",
        "recent_fb_bear",
    ),
    FeatureFamily.SENTIMENT: (
        "sentiment_net_score",
        "sentiment_mean_conf",
        "sentiment_article_count_log",
    ),
    FeatureFamily.ORDERFLOW: (
        "taker_buy_ratio",
        "cvd_zscore_20",
    ),
}

ALL_FEATURE_NAMES: tuple[str, ...] = (
    *FEATURE_FAMILIES[FeatureFamily.TECHNICAL],
    *FEATURE_FAMILIES[FeatureFamily.STRUCTURE_PIVOT],
    *FEATURE_FAMILIES[FeatureFamily.SENTIMENT],
)


def _structure_at_time(
    confirmed_pivots: Sequence[StructureEvent],
) -> tuple[MarketStructure, StructureEvent | None, StructureEvent | None]:
    highs = [p for p in confirmed_pivots if p.event_type is StructureEventType.SWING_HIGH]
    lows = [p for p in confirmed_pivots if p.event_type is StructureEventType.SWING_LOW]
    latest_sh = highs[-1] if highs else None
    latest_sl = lows[-1] if lows else None

    if len(highs) < 2 or len(lows) < 2:
        return MarketStructure.MIXED, latest_sh, latest_sl

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    if h2.level > h1.level and l2.level > l1.level:
        return MarketStructure.BULLISH, latest_sh, latest_sl
    if h2.level < h1.level and l2.level < l1.level:
        return MarketStructure.BEARISH, latest_sh, latest_sl
    return MarketStructure.MIXED, latest_sh, latest_sl


def _compute_causal_feature_rows(
    candles: Sequence[MarketCandle],
    *,
    config: MlExperimentConfig,
    sentiment_observations: Sequence[SentimentObservation] | None = None,
    use_available_time: bool = False,
    include_target_horizon: bool = True,
) -> list[FeatureRow]:
    """Shared strictly causal feature computation core used by training & production."""
    validate_closed_candles(candles)
    if not candles:
        raise ValueError("candles cannot be empty")
    if candles[0].canonical_symbol != config.canonical_symbol:
        raise ValueError("candles canonical_symbol must match MlExperimentConfig.canonical_symbol")
    if candles[0].timeframe is not config.timeframe:
        raise ValueError("candles timeframe must match MlExperimentConfig.timeframe")

    tf_secs = int(config.timeframe.duration.total_seconds())
    horizon_bars = max(1, round((config.horizon_hours * 3600) / tf_secs))
    start_idx = max(MIN_HISTORY_EMA_50 - 1, 120)
    end_idx = (len(candles) - horizon_bars) if include_target_horizon else len(candles)
    if end_idx <= start_idx:
        raise ValueError("insufficient candles for warmup and feature computation")

    closes = np.array([c.close for c in candles], dtype=np.float64)
    highs = np.array([c.high for c in candles], dtype=np.float64)
    lows = np.array([c.low for c in candles], dtype=np.float64)
    vols = np.array([c.base_volume for c in candles], dtype=np.float64)

    ret_1_s = simple_return_series(closes, 1)
    ret_6_s = simple_return_series(closes, 6)
    ret_20_s = simple_return_series(closes, 20)
    ema_20_s = ema_series(closes, 20)
    ema_50_s = ema_series(closes, 50)
    rsi_s, _ = rsi_series(closes, 14)
    _, _, macd_hist_s = macd_series(closes, 12, 26, 9)
    atr_s, atr_pct_s = atr_series(highs, lows, closes, 14)
    adx_s, pdi_s, mdi_s = adx_dmi_series(highs, lows, closes, 14)
    rvol_s, _ = relative_volume_series(vols, 20)
    atr_pct_rank_s = atr_percentile_series(atr_pct_s, 100)

    all_pivots = detect_confirmed_swings(candles)
    all_fbs = detect_failed_breakouts(candles)
    threshold_rate = (config.cost_assumptions.round_trip_cost_bps + config.noise_buffer_bps) / (
        10_000.0
    )
    fb_lookback = 3 * config.timeframe.duration

    pivot_cursor = 0
    confirmed_pivots: list[StructureEvent] = []
    fb_cursor = 0
    confirmed_fbs: list[StructureEvent] = []

    rows: list[FeatureRow] = []
    for idx in range(len(candles)):
        candle = candles[idx]
        bar_time = candle.available_time if use_available_time else candle.close_time

        while pivot_cursor < len(all_pivots) and all_pivots[pivot_cursor].confirmation_time <= (
            bar_time
        ):
            confirmed_pivots.append(all_pivots[pivot_cursor])
            pivot_cursor += 1
        while fb_cursor < len(all_fbs) and all_fbs[fb_cursor].confirmation_time <= bar_time:
            confirmed_fbs.append(all_fbs[fb_cursor])
            fb_cursor += 1

        if idx < start_idx or idx >= end_idx:
            continue

        c_price = float(closes[idx])
        vals_to_check = (
            float(ret_1_s[idx]),
            float(ret_6_s[idx]),
            float(ret_20_s[idx]),
            float(ema_20_s[idx]),
            float(ema_50_s[idx]),
            float(rsi_s[idx]),
            float(macd_hist_s[idx]),
            float(atr_s[idx]),
            float(atr_pct_s[idx]),
            float(adx_s[idx]),
            float(pdi_s[idx]),
            float(mdi_s[idx]),
            float(atr_pct_rank_s[idx]),
        )
        if not all(math.isfinite(v) for v in vals_to_check) or c_price <= 0.0:
            continue

        ema20 = float(ema_20_s[idx])
        ema50 = float(ema_50_s[idx])
        atr_val = float(atr_s[idx])
        rvol_val = float(rvol_s[idx]) if math.isfinite(float(rvol_s[idx])) else 1.0

        tech = TechnicalMetrics(
            last_close=c_price,
            return_1=float(ret_1_s[idx]),
            return_20=float(ret_20_s[idx]),
            ema_20=ema20,
            ema_50=ema50,
            rsi_14=float(rsi_s[idx]),
            macd_histogram=float(macd_hist_s[idx]),
            atr_14=atr_val,
            atr_pct=float(atr_pct_s[idx]),
            adx_14=float(adx_s[idx]),
            plus_di_14=float(pdi_s[idx]),
            minus_di_14=float(mdi_s[idx]),
            atr_percentile_100=float(atr_pct_rank_s[idx]),
            relative_volume_20=rvol_val,
        )
        regime = classify_regime(tech)

        struct, latest_sh, latest_sl = _structure_at_time(confirmed_pivots)
        safe_atr = max(atr_val, c_price * 1e-4)
        dist_sh = (latest_sh.level - c_price) / safe_atr if latest_sh is not None else 0.0
        dist_sl = (c_price - latest_sl.level) / safe_atr if latest_sl is not None else 0.0
        recent_fb_bull = (
            1.0
            if any(
                ev.event_type is StructureEventType.FAILED_BREAKOUT_BULLISH
                and (bar_time - ev.confirmation_time) <= fb_lookback
                for ev in confirmed_fbs[-5:]
            )
            else 0.0
        )
        recent_fb_bear = (
            1.0
            if any(
                ev.event_type is StructureEventType.FAILED_BREAKOUT_BEARISH
                and (bar_time - ev.confirmation_time) <= fb_lookback
                for ev in confirmed_fbs[-5:]
            )
            else 0.0
        )

        sent_score = 0.0
        sent_conf = 0.0
        sent_count_log = 0.0
        row_avail: datetime = bar_time
        if sentiment_observations:
            sent = aggregate_sentiment(
                sentiment_observations,
                canonical_symbol=config.canonical_symbol,
                as_of=bar_time,
                window_hours=config.horizon_hours,
            )
            if sent is not None and sent.article_count > 0:
                sent_score = sent.net_sentiment_score
                sent_conf = sent.mean_confidence
                sent_count_log = math.log1p(float(sent.article_count))
                if sent.latest_available_time is not None:
                    row_avail = max(row_avail, sent.latest_available_time)

        feat_dict: dict[str, float] = {
            "ret_1": float(ret_1_s[idx]),
            "ret_6": float(ret_6_s[idx]),
            "ret_20": float(ret_20_s[idx]),
            "close_vs_ema20": (c_price - ema20) / ema20 if ema20 > 0.0 else 0.0,
            "ema20_vs_ema50": (ema20 - ema50) / ema50 if ema50 > 0.0 else 0.0,
            "rsi_14_norm": (float(rsi_s[idx]) - 50.0) / 50.0,
            "macd_hist_pct": float(macd_hist_s[idx]) / c_price,
            "atr_pct": float(atr_pct_s[idx]),
            "adx_14_norm": float(adx_s[idx]) / 100.0,
            "di_diff_norm": (float(pdi_s[idx]) - float(mdi_s[idx])) / 100.0,
            "rvol_20": rvol_val,
            "dist_to_sh_atr": float(max(-10.0, min(10.0, dist_sh))),
            "dist_to_sl_atr": float(max(-10.0, min(10.0, dist_sl))),
            "structure_bullish": 1.0 if struct is MarketStructure.BULLISH else 0.0,
            "structure_bearish": 1.0 if struct is MarketStructure.BEARISH else 0.0,
            "recent_fb_bull": recent_fb_bull,
            "recent_fb_bear": recent_fb_bear,
            "sentiment_net_score": sent_score,
            "sentiment_mean_conf": sent_conf,
            "sentiment_article_count_log": sent_count_log,
        }

        # Order flow features
        tbr = (
            candle.taker_buy_base_volume / candle.base_volume
            if (candle.taker_buy_base_volume is not None and candle.base_volume > 0.0)
            else 0.5
        )
        cvd_window_start = max(0, idx - 19)
        c_sub = candles[cvd_window_start : idx + 1]
        sum_delta = 0.0
        sum_vol = 0.0
        for sc in c_sub:
            if sc.taker_buy_base_volume is not None:
                sum_delta += (2.0 * sc.taker_buy_base_volume) - sc.base_volume
                sum_vol += sc.base_volume
        cvd_ratio = sum_delta / sum_vol if sum_vol > 0.0 else 0.0

        feat_dict["taker_buy_ratio"] = float(tbr)
        feat_dict["cvd_zscore_20"] = float(max(-3.0, min(3.0, cvd_ratio)))

        if include_target_horizon:
            target_candle = candles[idx + horizon_bars]
            fwd_ret = (target_candle.close / c_price) - 1.0
            label = 1 if fwd_ret > threshold_rate else 0
            label_end = target_candle.close_time
        else:
            fwd_ret = 0.0
            label = 0
            label_end = bar_time + config.timeframe.duration

        rows.append(
            FeatureRow(
                prediction_time=bar_time,
                available_time=row_avail,
                label_end_time=label_end,
                regime=regime,
                features=feat_dict,
                forward_return=fwd_ret,
                label=label,
            )
        )
    return rows


def build_point_in_time_dataset(
    candles: Sequence[MarketCandle],
    *,
    config: MlExperimentConfig,
    sentiment_observations: Sequence[SentimentObservation] | None = None,
    use_available_time: bool = False,
) -> tuple[tuple[FeatureRow, ...], str]:
    """Build strictly causal feature/target rows with deterministic dataset SHA-256."""
    rows = _compute_causal_feature_rows(
        candles,
        config=config,
        sentiment_observations=sentiment_observations,
        use_available_time=use_available_time,
        include_target_horizon=True,
    )
    if len(rows) < 40:
        raise ValueError("insufficient point-in-time feature rows produced")

    digest_items = [
        {
            "t": row.prediction_time.isoformat(),
            "le": row.label_end_time.isoformat(),
            "y": row.label,
            "f": {k: round(v, 10) for k, v in sorted(row.features.items())},
        }
        for row in rows
    ]
    encoded = json.dumps(digest_items, sort_keys=True, separators=(",", ":"), allow_nan=False)
    dataset_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return tuple(rows), dataset_sha256


def extract_latest_production_features(
    candles: Sequence[MarketCandle],
    *,
    config: MlExperimentConfig,
    sentiment_observations: Sequence[SentimentObservation] | None = None,
    use_available_time: bool = False,
) -> FeatureRow:
    """Extract the latest causal feature row using the exact training feature pipeline."""
    rows = _compute_causal_feature_rows(
        candles,
        config=config,
        sentiment_observations=sentiment_observations,
        use_available_time=use_available_time,
        include_target_horizon=False,
    )
    if not rows:
        raise ValueError("insufficient closed candles for production feature extraction")
    return rows[-1]
