"""Deterministic analysis service: snapshot assembly, hashing, and recomputation."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from crypto_chatbot.analysis.cross_venue import compute_cross_venue_derivatives
from crypto_chatbot.analysis.derivatives import (
    DERIVATIVES_PARAMETERS,
    DerivativesAnalysisResult,
    compute_derivatives_metrics,
)
from crypto_chatbot.analysis.factors import build_factors
from crypto_chatbot.analysis.failed_breakout import detect_failed_breakouts
from crypto_chatbot.analysis.indicators import (
    CALC_VERSION,
    INDICATOR_PARAMETERS,
    compute_technical_metrics,
)
from crypto_chatbot.analysis.macro import build_macro_benchmark_context
from crypto_chatbot.analysis.mtf import (
    MTF_REGIME_LABELS as MTF_REGIME_LABELS,
)
from crypto_chatbot.analysis.mtf import (
    MtfVenueMismatchError,
    evaluate_mtf_overview,
    filter_causal_venue_candles,
)
from crypto_chatbot.analysis.orderflow import compute_closed_candle_orderflow
from crypto_chatbot.analysis.positioning import compute_positioning_panel
from crypto_chatbot.analysis.regimes import classify_regime, classify_volatility_regime
from crypto_chatbot.analysis.setup import SETUP_PARAMETERS, build_trade_setup
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.statistics import (
    STATISTICS_PARAMETERS,
    StatisticalMetrics,
    compute_statistical_metrics,
)
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    MIN_FULL_HISTORY_CANDLES,
    DerivativesLiquidityProvider,
    FuturesTickerBatch,
    FuturesTickerProvider,
    HistoryMarketDataProvider,
    MarketDataProvider,
    OhlcvBatch,
)
from crypto_chatbot.market_data.bybit_derivatives import BybitDerivativesProvider
from crypto_chatbot.market_data.fallback_provider import (
    FallbackDerivativesLiquidityProvider,
    FallbackMarketDataProvider,
)
from crypto_chatbot.market_data.gate_risk_tiers import GateRiskTiersClient
from crypto_chatbot.market_data.macro_provider import DxyMarketDataProvider
from crypto_chatbot.market_data.quality import (
    determine_quality_status,
    is_candle_stale,
)
from crypto_chatbot.market_data.resampling import resample_candles
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.market_data.tradingview import compare_packet_with_exchange_candles
from crypto_chatbot.ml.shadow import ShadowInferenceEngine
from crypto_chatbot.news.pipeline import aggregate_sentiment
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DerivativesMetrics,
    Fact,
    InputRange,
    PositioningPanel,
    SentimentPanelDiagnostics,
    SentimentSourceStatus,
    SetupStatus,
    SmcSummary,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
    TimeframeOverview,
    TradeSetup,
)
from crypto_chatbot.schemas.derivatives import (
    CrossVenueDerivativesSummary,
    DerivativesContext,
    OrderFlowMetrics,
)
from crypto_chatbot.schemas.macro import MacroBenchmarkContext
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityFlag,
    QualityStatus,
    Timeframe,
)
from crypto_chatbot.schemas.news import FearGreedObservation, SentimentMetrics, SentimentObservation
from crypto_chatbot.schemas.tradingview import TradingViewMetrics, TradingViewPacket
from crypto_chatbot.storage.repositories import (
    ContractStatsRepository,
    DerivativesRepository,
    FearGreedRepository,
    MarketRepository,
    ModelPredictionRepository,
    NewsRepository,
    SetupLogRepository,
    TradingViewRepository,
)

logger = logging.getLogger(__name__)

_BLOCKING_BATCH_FLAGS: frozenset[QualityFlag] = frozenset(
    {
        QualityFlag.PROVIDER_ERROR,
        QualityFlag.STALE,
        QualityFlag.INVALID_OHLC,
        QualityFlag.GAP,
        QualityFlag.SYMBOL_MISMATCH,
    }
)

_FLAG_LIMITATIONS: dict[QualityFlag, str] = {
    QualityFlag.STALE: "Data candle tertutup terbaru melewati batas kesegaran (stale).",
    QualityFlag.GAP: (
        "Terdapat celah interval (gap) pada riwayat candle; tidak dilakukan interpolasi."
    ),
    QualityFlag.INVALID_OHLC: "Ditemukan data OHLC tidak valid atau konflik duplikat candle.",
    QualityFlag.SYMBOL_MISMATCH: "Venue atau simbol penyedia tidak cocok dengan simbol target.",
    QualityFlag.SHORT_HISTORY: (
        "Riwayat candle di bawah 600 bar sehingga EMA 200 tidak tersedia (status DEGRADED)."
    ),
    QualityFlag.FLAT_PRICE: "Harga datar pada jendela RSI terbaru (pergerakan naik dan turun nol).",
    QualityFlag.ZERO_VOLUME: (
        "Rata-rata volume 20 bar sebelumnya bernilai nol; volume relatif tidak tersedia."
    ),
    QualityFlag.DUPLICATE: "Terdapat candle duplikat pada respons mentah yang telah dideduplikasi.",
    QualityFlag.PROVIDER_ERROR: "Gagal mengambil data dari penyedia market data.",
    QualityFlag.DERIVATIVES_UNAVAILABLE: (
        "Sebagian data derivatif perpetual tidak tersedia; nilai tidak direkayasa."
    ),
    QualityFlag.ORDERBOOK_UNAVAILABLE: (
        "Data order book tidak tersedia atau tidak valid; kedalaman tidak direkayasa."
    ),
    QualityFlag.SEQUENCE_GAP: (
        "Terdeteksi celah urutan (sequence gap) pada order book dan dilakukan resinkronisasi."
    ),
    QualityFlag.PRICE_MISMATCH: "Terdapat deviasi harga lintas sumber di atas ambang toleransi.",
    QualityFlag.SEVERE_PRICE_MISMATCH: (
        "Terdeteksi deviasi harga lintas sumber berat; kondisi terkini tidak disajikan."
    ),
    QualityFlag.UNIT_MISMATCH: (
        "Unit open interest tidak konsisten dengan notional dan harga referensi."
    ),
}


def compute_candles_sha256(candles: Sequence[MarketCandle]) -> str:
    """Compute a deterministic SHA-256 hash over canonical immutable candle fields."""
    canonical_rows = [
        [
            c.venue,
            c.canonical_symbol,
            c.market_type.value,
            c.timeframe.value,
            c.open_time.isoformat(),
            c.close_time.isoformat(),
            repr(c.open),
            repr(c.high),
            repr(c.low),
            repr(c.close),
            repr(c.base_volume),
            repr(c.quote_volume) if c.quote_volume is not None else None,
        ]
        for c in candles
    ]
    encoded = json.dumps(canonical_rows, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_derivatives_digest(metrics: DerivativesMetrics) -> str:
    encoded = json.dumps(
        metrics.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_tradingview_digest(metrics: TradingViewMetrics) -> str:
    encoded = json.dumps(
        metrics.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_sentiment_digest(metrics: SentimentMetrics) -> str:
    encoded = json.dumps(
        metrics.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_macro_digest(metrics: MacroBenchmarkContext) -> str:
    encoded = json.dumps(
        metrics.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_snapshot_id(
    *,
    provider: str,
    venue: str,
    canonical_symbol: str,
    market_type: str,
    timeframe: str,
    as_of: datetime,
    calc_version: str,
    input_sha256: str | None,
    mtf_hashes: Mapping[str, str],
    quality_status: QualityStatus,
    quality_flags: Sequence[QualityFlag],
    derivatives_digest: str | None = None,
    tradingview_digest: str | None = None,
    sentiment_digest: str | None = None,
    macro_digest: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "calc_version": calc_version,
        "canonical_symbol": canonical_symbol,
        "input_sha256": input_sha256,
        "market_type": market_type,
        "mtf_hashes": dict(sorted(mtf_hashes.items())),
        "provider": provider,
        "quality_flags": [flag.value for flag in quality_flags],
        "quality_status": quality_status.value,
        "timeframe": timeframe,
        "venue": venue,
    }
    if derivatives_digest is not None:
        payload["derivatives_digest"] = derivatives_digest
    if tradingview_digest is not None:
        payload["tradingview_digest"] = tradingview_digest
    if sentiment_digest is not None:
        payload["sentiment_digest"] = sentiment_digest
    if macro_digest is not None:
        payload["macro_digest"] = macro_digest
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    return f"snap-{digest[:16]}"


def _select_recent_events(
    swings: Sequence[StructureEvent],
    failed_breakouts: Sequence[StructureEvent],
) -> tuple[
    tuple[StructureEvent, ...],
    StructureEvent | None,
    StructureEvent | None,
    StructureEvent | None,
]:
    swing_highs = [e for e in swings if e.event_type is StructureEventType.SWING_HIGH][-2:]
    swing_lows = [e for e in swings if e.event_type is StructureEventType.SWING_LOW][-2:]
    recent_fbs = list(failed_breakouts[-2:])

    combined = sorted(
        [*swing_highs, *swing_lows, *recent_fbs],
        key=lambda ev: (ev.confirmation_time, ev.event_time, ev.event_type.value),
    )
    latest_sh = swing_highs[-1] if swing_highs else None
    latest_sl = swing_lows[-1] if swing_lows else None
    latest_fb = recent_fbs[-1] if recent_fbs else None
    return tuple(combined), latest_sh, latest_sl, latest_fb


def _build_facts(
    *,
    metrics: TechnicalMetrics,
    mtf_rsi_values: Mapping[Timeframe, float | None],
    events: Sequence[StructureEvent],
    latest_sh: StructureEvent | None,
    latest_sl: StructureEvent | None,
    latest_fb: StructureEvent | None,
    source: str,
    as_of: datetime,
    base_unit: str,
    quote_currency: str,
    timeframe_label: str,
    derivatives: DerivativesMetrics | None = None,
    smc: SmcSummary | None = None,
    trade_setup: TradeSetup | None = None,
    tradingview: TradingViewMetrics | None = None,
    sentiment: SentimentMetrics | None = None,
    order_flow: OrderFlowMetrics | None = None,
    cross_venue_derivatives: CrossVenueDerivativesSummary | None = None,
    statistics: StatisticalMetrics | None = None,
    volatility_percentile: float | None = None,
    macro_context: MacroBenchmarkContext | None = None,
) -> tuple[tuple[Fact, ...], dict[str, str]]:
    facts: list[Fact] = []
    fact_ids_by_key: dict[str, str] = {}

    def add_fact(
        key: str | None,
        label: str,
        value: float | None,
        unit: str,
        display_format: str,
        fact_as_of: datetime = as_of,
        fact_source: str = source,
    ) -> str | None:
        if value is None:
            return None
        fid = f"F{len(facts) + 1}"
        facts.append(
            Fact(
                fact_id=fid,
                label=label,
                value=value,
                unit=unit,
                display_format=display_format,
                source=fact_source,
                as_of=fact_as_of,
            )
        )
        if key is not None:
            fact_ids_by_key[key] = fid
        return fid

    add_fact("last_close", "Harga penutupan terakhir", metrics.last_close, quote_currency, "price")
    add_fact(
        "return_1",
        "Return 1 bar",
        metrics.return_1 * 100.0 if metrics.return_1 is not None else None,
        "%",
        "signed_percent",
    )
    add_fact(
        "return_20",
        "Return 20 bar",
        metrics.return_20 * 100.0 if metrics.return_20 is not None else None,
        "%",
        "signed_percent",
    )
    add_fact("ema_20", "EMA 20", metrics.ema_20, quote_currency, "price")
    add_fact("ema_50", "EMA 50", metrics.ema_50, quote_currency, "price")
    add_fact("ema_200", "EMA 200", metrics.ema_200, quote_currency, "price")
    add_fact("rsi_14", "RSI 14", metrics.rsi_14, "index", "decimal_2")
    add_fact("macd_line", "MACD line (12/26)", metrics.macd_line, quote_currency, "decimal_2")
    add_fact("macd_signal", "MACD signal (9)", metrics.macd_signal, quote_currency, "decimal_2")
    add_fact(
        "macd_histogram", "MACD histogram", metrics.macd_histogram, quote_currency, "decimal_2"
    )
    add_fact("atr_14", "ATR 14", metrics.atr_14, quote_currency, "price")
    add_fact(
        "atr_pct",
        "ATR% 14",
        metrics.atr_pct * 100.0 if metrics.atr_pct is not None else None,
        "%",
        "percent",
    )
    add_fact(
        "atr_percentile_100",
        "Persentil ATR (100 bar)",
        metrics.atr_percentile_100,
        "percentile",
        "decimal_1",
    )
    add_fact("adx_14", "ADX 14", metrics.adx_14, "index", "decimal_2")
    add_fact("plus_di_14", "+DI 14", metrics.plus_di_14, "index", "decimal_2")
    add_fact("minus_di_14", "-DI 14", metrics.minus_di_14, "index", "decimal_2")
    add_fact(
        "relative_volume_20",
        "Volume relatif (20 bar)",
        metrics.relative_volume_20,
        "ratio",
        "ratio_2",
    )
    add_fact(
        "bollinger_upper_20",
        "Bollinger upper (20)",
        metrics.bollinger_upper_20,
        quote_currency,
        "price",
    )
    add_fact(
        "bollinger_mid_20",
        "Bollinger basis (20)",
        metrics.bollinger_mid_20,
        quote_currency,
        "price",
    )
    add_fact(
        "bollinger_lower_20",
        "Bollinger lower (20)",
        metrics.bollinger_lower_20,
        quote_currency,
        "price",
    )
    add_fact(
        "bollinger_percent_b",
        "Bollinger %B",
        metrics.bollinger_percent_b,
        "ratio",
        "signed_decimal_2",
    )
    add_fact(
        "bollinger_bandwidth",
        "Bollinger bandwidth",
        metrics.bollinger_bandwidth * 100.0 if metrics.bollinger_bandwidth is not None else None,
        "%",
        "percent",
    )
    add_fact(
        "donchian_upper_20",
        "Donchian upper (20)",
        metrics.donchian_upper_20,
        quote_currency,
        "price",
    )
    add_fact(
        "donchian_lower_20",
        "Donchian lower (20)",
        metrics.donchian_lower_20,
        quote_currency,
        "price",
    )
    add_fact(
        "vwap_rolling_20", "VWAP rolling (20)", metrics.vwap_rolling_20, quote_currency, "price"
    )
    add_fact(
        "vwap_z_score",
        "Deviasi harga vs VWAP (z-score)",
        metrics.vwap_z_score,
        "sigma",
        "signed_decimal_2",
    )

    if statistics is not None:
        add_fact(
            "realized_vol_cc",
            "Volatilitas terealisasi (20 bar)",
            statistics.realized_vol_cc * 100.0 if statistics.realized_vol_cc is not None else None,
            "%",
            "percent",
        )
        add_fact(
            "parkinson_vol",
            "Volatilitas Parkinson (20 bar)",
            statistics.parkinson_vol * 100.0 if statistics.parkinson_vol is not None else None,
            "%",
            "percent",
        )
        add_fact(
            "garman_klass_vol",
            "Volatilitas Garman-Klass (20 bar)",
            statistics.garman_klass_vol * 100.0
            if statistics.garman_klass_vol is not None
            else None,
            "%",
            "percent",
        )
        add_fact(
            "hurst_exponent",
            "Eksponen Hurst (persistensi)",
            statistics.hurst_exponent,
            "index",
            "decimal_2",
        )
        add_fact(
            "ou_half_life",
            "Half-life mean reversion",
            statistics.ou_half_life,
            "bar",
            "decimal_1",
        )

    if volatility_percentile is not None:
        add_fact(
            "realized_vol_percentile",
            "Persentil volatilitas terealisasi",
            volatility_percentile,
            "percentile",
            "decimal_1",
        )

    for tf, key in (
        (Timeframe.H1, "mtf_rsi_1h"),
        (Timeframe.H4, "mtf_rsi_4h"),
        (Timeframe.D1, "mtf_rsi_1d"),
    ):
        add_fact(key, f"RSI 14 ({tf.value})", mtf_rsi_values.get(tf), "index", "decimal_2")

    add_fact(
        "prev_day_high", "High hari UTC sebelumnya", metrics.prev_day_high, quote_currency, "price"
    )
    add_fact(
        "prev_day_low", "Low hari UTC sebelumnya", metrics.prev_day_low, quote_currency, "price"
    )
    add_fact(
        "prev_day_close",
        "Close hari UTC sebelumnya",
        metrics.prev_day_close,
        quote_currency,
        "price",
    )
    add_fact(
        "prev_week_high",
        "High minggu ISO sebelumnya",
        metrics.prev_week_high,
        quote_currency,
        "price",
    )
    add_fact(
        "prev_week_low", "Low minggu ISO sebelumnya", metrics.prev_week_low, quote_currency, "price"
    )
    add_fact(
        "prev_week_close",
        "Close minggu ISO sebelumnya",
        metrics.prev_week_close,
        quote_currency,
        "price",
    )

    event_labels = {
        StructureEventType.SWING_HIGH: "Swing high terkonfirmasi",
        StructureEventType.SWING_LOW: "Swing low terkonfirmasi",
        StructureEventType.FAILED_BREAKOUT_BULLISH: "Level gagal breakout atas",
        StructureEventType.FAILED_BREAKOUT_BEARISH: "Level gagal breakdown bawah",
    }
    for event in events:
        event_key: str | None = None
        if event == latest_sh:
            event_key = "latest_swing_high"
        elif event == latest_sl:
            event_key = "latest_swing_low"
        elif event == latest_fb:
            event_key = "latest_failed_breakout"
        add_fact(
            event_key,
            event_labels[event.event_type],
            event.level,
            quote_currency,
            "price",
            fact_as_of=event.confirmation_time,
        )

    if derivatives is not None:
        if derivatives.funding_venue and derivatives.funding_symbol:
            f_source = f"{derivatives.funding_venue}:{derivatives.funding_symbol}"
            add_fact(
                "funding_rate_per_8h",
                "Funding rate ternormalisasi (8h)",
                derivatives.funding_rate_per_8h * 100.0
                if derivatives.funding_rate_per_8h is not None
                else None,
                "%",
                "funding_percent",
                fact_source=f_source,
            )
            add_fact(
                "funding_rate_annualized",
                "Funding rate tahunan",
                derivatives.funding_rate_annualized * 100.0
                if derivatives.funding_rate_annualized is not None
                else None,
                "%",
                "signed_percent",
                fact_source=f_source,
            )
            add_fact(
                "mark_index_basis_bps",
                "Basis mark vs index price",
                derivatives.mark_index_basis_bps,
                "bps",
                "signed_bps",
                fact_source=f_source,
            )

        if derivatives.open_interest_venue and derivatives.open_interest_symbol:
            oi_source = f"{derivatives.open_interest_venue}:{derivatives.open_interest_symbol}"
            add_fact(
                "open_interest_base",
                "Open interest (unit dasar)",
                derivatives.open_interest_base,
                derivatives.open_interest_base_unit or base_unit,
                "quantity",
                fact_source=oi_source,
            )
            add_fact(
                "open_interest_quote",
                "Open interest (notional)",
                derivatives.open_interest_quote,
                derivatives.open_interest_quote_currency or quote_currency,
                "notional_quote",
                fact_source=oi_source,
            )
            oi_win = derivatives.open_interest_change_window or timeframe_label
            add_fact(
                "open_interest_change_pct",
                f"Perubahan open interest ({oi_win})",
                derivatives.open_interest_change_pct * 100.0
                if derivatives.open_interest_change_pct is not None
                else None,
                "%",
                "signed_percent",
                fact_source=oi_source,
            )

        if derivatives.orderbook_venue and derivatives.orderbook_symbol:
            ob_mkt = (
                derivatives.orderbook_market_type.value
                if derivatives.orderbook_market_type is not None
                else MarketType.SPOT.value
            )
            ob_source = f"{derivatives.orderbook_venue}:{derivatives.orderbook_symbol}:{ob_mkt}"
            add_fact(
                "spread_bps",
                "Spread bid-ask order book",
                derivatives.spread_bps,
                "bps",
                "bps",
                fact_source=ob_source,
            )
            add_fact(
                "depth_10bps_bid_quote",
                "Kedalaman bid (10 bps)",
                derivatives.depth_10bps_bid_quote,
                quote_currency,
                "notional_quote",
                fact_source=ob_source,
            )
            add_fact(
                "depth_10bps_ask_quote",
                "Kedalaman ask (10 bps)",
                derivatives.depth_10bps_ask_quote,
                quote_currency,
                "notional_quote",
                fact_source=ob_source,
            )
            add_fact(
                "imbalance_10bps",
                "Imbalance order book (10 bps)",
                derivatives.imbalance_10bps,
                "ratio",
                "signed_decimal_2",
                fact_source=ob_source,
            )
            add_fact(
                "depth_50bps_bid_quote",
                "Kedalaman bid (50 bps)",
                derivatives.depth_50bps_bid_quote,
                quote_currency,
                "notional_quote",
                fact_source=ob_source,
            )
            add_fact(
                "depth_50bps_ask_quote",
                "Kedalaman ask (50 bps)",
                derivatives.depth_50bps_ask_quote,
                quote_currency,
                "notional_quote",
                fact_source=ob_source,
            )
            add_fact(
                "imbalance_50bps",
                "Imbalance order book (50 bps)",
                derivatives.imbalance_50bps,
                "ratio",
                "signed_decimal_2",
                fact_source=ob_source,
            )

        if derivatives.liquidation_source_available:
            liq_venue = derivatives.funding_venue or derivatives.open_interest_venue or "binance"
            liq_symbol = derivatives.funding_symbol or derivatives.open_interest_symbol or "perp"
            liq_source = f"{liq_venue}:{liq_symbol}"
            add_fact(
                "liquidation_long_quote",
                "Likuidasi posisi long",
                derivatives.liquidation_long_quote,
                quote_currency,
                "notional_quote",
                fact_source=liq_source,
            )
            add_fact(
                "liquidation_short_quote",
                "Likuidasi posisi short",
                derivatives.liquidation_short_quote,
                quote_currency,
                "notional_quote",
                fact_source=liq_source,
            )

        if derivatives.cross_source_deviation_bps is not None:
            add_fact(
                "cross_source_deviation_bps",
                "Deviasi harga lintas sumber",
                derivatives.cross_source_deviation_bps,
                "bps",
                "bps",
                fact_source=derivatives.cross_source_pair or source,
            )

    if smc is not None:
        add_fact(
            "smc_dealing_range_high",
            "High dealing range",
            smc.dealing_range_high,
            quote_currency,
            "price",
        )
        add_fact(
            "smc_dealing_range_low",
            "Low dealing range",
            smc.dealing_range_low,
            quote_currency,
            "price",
        )
        add_fact(
            "smc_equilibrium",
            "Equilibrium dealing range",
            smc.equilibrium,
            quote_currency,
            "price",
        )
        if smc.latest_structure_break is not None:
            brk_ev = smc.latest_structure_break
            add_fact(
                "smc_latest_break",
                "Level BOS/CHoCH terbaru",
                brk_ev.level,
                quote_currency,
                "price",
                fact_as_of=brk_ev.confirmation_time,
            )
        if smc.latest_sweep is not None:
            swp_ev = smc.latest_sweep
            add_fact(
                "smc_latest_sweep",
                "Level Liquidity Sweep terbaru",
                swp_ev.level,
                quote_currency,
                "price",
                fact_as_of=swp_ev.confirmation_time,
            )
        for z_key, z_label, zone in (
            ("smc_bullish_fvg", "Bullish FVG", smc.active_bullish_fvg),
            ("smc_bearish_fvg", "Bearish FVG", smc.active_bearish_fvg),
            ("smc_bullish_ob", "Bullish Order Block", smc.active_bullish_ob),
            ("smc_bearish_ob", "Bearish Order Block", smc.active_bearish_ob),
        ):
            if zone is not None:
                add_fact(
                    f"{z_key}_low",
                    f"{z_label} (bawah)",
                    zone.zone_low,
                    quote_currency,
                    "price",
                    fact_as_of=zone.confirmation_time,
                )
                add_fact(
                    f"{z_key}_high",
                    f"{z_label} (atas)",
                    zone.zone_high,
                    quote_currency,
                    "price",
                    fact_as_of=zone.confirmation_time,
                )

    if trade_setup is not None and trade_setup.status is SetupStatus.READY:
        add_fact(
            "setup_entry_low",
            "Zona entry (bawah)",
            trade_setup.entry_low,
            quote_currency,
            "price",
        )
        add_fact(
            "setup_entry_high",
            "Zona entry (atas)",
            trade_setup.entry_high,
            quote_currency,
            "price",
        )
        add_fact(
            "setup_stop_loss",
            "Stop loss (invalidasi)",
            trade_setup.stop_loss,
            quote_currency,
            "price",
        )
        add_fact(
            "setup_tp1",
            "Target TP 1",
            trade_setup.tp1,
            quote_currency,
            "price",
        )
        add_fact(
            "setup_tp2",
            "Target TP 2",
            trade_setup.tp2,
            quote_currency,
            "price",
        )
        add_fact(
            "setup_tp3",
            "Target TP 3",
            trade_setup.tp3,
            quote_currency,
            "price",
        )
        add_fact("setup_rr_tp1", "Rasio R:R TP 1", trade_setup.rr_tp1, "ratio", "rr_ratio")
        add_fact("setup_rr_tp2", "Rasio R:R TP 2", trade_setup.rr_tp2, "ratio", "rr_ratio")
        add_fact("setup_rr_tp3", "Rasio R:R TP 3", trade_setup.rr_tp3, "ratio", "rr_ratio")
        add_fact(
            "setup_stop_loss_distance_pct",
            "Jarak stop loss ke invalidasi",
            trade_setup.stop_loss_distance_pct,
            "%",
            "percent",
        )
        add_fact(
            "setup_max_theoretical_leverage",
            "Leverage referensi teoretis",
            float(trade_setup.max_theoretical_leverage)
            if trade_setup.max_theoretical_leverage is not None
            else None,
            "x",
            "integer",
        )

    if trade_setup is not None:
        m15_src = f"{source.rsplit(':', 1)[0]}:15m"
        add_fact(
            "setup_ltf_support",
            "Support struktur 15m",
            trade_setup.ltf_support,
            quote_currency,
            "price",
            fact_source=m15_src,
        )
        add_fact(
            "setup_ltf_resistance",
            "Resistance struktur 15m",
            trade_setup.ltf_resistance,
            quote_currency,
            "price",
            fact_source=m15_src,
        )

    if tradingview is not None:
        tv_src = (
            f"tradingview:{tradingview.venue}:{tradingview.provider_symbol}:"
            f"{tradingview.timeframe.value}"
        )
        add_fact(
            "tv_close",
            "Harga close TradingView",
            tradingview.tv_close,
            quote_currency,
            "price",
            fact_as_of=tradingview.candle_close_time,
            fact_source=tv_src,
        )
        add_fact(
            "tv_exchange_deviation_bps",
            "Deviasi harga TradingView vs bursa",
            tradingview.max_deviation_bps,
            "bps",
            "bps",
            fact_as_of=tradingview.candle_close_time,
            fact_source=tv_src,
        )
        add_fact(
            "tv_rule_score",
            "Skor aturan TradingView (bukan probabilitas)",
            tradingview.rule_score,
            "score",
            "signed_decimal_2",
            fact_as_of=tradingview.candle_close_time,
            fact_source=tv_src,
        )

    if sentiment is not None and sentiment.article_count > 0:
        sent_src = f"news:{sentiment.model_version}:{sentiment.canonical_symbol}"
        sent_time = sentiment.latest_available_time or as_of
        add_fact(
            "sentiment_net_score",
            "Skor sentimen berita 24h (bukan probabilitas)",
            sentiment.net_sentiment_score,
            "score",
            "signed_decimal_2",
            fact_as_of=sent_time,
            fact_source=sent_src,
        )
        add_fact(
            "sentiment_confidence",
            "Confidence sentimen berita 24h",
            sentiment.mean_confidence * 100.0,
            "%",
            "percent",
            fact_as_of=sent_time,
            fact_source=sent_src,
        )
        add_fact(
            "sentiment_article_count",
            "Artikel berita unik (24h)",
            float(sentiment.article_count),
            "artikel",
            "integer",
            fact_as_of=sent_time,
            fact_source=sent_src,
        )

    if order_flow is not None and order_flow.has_order_flow:
        add_fact(
            "order_flow_taker_buy_ratio",
            "Taker buy ratio (bar tertutup)",
            order_flow.taker_buy_ratio * 100.0,
            "%",
            "percent",
        )
        add_fact(
            "order_flow_cvd_20",
            "Cumulative Volume Delta (20 bar)",
            order_flow.cvd_20,
            base_unit,
            "quantity",
        )

    if cross_venue_derivatives is not None:
        cv_src = f"cross_venue:{','.join(cross_venue_derivatives.venues)}"
        if cross_venue_derivatives.funding_dispersion_bps is not None:
            add_fact(
                "funding_dispersion_bps",
                "Dispersi funding rate lintas venue",
                cross_venue_derivatives.funding_dispersion_bps,
                "bps",
                "bps",
                fact_source=cv_src,
            )
        if cross_venue_derivatives.mark_price_spread_bps is not None:
            add_fact(
                "mark_price_spread_bps",
                "Spread mark price lintas venue",
                cross_venue_derivatives.mark_price_spread_bps,
                "bps",
                "bps",
                fact_source=cv_src,
            )

    if macro_context is not None:
        if macro_context.dxy is not None:
            dxy_src = f"yahoo:nyb:{macro_context.dxy.symbol}"
            add_fact(
                "dxy_last_close",
                "Harga penutupan DXY",
                macro_context.dxy.last_close,
                "USD",
                "price",
                fact_as_of=macro_context.dxy.as_of,
                fact_source=dxy_src,
            )
            add_fact(
                "dxy_rolling_corr_20",
                "Korelasi rolling DXY (20 bar)",
                macro_context.dxy.rolling_corr_20,
                "r",
                "signed_decimal_2",
                fact_as_of=macro_context.dxy.as_of,
                fact_source=dxy_src,
            )
        if (
            macro_context.btc_benchmark is not None
            and macro_context.btc_benchmark.leader_alignment != "self"
        ):
            btc_src = f"binance:{macro_context.btc_benchmark.canonical_symbol}"
            add_fact(
                "alt_relative_strength_vs_btc_20",
                "Relative strength vs BTC (20 bar)",
                macro_context.btc_benchmark.relative_strength_20_pct,
                "%",
                "percent",
                fact_as_of=macro_context.btc_benchmark.as_of,
                fact_source=btc_src,
            )
            add_fact(
                "alt_beta_vs_btc_20",
                "Rolling Beta vs BTC (20 bar)",
                macro_context.btc_benchmark.rolling_beta_20,
                "beta",
                "decimal_2",
                fact_as_of=macro_context.btc_benchmark.as_of,
                fact_source=btc_src,
            )

    return tuple(facts), fact_ids_by_key


def _build_mtf_overview(
    batch: OhlcvBatch,
    *,
    mtf_batches: Mapping[Timeframe, OhlcvBatch] | None,
    as_of: datetime,
    use_available_time: bool,
) -> tuple[TimeframeOverview, ...]:
    """Delegate to the single shared MTF context evaluator (`analysis/mtf.py`)."""
    return evaluate_mtf_overview(
        batch,
        mtf_batches=mtf_batches,
        as_of=as_of,
        use_available_time=use_available_time,
    )


def _build_limitations(
    *,
    batch: OhlcvBatch,
    merged_flags: Sequence[QualityFlag],
    status: QualityStatus,
    dropped_mtf: Sequence[Timeframe],
    extra_notes: Sequence[str] = (),
) -> tuple[str, ...]:
    notes: list[str] = []
    if batch.error_message:
        notes.append(batch.error_message)
    notes.extend(extra_notes)
    for flag in merged_flags:
        if flag in _FLAG_LIMITATIONS:
            notes.append(_FLAG_LIMITATIONS[flag])
    if dropped_mtf:
        tfs = ", ".join(tf.value for tf in dropped_mtf)
        notes.append(
            "Penyelarasan RSI multi-timeframe tidak lengkap karena data "
            f"{tfs} tidak memenuhi syarat."
        )
    if status is QualityStatus.INSUFFICIENT_DATA and not batch.has_provider_error:
        notes.append(
            "Analisis kondisi saat ini ditahan (INSUFFICIENT_DATA); data tidak lengkap atau basi "
            "tidak disajikan sebagai kondisi terkini."
        )
    notes.append(
        f"Sumber data terbatas pada {batch.metadata.market_type.value} {batch.metadata.venue} "
        f"untuk {batch.canonical_symbol}."
    )
    notes.append(
        "Output indikator dan daftar faktor adalah pengukuran deskriptif, bukan probabilitas "
        "keberhasilan maupun saran transaksi."
    )
    return tuple(dict.fromkeys(notes))


def _insufficient_snapshot(
    *,
    batch: OhlcvBatch,
    input_range: InputRange | None,
    as_of: datetime,
    flags: tuple[QualityFlag, ...],
    dropped_mtf: Sequence[Timeframe] = (),
    extra_notes: Sequence[str] = (),
    extra_params: Mapping[str, int | float | str] | None = None,
    derivatives_digest: str | None = None,
    tradingview_digest: str | None = None,
    sentiment_digest: str | None = None,
) -> AnalysisSnapshot:
    snap_id = _compute_snapshot_id(
        provider=batch.metadata.provider,
        venue=batch.metadata.venue,
        canonical_symbol=batch.canonical_symbol,
        market_type=batch.metadata.market_type.value,
        timeframe=batch.timeframe.value,
        as_of=as_of,
        calc_version=CALC_VERSION,
        input_sha256=input_range.sha256 if input_range else None,
        mtf_hashes={},
        quality_status=QualityStatus.INSUFFICIENT_DATA,
        quality_flags=flags,
        derivatives_digest=derivatives_digest,
        tradingview_digest=tradingview_digest,
        sentiment_digest=sentiment_digest,
    )
    params = dict(INDICATOR_PARAMETERS)
    if extra_params:
        params.update(extra_params)
    return AnalysisSnapshot(
        snapshot_id=snap_id,
        provider=batch.metadata.provider,
        venue=batch.metadata.venue,
        canonical_symbol=batch.canonical_symbol,
        provider_symbol=batch.provider_symbol,
        market_type=batch.metadata.market_type,
        timeframe=batch.timeframe,
        as_of=as_of,
        input_range=input_range,
        calc_version=CALC_VERSION,
        indicator_parameters=params,
        technical=None,
        derivatives=None,
        facts=(),
        regime=None,
        market_structure=None,
        events=(),
        factors=(),
        quality_status=QualityStatus.INSUFFICIENT_DATA,
        quality_flags=flags,
        limitations=_build_limitations(
            batch=batch,
            merged_flags=flags,
            status=QualityStatus.INSUFFICIENT_DATA,
            dropped_mtf=dropped_mtf,
            extra_notes=extra_notes,
        ),
    )


def build_analysis_snapshot(
    batch: OhlcvBatch,
    *,
    mtf_batches: Mapping[Timeframe, OhlcvBatch] | None = None,
    derivatives_context: DerivativesContext | None = None,
    tradingview_packet: TradingViewPacket | None = None,
    news_observations: Sequence[SentimentObservation] | None = None,
    cross_venue_derivatives: CrossVenueDerivativesSummary | None = None,
    order_flow: OrderFlowMetrics | None = None,
    use_available_time: bool = False,
    dxy_candles: Sequence[MarketCandle] | None = None,
    btc_candles: Sequence[MarketCandle] | None = None,
    macro_context: MacroBenchmarkContext | None = None,
    positioning: PositioningPanel | None = None,
    fear_greed: FearGreedObservation | None = None,
    sentiment_diagnostics: SentimentPanelDiagnostics | None = None,
    maintenance_margin_rate: float | None = None,
    venue_max_leverage: float | None = None,
) -> AnalysisSnapshot:
    """Assemble a complete deterministic `AnalysisSnapshot` from spot and optional inputs."""
    if batch.has_provider_error or not batch.candles:
        flags = tuple(dict.fromkeys(batch.quality_flags))
        return _insufficient_snapshot(
            batch=batch,
            input_range=None,
            as_of=batch.received_time,
            flags=flags,
        )

    as_of = batch.candles[-1].close_time
    input_sha256 = compute_candles_sha256(batch.candles)
    input_range = InputRange(
        first_open_time=batch.candles[0].open_time,
        last_open_time=batch.candles[-1].open_time,
        candle_count=len(batch.candles),
        sha256=input_sha256,
    )

    pre_flags = tuple(dict.fromkeys(batch.quality_flags))
    pre_status = determine_quality_status(len(batch.candles), pre_flags)
    if pre_status is QualityStatus.INSUFFICIENT_DATA:
        return _insufficient_snapshot(
            batch=batch,
            input_range=input_range,
            as_of=as_of,
            flags=pre_flags,
        )

    mtf_candles: dict[Timeframe, tuple[MarketCandle, ...]] | None = None
    mtf_hashes: dict[str, str] = {}
    dropped_mtf: list[Timeframe] = []
    ltf_15m_candles: tuple[MarketCandle, ...] | None = None
    micro_5m_candles: tuple[MarketCandle, ...] | None = None
    htf_candles_by_tf: dict[Timeframe, tuple[MarketCandle, ...]] = {}

    if mtf_batches is not None:
        mtf_candles = (
            {batch.timeframe: batch.candles}
            if batch.timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.D1)
            else {}
        )
        expected_venue = batch.metadata.venue.strip().lower() or None
        for tf, mtf_batch in mtf_batches.items():
            if tf is batch.timeframe:
                continue
            if not mtf_batch.candles or (set(mtf_batch.quality_flags) & _BLOCKING_BATCH_FLAGS):
                if tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
                    dropped_mtf.append(tf)
                continue
            try:
                causal_mtf = filter_causal_venue_candles(
                    mtf_batch.candles,
                    as_of=as_of,
                    expected_venue=expected_venue,
                    use_available_time=use_available_time,
                )
            except MtfVenueMismatchError as exc:
                logger.warning("Dropping MTF %s batch: %s", tf.value, exc)
                if tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
                    dropped_mtf.append(tf)
                continue
            if not causal_mtf:
                if tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
                    dropped_mtf.append(tf)
                continue
            mtf_hashes[tf.value] = compute_candles_sha256(causal_mtf)
            if tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
                mtf_candles[tf] = causal_mtf
                if batch.timeframe is Timeframe.M15:
                    htf_candles_by_tf[tf] = causal_mtf
            elif tf is Timeframe.M15 and batch.timeframe is not Timeframe.M15:
                ltf_15m_candles = causal_mtf
            elif tf is Timeframe.M5 and batch.timeframe is not Timeframe.M5:
                micro_5m_candles = causal_mtf

    ind_result = compute_technical_metrics(
        batch.candles,
        mtf_candles=mtf_candles,
        use_available_time=use_available_time,
    )
    regime = classify_regime(ind_result.metrics)
    structure = analyze_market_structure(batch.candles)
    failed_breakouts = detect_failed_breakouts(batch.candles)
    statistical_metrics = compute_statistical_metrics(batch.candles)
    volatility_regime_result = classify_volatility_regime(batch.candles)

    deriv_result: DerivativesAnalysisResult | None = None
    extra_params: dict[str, int | float | str] = {}
    deriv_digest: str | None = None
    if derivatives_context is not None:
        deriv_result = compute_derivatives_metrics(
            derivatives_context,
            spot_symbol=batch.canonical_symbol,
            timeframe=batch.timeframe,
            as_of=as_of,
            use_available_time=use_available_time,
        )
        extra_params = dict(DERIVATIVES_PARAMETERS)
        if deriv_result.metrics is not None:
            deriv_digest = _compute_derivatives_digest(deriv_result.metrics)

    tv_metrics: TradingViewMetrics | None = None
    tv_digest: str | None = None
    tv_flags: tuple[QualityFlag, ...] = ()
    tv_notes: list[str] = []
    if tradingview_packet is not None:
        tv_avail = (
            tradingview_packet.available_time
            if use_available_time
            else tradingview_packet.candle_close_time
        )
        if tradingview_packet.candle_close_time > as_of or tv_avail > as_of:
            tv_notes.append(
                "Paket TradingView ai-agent-lens/v1 diabaikan karena melampaui as_of "
                "(anti-lookahead)."
            )
        elif (
            tradingview_packet.canonical_symbol != batch.canonical_symbol
            or tradingview_packet.market_type is not batch.metadata.market_type
        ):
            tv_flags = (QualityFlag.SYMBOL_MISMATCH,)
            tv_notes.append(
                "Simbol atau tipe pasar paket TradingView tidak cocok dengan batch bursa."
            )
        else:
            tv_cmp = compare_packet_with_exchange_candles(
                tradingview_packet,
                batch.candles,
                as_of=as_of,
                use_available_time=use_available_time,
            )
            tv_flags = tv_cmp.quality_flags
            tv_notes.extend(tv_cmp.limitations)
            tv_metrics = TradingViewMetrics(
                schema_version=tradingview_packet.schema_version,
                attribution_commit=tradingview_packet.attribution_commit,
                idempotency_key=tradingview_packet.idempotency_key,
                venue=tradingview_packet.venue,
                canonical_symbol=tradingview_packet.canonical_symbol,
                provider_symbol=tradingview_packet.provider_symbol,
                market_type=tradingview_packet.market_type,
                timeframe=tradingview_packet.timeframe,
                event_type=tradingview_packet.event_type.value,
                candle_close_time=tradingview_packet.candle_close_time,
                tv_close=tradingview_packet.close,
                exchange_close=tv_cmp.exchange_close,
                max_deviation_bps=tv_cmp.max_deviation_bps,
                rule_score=tradingview_packet.rule_score,
                rule_label=tradingview_packet.rule_label,
                is_probabilistic=False,
            )
            tv_digest = _compute_tradingview_digest(tv_metrics)

    sent_metrics: SentimentMetrics | None = None
    sent_digest: str | None = None
    if news_observations:
        sent_metrics = aggregate_sentiment(
            news_observations,
            canonical_symbol=batch.canonical_symbol,
            as_of=as_of,
        )
        if sent_metrics is not None:
            sent_digest = _compute_sentiment_digest(sent_metrics)
            extra_params["sentiment_window_hours"] = sent_metrics.window_hours

    combined_notes = (
        *(deriv_result.limitations if deriv_result is not None else ()),
        *tv_notes,
    )
    merged_flags = tuple(
        dict.fromkeys(
            [
                *batch.quality_flags,
                *ind_result.quality_flags,
                *(deriv_result.quality_flags if deriv_result is not None else ()),
                *tv_flags,
            ]
        )
    )
    final_status = determine_quality_status(len(batch.candles), merged_flags)
    if regime is None or final_status is QualityStatus.INSUFFICIENT_DATA:
        return _insufficient_snapshot(
            batch=batch,
            input_range=input_range,
            as_of=as_of,
            flags=merged_flags,
            dropped_mtf=dropped_mtf,
            extra_notes=combined_notes,
            extra_params=extra_params,
            derivatives_digest=deriv_digest,
            tradingview_digest=tv_digest,
            sentiment_digest=sent_digest,
        )

    recent_events, latest_sh, latest_sl, latest_fb = _select_recent_events(
        structure.pivots,
        failed_breakouts,
    )
    smc = analyze_smc(batch.candles, structure=structure)
    actual_order_flow = order_flow or compute_closed_candle_orderflow(batch.candles)
    if macro_context is None:
        macro_context = build_macro_benchmark_context(
            target_candles=batch.candles,
            target_canonical_symbol=batch.canonical_symbol,
            as_of=as_of,
            dxy_candles=dxy_candles,
            btc_candles=btc_candles,
            derivatives=deriv_result.metrics if deriv_result is not None else None,
            order_flow=actual_order_flow,
        )
    macro_digest: str | None = None
    if macro_context is not None:
        macro_digest = _compute_macro_digest(macro_context)

    trade_setup = build_trade_setup(
        metrics=ind_result.metrics,
        regime=regime,
        structure=structure,
        smc=smc,
        derivatives=deriv_result.metrics if deriv_result is not None else None,
        sentiment=sent_metrics,
        primary_timeframe=batch.timeframe,
        ltf_candles=ltf_15m_candles,
        micro_candles=micro_5m_candles,
        htf_candles_by_tf=htf_candles_by_tf or None,
        as_of=as_of,
        use_available_time=use_available_time,
        macro_context=macro_context,
        maintenance_margin_rate=maintenance_margin_rate,
        venue_max_leverage=venue_max_leverage,
    )
    source = f"{batch.metadata.venue}:{batch.canonical_symbol}:{batch.timeframe.value}"
    facts, fact_ids_by_key = _build_facts(
        metrics=ind_result.metrics,
        mtf_rsi_values=ind_result.mtf_rsi_values,
        events=recent_events,
        latest_sh=latest_sh,
        latest_sl=latest_sl,
        latest_fb=latest_fb,
        source=source,
        as_of=as_of,
        base_unit=batch.metadata.base_unit,
        quote_currency=batch.metadata.quote_currency,
        timeframe_label=batch.timeframe.value,
        derivatives=deriv_result.metrics if deriv_result is not None else None,
        smc=smc,
        trade_setup=trade_setup,
        tradingview=tv_metrics,
        sentiment=sent_metrics,
        order_flow=actual_order_flow,
        cross_venue_derivatives=cross_venue_derivatives,
        statistics=statistical_metrics,
        volatility_percentile=volatility_regime_result.percentile,
        macro_context=macro_context,
    )

    recent_cutoff = as_of - 10 * batch.timeframe.duration
    factors = build_factors(
        metrics=ind_result.metrics,
        regime=regime,
        structure=structure,
        failed_breakouts=failed_breakouts,
        recent_cutoff_time=recent_cutoff,
        fact_ids_by_key=fact_ids_by_key,
        derivatives=deriv_result.metrics if deriv_result is not None else None,
        smc=smc,
        tradingview=tv_metrics,
        sentiment=sent_metrics,
    )

    snap_id = _compute_snapshot_id(
        provider=batch.metadata.provider,
        venue=batch.metadata.venue,
        canonical_symbol=batch.canonical_symbol,
        market_type=batch.metadata.market_type.value,
        timeframe=batch.timeframe.value,
        as_of=as_of,
        calc_version=CALC_VERSION,
        input_sha256=input_sha256,
        mtf_hashes=mtf_hashes,
        quality_status=final_status,
        quality_flags=merged_flags,
        derivatives_digest=deriv_digest,
        tradingview_digest=tv_digest,
        sentiment_digest=sent_digest,
        macro_digest=macro_digest,
    )
    params = dict(INDICATOR_PARAMETERS)
    params.update(STATISTICS_PARAMETERS)
    params.update(SETUP_PARAMETERS)
    params.update(extra_params)

    mtf_overview = _build_mtf_overview(
        batch,
        mtf_batches=mtf_batches,
        as_of=as_of,
        use_available_time=use_available_time,
    )
    resolved_sent_diag = sentiment_diagnostics
    if resolved_sent_diag is None:
        if positioning is None:
            pos_status = SentimentSourceStatus(available=False, reason="not_wired")
        elif positioning.status == "UNAVAILABLE":
            pos_status = SentimentSourceStatus(
                available=False,
                reason=positioning.reason_code or "no_data",
                detail=positioning.note,
            )
        elif positioning.status == "WARMING_UP":
            pos_status = SentimentSourceStatus(
                available=True,
                reason="insufficient_history",
                detail=(
                    f"{positioning.observation_count} bars / {positioning.history_days:.1f}d < 7d"
                ),
            )
        else:
            pos_status = SentimentSourceStatus(available=True)

        fg_status = (
            SentimentSourceStatus(available=True)
            if fear_greed is not None
            else SentimentSourceStatus(available=False, reason="not_wired")
        )

        if sent_metrics is not None:
            news_status = SentimentSourceStatus(available=True)
        elif news_observations is None:
            news_status = SentimentSourceStatus(available=False, reason="not_wired")
        elif any(obs.available_time.astimezone(UTC) > as_of for obs in news_observations):
            news_status = SentimentSourceStatus(
                available=False,
                reason="not_yet_available",
                detail="observasi berita tersedia setelah as_of",
            )
        else:
            news_status = SentimentSourceStatus(available=False, reason="no_data")

        resolved_sent_diag = SentimentPanelDiagnostics(
            positioning=pos_status,
            fear_greed=fg_status,
            news=news_status,
        )

    return AnalysisSnapshot(
        snapshot_id=snap_id,
        provider=batch.metadata.provider,
        venue=batch.metadata.venue,
        canonical_symbol=batch.canonical_symbol,
        provider_symbol=batch.provider_symbol,
        market_type=batch.metadata.market_type,
        timeframe=batch.timeframe,
        as_of=as_of,
        input_range=input_range,
        calc_version=CALC_VERSION,
        indicator_parameters=params,
        technical=ind_result.metrics,
        derivatives=deriv_result.metrics if deriv_result is not None else None,
        smc=smc,
        mtf_overview=mtf_overview,
        tradingview=tv_metrics,
        sentiment=sent_metrics,
        trade_setup=trade_setup,
        order_flow=actual_order_flow,
        cross_venue_derivatives=cross_venue_derivatives,
        macro_context=macro_context,
        positioning=positioning,
        fear_greed=fear_greed,
        sentiment_diagnostics=resolved_sent_diag,
        facts=facts,
        regime=regime,
        market_structure=structure.market_structure,
        events=recent_events,
        factors=factors,
        quality_status=final_status,
        quality_flags=merged_flags,
        limitations=_build_limitations(
            batch=batch,
            merged_flags=merged_flags,
            status=final_status,
            dropped_mtf=dropped_mtf,
            extra_notes=combined_notes,
        ),
    )


def _batch_from_stored_candles(
    candles: Sequence[MarketCandle],
    *,
    as_of: datetime,
    is_primary: bool,
) -> OhlcvBatch:
    first = candles[0]
    flags: list[QualityFlag] = []
    gaps: list[tuple[datetime, datetime]] = []

    for prev, curr in itertools.pairwise(candles):
        expected_open = prev.open_time + first.timeframe.duration
        if curr.open_time > expected_open:
            gaps.append((expected_open, curr.open_time))
            flags.append(QualityFlag.GAP)
        elif curr.open_time < expected_open:
            flags.append(QualityFlag.INVALID_OHLC)

    if is_primary and len(candles) < MIN_FULL_HISTORY_CANDLES:
        flags.append(QualityFlag.SHORT_HISTORY)

    if is_candle_stale(candles[-1].close_time, first.timeframe, as_of):
        flags.append(QualityFlag.STALE)

    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider=first.provider,
            venue=first.venue,
            market_type=first.market_type,
            base_unit=first.base_unit,
            quote_currency=first.quote_currency,
        ),
        canonical_symbol=first.canonical_symbol,
        provider_symbol=first.provider_symbol,
        timeframe=first.timeframe,
        received_time=as_of,
        candles=tuple(candles),
        quality_flags=tuple(dict.fromkeys(flags)),
        gap_intervals=tuple(gaps),
    )


class AnalysisService:
    """Orchestrates spot & derivatives fetching, deterministic snapshot creation, and storage."""

    def __init__(
        self,
        provider: MarketDataProvider,
        repository: MarketRepository | None = None,
        *,
        derivatives_provider: DerivativesLiquidityProvider | None = None,
        derivatives_repository: DerivativesRepository | None = None,
        setup_repository: SetupLogRepository | None = None,
        tradingview_repository: TradingViewRepository | None = None,
        news_repository: NewsRepository | None = None,
        bybit_provider: BybitDerivativesProvider | None = None,
        shadow_engine: ShadowInferenceEngine | None = None,
        prediction_repository: ModelPredictionRepository | None = None,
        dxy_provider: DxyMarketDataProvider | None = None,
        contract_stats_repository: ContractStatsRepository | None = None,
        fear_greed_repository: FearGreedRepository | None = None,
        risk_tiers_client: GateRiskTiersClient | None = None,
        default_position_notional_usdt: float = 1_000.0,
        resample_from: Timeframe | None = None,
        cache_ttl_seconds: float = 0.0,
        cache_max_entries: int = 256,
    ) -> None:
        self._provider = provider
        self._repository = repository
        self._derivatives_provider = derivatives_provider
        self._bybit_provider = bybit_provider
        self._derivatives_repository = derivatives_repository
        self._setup_repository = setup_repository
        self._tradingview_repository = tradingview_repository
        self._news_repository = news_repository
        self._shadow_engine = shadow_engine
        self._prediction_repository = prediction_repository
        self._dxy_provider = dxy_provider
        self._contract_stats_repository = contract_stats_repository
        self._fear_greed_repository = fear_greed_repository
        self._risk_tiers_client = risk_tiers_client
        self._default_position_notional_usdt = max(0.0, float(default_position_notional_usdt))
        self._resample_from = resample_from
        self._cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._cache_max_entries = max(1, int(cache_max_entries))
        self._ohlcv_cache: OrderedDict[tuple[str, Timeframe, int], tuple[float, OhlcvBatch]] = (
            OrderedDict()
        )

    async def _fetch_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int,
    ) -> OhlcvBatch:
        key = (symbol, timeframe, limit)
        if self._cache_ttl_seconds > 0.0:
            cached = self._ohlcv_cache.get(key)
            if cached is not None:
                ts, cached_batch = cached
                if (time.monotonic() - ts) < self._cache_ttl_seconds:
                    self._ohlcv_cache.move_to_end(key)
                    return cached_batch
                self._ohlcv_cache.pop(key, None)

        batch = await self._provider.get_closed_ohlcv(symbol, timeframe, limit=limit)
        if self._cache_ttl_seconds > 0.0 and not batch.has_provider_error and batch.candles:
            self._ohlcv_cache[key] = (time.monotonic(), batch)
            self._ohlcv_cache.move_to_end(key)
            if len(self._ohlcv_cache) > self._cache_max_entries:
                self._ohlcv_cache.popitem(last=False)
        return batch

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        """Fetch exchange-wide 24h futures tickers if the market provider supports it."""
        if isinstance(self._provider, FuturesTickerProvider):
            return await self._provider.get_futures_tickers()
        return FuturesTickerBatch(
            venue="unknown",
            received_time=datetime.now(UTC),
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message="Market provider does not support exchange-wide futures tickers.",
        )

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        limit: int = 100,
    ) -> OhlcvBatch:
        """Fetch normalized closed candles for one symbol/timeframe via active provider."""
        return await self._fetch_closed_ohlcv(symbol, timeframe, limit)

    def _can_resample(self, target: Timeframe) -> bool:
        if self._resample_from is None:
            return False
        base_secs = int(self._resample_from.duration.total_seconds())
        target_secs = int(target.duration.total_seconds())
        return target_secs > base_secs and target_secs % base_secs == 0

    async def _resolve_maintenance_margin(
        self,
        canonical_symbol: str,
        *,
        pinned_venue: str | None,
    ) -> tuple[float | None, float | None]:
        if self._risk_tiers_client is None:
            return None, None
        if pinned_venue is not None and pinned_venue not in {"gate", "gateio"}:
            return None, None
        return await self._risk_tiers_client.get_mmr_for_symbol(
            canonical_symbol,
            notional_usdt=self._default_position_notional_usdt,
        )

    async def _fetch_timeframe_bundle(
        self,
        canonical_symbol: str,
        requests: Sequence[tuple[Timeframe, int]],
    ) -> tuple[dict[Timeframe, OhlcvBatch], MarketDataProvider, str | None]:
        """Fetch a coherent set of timeframes from one pinned venue, resampling when configured."""
        dedup_limits: dict[Timeframe, int] = {}
        for tf, req_limit in requests:
            dedup_limits[tf] = max(dedup_limits.get(tf, 0), max(1, req_limit))

        resample_reqs: list[tuple[Timeframe, int]] = []
        fetch_limits: dict[Timeframe, int] = {}
        base_tf = self._resample_from
        for tf, req_limit in dedup_limits.items():
            if base_tf is not None and self._can_resample(tf):
                resample_reqs.append((tf, req_limit))
                ratio = int(tf.duration.total_seconds()) // int(base_tf.duration.total_seconds())
                needed_base = req_limit * ratio + ratio
                fetch_limits[base_tf] = max(fetch_limits.get(base_tf, 0), needed_base)
            else:
                fetch_limits[tf] = max(fetch_limits.get(tf, 0), req_limit)

        pinned_provider: MarketDataProvider = self._provider
        pinned_venue: str | None = None
        if isinstance(self._provider, FallbackMarketDataProvider):
            bundle_requests = [(canonical_symbol, tf, lim) for tf, lim in fetch_limits.items()]
            bundle = await self._provider.get_closed_ohlcv_bundle(bundle_requests)
            if bundle.provider is not None:
                pinned_provider = bundle.provider
                pinned_venue = bundle.venue
            batches = {tf: bundle.batches[(canonical_symbol, tf)] for tf in fetch_limits}
            if (
                base_tf is not None
                and base_tf in batches
                and fetch_limits[base_tf] > 999
                and len(batches[base_tf].candles) < fetch_limits[base_tf]
                and isinstance(pinned_provider, HistoryMarketDataProvider)
            ):
                batches[base_tf] = await pinned_provider.get_closed_ohlcv_history(
                    canonical_symbol,
                    base_tf,
                    limit=fetch_limits[base_tf],
                )
        else:
            ordered_reqs = list(fetch_limits.items())

            async def _fetch_one(tf: Timeframe, lim: int) -> OhlcvBatch:
                if lim > 999 and isinstance(self._provider, HistoryMarketDataProvider):
                    return await self._provider.get_closed_ohlcv_history(
                        canonical_symbol, tf, limit=lim
                    )
                return await self._fetch_closed_ohlcv(canonical_symbol, tf, lim)

            results = await asyncio.gather(*(_fetch_one(tf, lim) for tf, lim in ordered_reqs))
            batches = dict(zip((tf for tf, _ in ordered_reqs), results, strict=True))

        if resample_reqs and base_tf is not None:
            base_batch = batches[base_tf]
            for target_tf, req_limit in resample_reqs:
                if base_batch.candles and not base_batch.has_provider_error:
                    synthesized = resample_candles(
                        base_batch.candles,
                        target_timeframe=target_tf,
                        as_of=base_batch.candles[-1].close_time,
                    )
                    if len(synthesized.candles) > req_limit:
                        synthesized = synthesized.model_copy(
                            update={"candles": synthesized.candles[-req_limit:]}
                        )
                    batches[target_tf] = synthesized
                else:
                    batches[target_tf] = OhlcvBatch(
                        metadata=base_batch.metadata,
                        canonical_symbol=base_batch.canonical_symbol,
                        provider_symbol=base_batch.provider_symbol,
                        timeframe=target_tf,
                        received_time=base_batch.received_time,
                        candles=(),
                        quality_flags=base_batch.quality_flags or (QualityFlag.PROVIDER_ERROR,),
                        error_message=base_batch.error_message,
                    )

        return batches, pinned_provider, pinned_venue

    async def warmup_candles(
        self,
        symbol: str,
        timeframes: Sequence[Timeframe],
        *,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> dict[str, int]:
        """Idempotently prefill `market_candles` for one symbol across timeframes from one venue."""
        mapping = resolve_default_perp_symbol(symbol)
        unique_tfs = tuple(dict.fromkeys(timeframes))
        requests = [(tf, limit) for tf in unique_tfs]
        batches, _pinned_provider, _pinned_venue = await self._fetch_timeframe_bundle(
            mapping.canonical_symbol,
            requests,
        )
        counts: dict[str, int] = {}
        for tf in unique_tfs:
            batch = batches.get(tf)
            if batch is None or batch.has_provider_error or not batch.candles:
                counts[tf.value] = 0
                continue
            if self._repository is not None:
                self._repository.save_candles(batch.candles)
            counts[tf.value] = len(batch.candles)
        return counts

    async def analyze_multi_style(
        self,
        symbol: str,
        timeframes: Sequence[Timeframe] = MULTI_STYLE_SCAN_TIMEFRAMES,
        *,
        limit: int = DEFAULT_CANDLE_LIMIT,
        include_derivatives: bool = True,
    ) -> dict[Timeframe, AnalysisSnapshot]:
        """Evaluate multiple trading-style timeframes concurrently with a single data fetch."""
        requested_tfs = tuple(dict.fromkeys(timeframes))
        if not requested_tfs:
            return {}

        mapping = resolve_default_perp_symbol(symbol)
        base_core = (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)
        core_tfs: list[Timeframe] = list(base_core)
        for tf in requested_tfs:
            if tf not in core_tfs:
                core_tfs.append(tf)

        def _fetch_limit(tf: Timeframe) -> int:
            if tf is Timeframe.M15:
                return max(limit, 200)
            return limit

        core_requests = [(tf, _fetch_limit(tf)) for tf in core_tfs]
        batches, pinned_provider, pinned_venue = await self._fetch_timeframe_bundle(
            mapping.canonical_symbol,
            core_requests,
        )

        effective_symbol = mapping.canonical_symbol
        effective_base = mapping.base
        any_requested_ok = False
        for tf in requested_tfs:
            batch = batches.get(tf)
            if (
                batch is not None
                and bool(batch.candles)
                and determine_quality_status(len(batch.candles), batch.quality_flags)
                is not QualityStatus.INSUFFICIENT_DATA
            ):
                if not any_requested_ok:
                    effective_symbol = batch.canonical_symbol
                    effective_base = batch.metadata.base_unit
                    if pinned_venue is None:
                        pinned_venue = batch.metadata.venue
                any_requested_ok = True

        deriv_source: DerivativesLiquidityProvider | None = self._derivatives_provider
        if isinstance(deriv_source, FallbackDerivativesLiquidityProvider) and pinned_venue:
            deriv_source = deriv_source.provider_for_venue(pinned_venue) or deriv_source

        derivatives_context: DerivativesContext | None = None
        cross_venue: CrossVenueDerivativesSummary | None = None
        if any_requested_ok:
            if Timeframe.M5 not in batches:
                m5_batch = await self._fetch_optional_ltf(
                    effective_symbol,
                    Timeframe.M5,
                    limit=150,
                    provider=pinned_provider,
                )
                if m5_batch is not None:
                    batches[Timeframe.M5] = m5_batch

            if deriv_source is not None and include_derivatives:
                oi_tf = max(
                    requested_tfs,
                    key=lambda t: (
                        Timeframe.M5,
                        Timeframe.M15,
                        Timeframe.H1,
                        Timeframe.H4,
                        Timeframe.D1,
                    ).index(t),
                )
                funding_batch, oi_batch, ob_batch, liq_batch = await asyncio.gather(
                    deriv_source.get_funding_rate(effective_base),
                    deriv_source.get_open_interest(effective_base, oi_tf),
                    deriv_source.get_order_book(
                        effective_symbol,
                        market_type=mapping.market_type,
                    ),
                    deriv_source.get_liquidations(effective_base),
                )
                derivatives_context = DerivativesContext(
                    funding=funding_batch,
                    open_interest=oi_batch,
                    order_book=ob_batch,
                    liquidations=liq_batch,
                )

            if self._bybit_provider is not None and include_derivatives:
                try:
                    bybit_funding = await self._bybit_provider.get_funding_rate(effective_base)
                    bybit_context = (
                        DerivativesContext(funding=bybit_funding)
                        if bybit_funding.observation is not None
                        else None
                    )
                    cross_venue = compute_cross_venue_derivatives(
                        derivatives_context,
                        bybit_context,
                    )
                except Exception:
                    cross_venue = None

        mmr_rate: float | None = None
        venue_max_lev: float | None = None
        if any_requested_ok:
            mmr_rate, venue_max_lev = await self._resolve_maintenance_margin(
                effective_symbol,
                pinned_venue=pinned_venue,
            )

        snapshots: dict[Timeframe, AnalysisSnapshot] = {}
        for tf in requested_tfs:
            primary = batches[tf]
            mtf_batches = {k: v for k, v in batches.items() if k is not tf}
            snapshots[tf] = build_analysis_snapshot(
                primary,
                mtf_batches=mtf_batches,
                derivatives_context=derivatives_context,
                cross_venue_derivatives=cross_venue,
                maintenance_margin_rate=mmr_rate,
                venue_max_leverage=venue_max_lev,
            )
        return snapshots

    async def _fetch_optional_ltf(
        self,
        canonical_symbol: str,
        tf: Timeframe,
        *,
        limit: int = 200,
        provider: MarketDataProvider | None = None,
    ) -> OhlcvBatch | None:
        try:
            if provider is not None and provider is not self._provider:
                batch = await provider.get_closed_ohlcv(canonical_symbol, tf, limit=limit)
            else:
                batch = await self._fetch_closed_ohlcv(
                    canonical_symbol,
                    tf,
                    limit,
                )
        except KeyError:
            return None
        if batch.has_provider_error or not batch.candles:
            return None
        return batch

    async def analyze(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        limit: int = DEFAULT_CANDLE_LIMIT,
        include_mtf: bool = True,
        include_derivatives: bool = True,
        persist: bool = True,
    ) -> AnalysisSnapshot:
        """Fetch market data, build an `AnalysisSnapshot`, and optionally persist it."""
        snapshot, _ = await self.analyze_with_batches(
            symbol,
            timeframe,
            limit=limit,
            include_mtf=include_mtf,
            include_derivatives=include_derivatives,
            persist=persist,
        )
        return snapshot

    async def analyze_with_batches(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        limit: int = DEFAULT_CANDLE_LIMIT,
        include_mtf: bool = True,
        include_derivatives: bool = True,
        persist: bool = True,
    ) -> tuple[AnalysisSnapshot, dict[Timeframe, OhlcvBatch]]:
        """Shared top-down analysis path returning `(AnalysisSnapshot, batches_by_tf)`.

        Used by both CLI/bot `analyze` and `alerts run` so 4h/1h/15m/5m candles,
        regimes, market structure, and setups are guaranteed identical.
        """
        mapping = resolve_default_perp_symbol(symbol)
        other_tfs = (
            [tf for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1) if tf is not timeframe]
            if include_mtf
            else []
        )
        bundle_requests: list[tuple[Timeframe, int]] = [
            (timeframe, limit),
            *((tf, limit) for tf in other_tfs),
        ]
        batches, pinned_provider, pinned_venue = await self._fetch_timeframe_bundle(
            mapping.canonical_symbol,
            bundle_requests,
        )
        primary = batches[timeframe]
        mtf_batches: dict[Timeframe, OhlcvBatch] | None = (
            {tf: batches[tf] for tf in other_tfs} if include_mtf else None
        )

        primary_ok = (
            bool(primary.candles)
            and determine_quality_status(len(primary.candles), primary.quality_flags)
            is not QualityStatus.INSUFFICIENT_DATA
        )

        effective_symbol = primary.canonical_symbol if primary_ok else mapping.canonical_symbol
        effective_base = primary.metadata.base_unit if primary_ok else mapping.base
        if primary_ok and pinned_venue is None:
            pinned_venue = primary.metadata.venue

        if include_mtf and primary_ok:
            if mtf_batches is None:  # pragma: no cover
                mtf_batches = {}
            optional_tasks: list[tuple[Timeframe, Any]] = []
            if timeframe is not Timeframe.M15:
                optional_tasks.append(
                    (
                        Timeframe.M15,
                        self._fetch_optional_ltf(
                            effective_symbol,
                            Timeframe.M15,
                            limit=200,
                            provider=pinned_provider,
                        ),
                    )
                )
            if timeframe is not Timeframe.M5:
                optional_tasks.append(
                    (
                        Timeframe.M5,
                        self._fetch_optional_ltf(
                            effective_symbol,
                            Timeframe.M5,
                            limit=150,
                            provider=pinned_provider,
                        ),
                    )
                )
            if optional_tasks:
                optional_results = await asyncio.gather(*(coro for _, coro in optional_tasks))
                for (opt_tf, _), opt_batch in zip(optional_tasks, optional_results, strict=True):
                    if opt_batch is not None:
                        mtf_batches[opt_tf] = opt_batch

        deriv_source = self._derivatives_provider
        if isinstance(deriv_source, FallbackDerivativesLiquidityProvider) and pinned_venue:
            deriv_source = deriv_source.provider_for_venue(pinned_venue) or deriv_source

        derivatives_context: DerivativesContext | None = None
        if deriv_source is not None and include_derivatives and primary_ok:
            funding_batch, oi_batch, ob_batch, liq_batch = await asyncio.gather(
                deriv_source.get_funding_rate(effective_base),
                deriv_source.get_open_interest(effective_base, timeframe),
                deriv_source.get_order_book(
                    effective_symbol,
                    market_type=mapping.market_type,
                ),
                deriv_source.get_liquidations(effective_base),
            )
            if (
                (funding_batch.observation is None or funding_batch.has_provider_error)
                and self._derivatives_repository is not None
                and primary.candles
            ):
                cached_funding = self._derivatives_repository.load_funding_observations(
                    venue=primary.metadata.venue,
                    canonical_symbol=effective_symbol,
                    end_available_time=primary.candles[-1].close_time,
                    limit=1,
                )
                if cached_funding:
                    funding_batch = funding_batch.model_copy(
                        update={
                            "observation": cached_funding[-1],
                            "quality_flags": tuple(
                                f
                                for f in funding_batch.quality_flags
                                if f is not QualityFlag.PROVIDER_ERROR
                            ),
                        }
                    )
            derivatives_context = DerivativesContext(
                funding=funding_batch,
                open_interest=oi_batch,
                order_book=ob_batch,
                liquidations=liq_batch,
            )

        cross_venue: CrossVenueDerivativesSummary | None = None
        if self._bybit_provider is not None and include_derivatives and primary_ok:
            try:
                bybit_funding = await self._bybit_provider.get_funding_rate(effective_base)
                bybit_context = (
                    DerivativesContext(funding=bybit_funding)
                    if bybit_funding.observation is not None
                    else None
                )
                cross_venue = compute_cross_venue_derivatives(derivatives_context, bybit_context)
            except Exception:
                cross_venue = None

        tv_packet: TradingViewPacket | None = None
        if self._tradingview_repository is not None and primary_ok and primary.candles:
            tv_packet = self._tradingview_repository.load_latest_packet(
                venue=primary.metadata.venue,
                canonical_symbol=mapping.canonical_symbol,
                market_type=primary.metadata.market_type,
                timeframe=timeframe,
                end_available_time=primary.candles[-1].close_time,
            )

        news_observations: tuple[SentimentObservation, ...] | None = None
        news_status = SentimentSourceStatus(available=False, reason="not_wired")
        if self._news_repository is not None and primary_ok and primary.candles:
            primary_as_of = primary.candles[-1].close_time
            spot_symbol_key = f"{mapping.base}/USDT"
            try:
                news_observations = self._news_repository.load_observations(
                    canonical_symbol=mapping.canonical_symbol,
                    start_available_time=primary_as_of - timedelta(hours=24),
                    end_available_time=primary_as_of,
                )
                if not news_observations and spot_symbol_key != mapping.canonical_symbol:
                    news_observations = self._news_repository.load_observations(
                        canonical_symbol=spot_symbol_key,
                        start_available_time=primary_as_of - timedelta(hours=24),
                        end_available_time=primary_as_of,
                    )
                if news_observations:
                    agg_check = aggregate_sentiment(
                        news_observations,
                        canonical_symbol=mapping.canonical_symbol,
                        as_of=primary_as_of,
                    )
                    if agg_check is not None:
                        news_status = SentimentSourceStatus(available=True)
                    else:
                        news_status = SentimentSourceStatus(
                            available=False,
                            reason="not_yet_available",
                            detail="observasi berita belum memenuhi syarat waktu/skor pada as_of",
                        )
                else:
                    future_probe = self._news_repository.load_observations(
                        canonical_symbol=mapping.canonical_symbol,
                        start_available_time=primary_as_of,
                        limit=1,
                    )
                    if not future_probe and spot_symbol_key != mapping.canonical_symbol:
                        future_probe = self._news_repository.load_observations(
                            canonical_symbol=spot_symbol_key,
                            start_available_time=primary_as_of,
                            limit=1,
                        )
                    if future_probe:
                        news_status = SentimentSourceStatus(
                            available=False,
                            reason="not_yet_available",
                            detail="berita tersedia setelah waktu as_of",
                        )
                    else:
                        news_status = SentimentSourceStatus(available=False, reason="no_data")
            except Exception as exc:
                logger.warning("News repository query failed for %s: %s", effective_symbol, exc)
                news_observations = None
                news_status = SentimentSourceStatus(
                    available=False,
                    reason="error",
                    detail=type(exc).__name__,
                )

        dxy_candles: tuple[MarketCandle, ...] | None = None
        if self._dxy_provider is not None and primary_ok and primary.candles:
            try:
                dxy_candles = await self._dxy_provider.get_dxy_candles(
                    timeframe,
                    limit=100,
                    now=primary.candles[-1].close_time,
                )
            except Exception:
                dxy_candles = None

        btc_candles: tuple[MarketCandle, ...] | None = None
        if "BTC" not in mapping.canonical_symbol.upper() and primary_ok:
            try:
                if self._can_resample(timeframe):
                    btc_batches, _, _ = await self._fetch_timeframe_bundle(
                        "BTC/USDT:USDT",
                        [(timeframe, limit)],
                    )
                    btc_batch = btc_batches[timeframe]
                else:
                    btc_batch = await pinned_provider.get_closed_ohlcv(
                        "BTC/USDT:USDT",
                        timeframe,
                        limit=limit,
                    )
                if btc_batch.candles:
                    btc_candles = btc_batch.candles
            except Exception:
                btc_candles = None

        positioning_panel: PositioningPanel | None = None
        pos_status = SentimentSourceStatus(available=False, reason="not_wired")
        if self._contract_stats_repository is not None and primary_ok and primary.candles:
            primary_as_of = primary.candles[-1].close_time
            win_start = primary_as_of - timedelta(days=30)
            try:
                stats_obs = self._contract_stats_repository.load_contract_stats(
                    venue="gateio",
                    canonical_symbol=mapping.canonical_symbol,
                    interval="5m",
                    start_time=win_start,
                    end_available_time=primary_as_of,
                )
                if not stats_obs:
                    stats_obs = self._contract_stats_repository.load_contract_stats(
                        venue="gate",
                        canonical_symbol=mapping.canonical_symbol,
                        interval="5m",
                        start_time=win_start,
                        end_available_time=primary_as_of,
                    )
                if not stats_obs:
                    future_stats = self._contract_stats_repository.load_contract_stats(
                        venue="gateio",
                        canonical_symbol=mapping.canonical_symbol,
                        interval="5m",
                        limit=1,
                    )
                    if not future_stats:
                        future_stats = self._contract_stats_repository.load_contract_stats(
                            venue="gate",
                            canonical_symbol=mapping.canonical_symbol,
                            interval="5m",
                            limit=1,
                        )
                    positioning_panel = compute_positioning_panel(
                        future_stats,
                        canonical_symbol=mapping.canonical_symbol,
                        as_of=primary_as_of,
                        use_available_time=True,
                    )
                else:
                    positioning_panel = compute_positioning_panel(
                        stats_obs,
                        canonical_symbol=mapping.canonical_symbol,
                        as_of=primary_as_of,
                        use_available_time=True,
                    )

                if positioning_panel.status == "UNAVAILABLE":
                    pos_status = SentimentSourceStatus(
                        available=False,
                        reason=positioning_panel.reason_code or "no_data",
                        detail=positioning_panel.note,
                    )
                elif positioning_panel.status == "WARMING_UP":
                    pos_status = SentimentSourceStatus(
                        available=True,
                        reason="insufficient_history",
                        detail=(
                            f"{positioning_panel.observation_count} bars / "
                            f"{positioning_panel.history_days:.1f}d < 7d"
                        ),
                    )
                else:
                    pos_status = SentimentSourceStatus(available=True)
            except Exception as exc:
                logger.warning(
                    "Contract stats repository query failed for %s: %s", effective_symbol, exc
                )
                positioning_panel = None
                pos_status = SentimentSourceStatus(
                    available=False,
                    reason="error",
                    detail=type(exc).__name__,
                )

        fear_greed_obs: FearGreedObservation | None = None
        fg_status = SentimentSourceStatus(available=False, reason="not_wired")
        if self._fear_greed_repository is not None and primary_ok and primary.candles:
            primary_as_of = primary.candles[-1].close_time
            try:
                fg_rows = self._fear_greed_repository.load_observations(
                    source="alternative.me",
                    end_available_time=primary_as_of,
                    limit=1,
                )
                if fg_rows:
                    fear_greed_obs = fg_rows[-1]
                    fg_status = SentimentSourceStatus(available=True)
                else:
                    any_fg = self._fear_greed_repository.load_observations(
                        source="alternative.me",
                        limit=1,
                    )
                    if any_fg:
                        fg_status = SentimentSourceStatus(
                            available=False,
                            reason="not_yet_available",
                            detail="data Fear & Greed tersedia setelah waktu as_of",
                        )
                    else:
                        fg_status = SentimentSourceStatus(available=False, reason="no_data")
            except Exception as exc:
                logger.warning("Fear & Greed repository query failed: %s", exc)
                fear_greed_obs = None
                fg_status = SentimentSourceStatus(
                    available=False,
                    reason="error",
                    detail=type(exc).__name__,
                )

        sent_diag = SentimentPanelDiagnostics(
            positioning=pos_status,
            fear_greed=fg_status,
            news=news_status,
        )

        mmr_rate: float | None = None
        venue_max_lev: float | None = None
        if primary_ok:
            mmr_rate, venue_max_lev = await self._resolve_maintenance_margin(
                effective_symbol,
                pinned_venue=pinned_venue or primary.metadata.venue,
            )

        snapshot = build_analysis_snapshot(
            primary,
            mtf_batches=mtf_batches,
            derivatives_context=derivatives_context,
            tradingview_packet=tv_packet,
            news_observations=news_observations,
            cross_venue_derivatives=cross_venue,
            dxy_candles=dxy_candles,
            btc_candles=btc_candles,
            positioning=positioning_panel,
            fear_greed=fear_greed_obs,
            sentiment_diagnostics=sent_diag,
            maintenance_margin_rate=mmr_rate,
            venue_max_leverage=venue_max_lev,
        )

        if self._shadow_engine is not None:
            record, summary = self._shadow_engine.predict_for_snapshot(
                snapshot,
                primary.candles,
                sentiment_observations=news_observations,
                use_available_time=True,
            )
            if summary is not None:
                snapshot = snapshot.model_copy(update={"ml_prediction": summary})
            if persist and self._prediction_repository is not None and record is not None:
                self._prediction_repository.save_prediction(record)

        if persist and self._repository is not None:
            if primary.candles:
                self._repository.save_candles(primary.candles)
            if mtf_batches is not None:
                for mtf_batch in mtf_batches.values():
                    if mtf_batch.candles:
                        self._repository.save_candles(mtf_batch.candles)
            self._repository.save_snapshot(snapshot)

        if persist and self._derivatives_repository is not None and derivatives_context is not None:
            if (
                derivatives_context.funding is not None
                and derivatives_context.funding.observation is not None
            ):
                self._derivatives_repository.save_funding_observations(
                    [derivatives_context.funding.observation]
                )
            if derivatives_context.open_interest is not None:
                oi_rows = [
                    *(
                        [derivatives_context.open_interest.latest]
                        if derivatives_context.open_interest.latest is not None
                        else []
                    ),
                    *derivatives_context.open_interest.history,
                ]
                if oi_rows:
                    self._derivatives_repository.save_open_interest_observations(oi_rows)
            if (
                derivatives_context.order_book is not None
                and derivatives_context.order_book.snapshot is not None
            ):
                self._derivatives_repository.save_order_book_snapshot(
                    derivatives_context.order_book.snapshot
                )
            if (
                derivatives_context.liquidations is not None
                and derivatives_context.liquidations.source_available
                and derivatives_context.liquidations.events
            ):
                self._derivatives_repository.save_liquidation_events(
                    derivatives_context.liquidations.events
                )

        if persist and self._setup_repository is not None and snapshot.trade_setup is not None:
            self._setup_repository.save_setup(snapshot, snapshot.trade_setup)

        all_batches: dict[Timeframe, OhlcvBatch] = {timeframe: primary}
        if mtf_batches is not None:
            all_batches.update(mtf_batches)
        return snapshot, all_batches

    def recompute_from_stored(self, snapshot_id: str) -> AnalysisSnapshot:
        """Recompute a persisted snapshot strictly from stored candles and verify identity."""
        if self._repository is None:
            raise ValueError("repository is required to recompute stored snapshots")

        stored = self._repository.get_snapshot(snapshot_id)
        if stored is None:
            raise ValueError(f"snapshot '{snapshot_id}' not found")
        if stored.input_range is None:
            raise ValueError(f"snapshot '{snapshot_id}' has no input_range to recompute")

        primary_candles = self._repository.load_candles(
            venue=stored.venue,
            canonical_symbol=stored.canonical_symbol,
            market_type=stored.market_type,
            timeframe=stored.timeframe,
            start_open_time=stored.input_range.first_open_time,
            end_open_time=stored.input_range.last_open_time,
        )
        if not primary_candles:
            raise ValueError(f"stored candles for snapshot '{snapshot_id}' were not found")

        primary_batch = _batch_from_stored_candles(
            primary_candles,
            as_of=stored.as_of,
            is_primary=True,
        )

        mtf_batches: dict[Timeframe, OhlcvBatch] | None = None
        if stored.technical is not None and (
            stored.technical.mtf_rsi_alignment is not None
            or (stored.trade_setup is not None and stored.trade_setup.ltf_confirmation is not None)
        ):
            mtf_batches = {}
            for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
                if tf is stored.timeframe:
                    continue
                mtf_candles = self._repository.load_candles(
                    venue=stored.venue,
                    canonical_symbol=stored.canonical_symbol,
                    market_type=stored.market_type,
                    timeframe=tf,
                    end_close_time=stored.as_of,
                    limit=DEFAULT_CANDLE_LIMIT,
                )
                if mtf_candles:
                    mtf_batches[tf] = _batch_from_stored_candles(
                        mtf_candles,
                        as_of=stored.as_of,
                        is_primary=False,
                    )
            if stored.timeframe is not Timeframe.M15:
                m15_candles = self._repository.load_candles(
                    venue=stored.venue,
                    canonical_symbol=stored.canonical_symbol,
                    market_type=stored.market_type,
                    timeframe=Timeframe.M15,
                    end_close_time=stored.as_of,
                    limit=200,
                )
                if m15_candles:
                    mtf_batches[Timeframe.M15] = _batch_from_stored_candles(
                        m15_candles,
                        as_of=stored.as_of,
                        is_primary=False,
                    )

        tv_packet: TradingViewPacket | None = None
        if self._tradingview_repository is not None and stored.tradingview is not None:
            candidates = self._tradingview_repository.load_packets(
                venue=stored.tradingview.venue,
                canonical_symbol=stored.tradingview.canonical_symbol,
                market_type=stored.tradingview.market_type,
                timeframe=stored.tradingview.timeframe,
                end_available_time=stored.as_of,
                limit=50,
            )
            tv_packet = next(
                (
                    pkt
                    for pkt in candidates
                    if pkt.idempotency_key == stored.tradingview.idempotency_key
                ),
                None,
            )

        news_observations: tuple[SentimentObservation, ...] | None = None
        if self._news_repository is not None and stored.sentiment is not None:
            news_observations = self._news_repository.load_observations(
                canonical_symbol=stored.canonical_symbol,
                start_available_time=stored.as_of - timedelta(hours=stored.sentiment.window_hours),
                end_available_time=stored.as_of,
            )

        return build_analysis_snapshot(
            primary_batch,
            mtf_batches=mtf_batches,
            tradingview_packet=tv_packet,
            news_observations=news_observations,
            macro_context=stored.macro_context,
        )
