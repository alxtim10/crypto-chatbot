"""Deterministic, non-probabilistic factor rules derived from AnalysisSnapshot facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from crypto_chatbot.analysis.regimes import (
    HIGH_VOLATILITY_ATR_PERCENTILE,
    RANGING_ADX_MAX,
    TRENDING_ADX_MIN,
)
from crypto_chatbot.analysis.structure import StructureAnalysis
from crypto_chatbot.schemas.analysis import (
    DealingZone,
    DerivativesMetrics,
    FactorDirection,
    FactorItem,
    MtfRsiAlignment,
    Regime,
    SmcSummary,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
)


def _add_if_facts_present(
    factors: list[FactorItem],
    fact_ids_by_key: Mapping[str, str],
    keys: Sequence[str],
    *,
    rule_id: str,
    description: str,
    direction: FactorDirection,
) -> None:
    resolved = [fact_ids_by_key[k] for k in keys if k in fact_ids_by_key]
    if len(resolved) != len(keys) or not resolved:
        return
    factors.append(
        FactorItem(
            rule_id=rule_id,
            description=description,
            direction=direction,
            fact_ids=tuple(resolved),
        )
    )


def build_factors(
    *,
    metrics: TechnicalMetrics,
    regime: Regime,
    structure: StructureAnalysis,
    failed_breakouts: Sequence[StructureEvent],
    recent_cutoff_time: datetime,
    fact_ids_by_key: Mapping[str, str],
    derivatives: DerivativesMetrics | None = None,
    smc: SmcSummary | None = None,
) -> tuple[FactorItem, ...]:
    """Build descriptive, strictly non-probabilistic rule factors referencing snapshot facts."""
    factors: list[FactorItem] = []

    if metrics.ema_200 is not None:
        if metrics.last_close > metrics.ema_200:
            desc = "Harga penutupan terakhir berada di atas EMA 200."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif metrics.last_close < metrics.ema_200:
            desc = "Harga penutupan terakhir berada di bawah EMA 200."
            direction = FactorDirection.SUPPORTS_BEARISH
        else:
            desc = "Harga penutupan terakhir tepat berada di EMA 200."
            direction = FactorDirection.NEUTRAL
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("last_close", "ema_200"),
            rule_id="price_vs_ema200",
            description=desc,
            direction=direction,
        )

    if metrics.ema_20 is not None and metrics.ema_50 is not None:
        if metrics.ema_20 > metrics.ema_50:
            desc = "EMA 20 berada di atas EMA 50 (keselarasan jangka pendek-menengah naik)."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif metrics.ema_20 < metrics.ema_50:
            desc = "EMA 20 berada di bawah EMA 50 (keselarasan jangka pendek-menengah turun)."
            direction = FactorDirection.SUPPORTS_BEARISH
        else:
            desc = "EMA 20 sejajar dengan EMA 50."
            direction = FactorDirection.NEUTRAL
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("ema_20", "ema_50"),
            rule_id="ema20_vs_ema50",
            description=desc,
            direction=direction,
        )

    if metrics.rsi_14 is not None:
        rsi = metrics.rsi_14
        if rsi > 70.0:
            desc = "RSI 14 berada di atas 70 (momentum kuat namun kondisi teregang/jenuh beli)."
            direction = FactorDirection.NEUTRAL
        elif rsi < 30.0:
            desc = (
                "RSI 14 berada di bawah 30 (tekanan turun kuat namun kondisi teregang/jenuh jual)."
            )
            direction = FactorDirection.NEUTRAL
        elif rsi > 50.0:
            desc = "RSI 14 berada di zona 50-70 (bias momentum naik)."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif rsi < 50.0:
            desc = "RSI 14 berada di zona 30-50 (bias momentum turun)."
            direction = FactorDirection.SUPPORTS_BEARISH
        else:
            desc = "RSI 14 tepat di level netral 50."
            direction = FactorDirection.NEUTRAL
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("rsi_14",),
            rule_id="rsi14_zone",
            description=desc,
            direction=direction,
        )

    if metrics.macd_histogram is not None:
        if metrics.macd_histogram > 0.0:
            desc = "Histogram MACD bernilai positif (garis MACD di atas garis sinyal)."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif metrics.macd_histogram < 0.0:
            desc = "Histogram MACD bernilai negatif (garis MACD di bawah garis sinyal)."
            direction = FactorDirection.SUPPORTS_BEARISH
        else:
            desc = "Histogram MACD tepat di nol."
            direction = FactorDirection.NEUTRAL
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("macd_histogram",),
            rule_id="macd_histogram",
            description=desc,
            direction=direction,
        )

    if (
        metrics.adx_14 is not None
        and metrics.plus_di_14 is not None
        and metrics.minus_di_14 is not None
    ):
        adx = metrics.adx_14
        if adx >= TRENDING_ADX_MIN and metrics.plus_di_14 > metrics.minus_di_14:
            desc = "ADX 14 >= 25 dengan +DI di atas -DI (pengukuran tren menguat naik)."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif adx >= TRENDING_ADX_MIN and metrics.minus_di_14 > metrics.plus_di_14:
            desc = "ADX 14 >= 25 dengan -DI di atas +DI (pengukuran tren menguat turun)."
            direction = FactorDirection.SUPPORTS_BEARISH
        elif adx < RANGING_ADX_MAX:
            desc = "ADX 14 < 20 (kekuatan tren lemah / kondisi cenderung mendatar)."
            direction = FactorDirection.NEUTRAL
        else:
            desc = "ADX 14 berada di zona transisi tanpa dominasi tren yang kuat."
            direction = FactorDirection.NEUTRAL
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("adx_14", "plus_di_14", "minus_di_14"),
            rule_id="adx_dmi_trend",
            description=desc,
            direction=direction,
        )

    if metrics.atr_percentile_100 is not None:
        if (
            regime is Regime.HIGH_VOLATILITY
            or metrics.atr_percentile_100 >= HIGH_VOLATILITY_ATR_PERCENTILE
        ):
            desc = "Persentil ATR >= 80 (volatilitas relatif tinggi dibanding 100 bar terakhir)."
        else:
            desc = "Persentil ATR di bawah 80 (volatilitas relatif dalam rentang normal)."
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("atr_percentile_100",),
            rule_id="volatility_regime",
            description=desc,
            direction=FactorDirection.NEUTRAL,
        )

    if metrics.relative_volume_20 is not None:
        rvol = metrics.relative_volume_20
        if rvol >= 1.5:
            desc = (
                "Volume relatif >= 1,5x rata-rata 20 bar sebelumnya "
                "(aktivitas transaksi meningkat)."
            )
        elif rvol < 0.7:
            desc = "Volume relatif < 0,7x rata-rata 20 bar sebelumnya (aktivitas transaksi rendah)."
        else:
            desc = "Volume relatif berada di sekitar rata-rata 20 bar sebelumnya."
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("relative_volume_20",),
            rule_id="relative_volume",
            description=desc,
            direction=FactorDirection.NEUTRAL,
        )

    if metrics.mtf_rsi_alignment is not None:
        if metrics.mtf_rsi_alignment is MtfRsiAlignment.ALIGNED_BULLISH:
            desc = "RSI 14 pada timeframe 1h, 4h, dan 1d seluruhnya berada di atas 50."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif metrics.mtf_rsi_alignment is MtfRsiAlignment.ALIGNED_BEARISH:
            desc = "RSI 14 pada timeframe 1h, 4h, dan 1d seluruhnya berada di bawah 50."
            direction = FactorDirection.SUPPORTS_BEARISH
        else:
            desc = "RSI 14 lintas timeframe 1h, 4h, dan 1d masih tercampur."
            direction = FactorDirection.NEUTRAL
        _add_if_facts_present(
            factors,
            fact_ids_by_key,
            ("mtf_rsi_1h", "mtf_rsi_4h", "mtf_rsi_1d"),
            rule_id="mtf_rsi_alignment",
            description=desc,
            direction=direction,
        )

    swing_fids = tuple(
        fact_ids_by_key[k]
        for k in ("latest_swing_high", "latest_swing_low")
        if k in fact_ids_by_key
    )
    if swing_fids:
        if structure.market_structure.value == "bullish":
            desc = "Struktur pivot terkonfirmasi membentuk higher high dan higher low."
            direction = FactorDirection.SUPPORTS_BULLISH
        elif structure.market_structure.value == "bearish":
            desc = "Struktur pivot terkonfirmasi membentuk lower high dan lower low."
            direction = FactorDirection.SUPPORTS_BEARISH
        else:
            desc = "Struktur pivot terkonfirmasi masih bercampur / belum membentuk arah tunggal."
            direction = FactorDirection.NEUTRAL
        factors.append(
            FactorItem(
                rule_id="market_structure",
                description=desc,
                direction=direction,
                fact_ids=swing_fids,
            )
        )

    if failed_breakouts:
        latest_fb = failed_breakouts[-1]
        if (
            latest_fb.event_time >= recent_cutoff_time
            and "latest_failed_breakout" in fact_ids_by_key
        ):
            if latest_fb.event_type is StructureEventType.FAILED_BREAKOUT_BULLISH:
                desc = (
                    "Terdeteksi gagal breakout atas pada swing high terkonfirmasi "
                    "dalam jendela bar terbaru."
                )
                direction = FactorDirection.SUPPORTS_BEARISH
            else:
                desc = (
                    "Terdeteksi gagal breakdown bawah pada swing low terkonfirmasi "
                    "dalam jendela bar terbaru."
                )
                direction = FactorDirection.SUPPORTS_BULLISH
            factors.append(
                FactorItem(
                    rule_id="latest_failed_breakout",
                    description=desc,
                    direction=direction,
                    fact_ids=(fact_ids_by_key["latest_failed_breakout"],),
                )
            )

    if derivatives is not None:
        if derivatives.funding_rate_per_8h is not None:
            fr_8h = derivatives.funding_rate_per_8h
            if fr_8h >= 0.0005:
                desc = (
                    "Funding rate ternormalisasi 8h bernilai positif tinggi "
                    "(konsentrasi posisi long lebih padat; bukan jaminan arah)."
                )
            elif fr_8h <= -0.0005:
                desc = (
                    "Funding rate ternormalisasi 8h bernilai negatif tinggi "
                    "(konsentrasi posisi short lebih padat; bukan jaminan arah)."
                )
            else:
                desc = (
                    "Funding rate ternormalisasi 8h berada dalam rentang moderat "
                    "(konteks posisi perpetual satu venue)."
                )
            _add_if_facts_present(
                factors,
                fact_ids_by_key,
                ("funding_rate_per_8h",),
                rule_id="funding_rate_8h",
                description=desc,
                direction=FactorDirection.NEUTRAL,
            )

        if derivatives.open_interest_change_pct is not None:
            oi_chg = derivatives.open_interest_change_pct
            if oi_chg >= 0.05:
                desc = (
                    "Open interest perpetual meningkat cukup kuat "
                    "(ekspansi kontrak terbuka di satu venue)."
                )
            elif oi_chg <= -0.05:
                desc = (
                    "Open interest perpetual menurun cukup kuat "
                    "(kontraksi/penutupan kontrak terbuka di satu venue)."
                )
            else:
                desc = "Perubahan open interest perpetual relatif moderat."
            _add_if_facts_present(
                factors,
                fact_ids_by_key,
                ("open_interest_change_pct",),
                rule_id="open_interest_change",
                description=desc,
                direction=FactorDirection.NEUTRAL,
            )

        if derivatives.spread_bps is not None and derivatives.imbalance_10bps is not None:
            imb = derivatives.imbalance_10bps
            if imb >= 0.25:
                desc = (
                    "Order book 10 bps condong pada sisi bid "
                    "(ketidakseimbangan likuiditas sesaat di satu venue)."
                )
            elif imb <= -0.25:
                desc = (
                    "Order book 10 bps condong pada sisi ask "
                    "(ketidakseimbangan likuiditas sesaat di satu venue)."
                )
            else:
                desc = "Kedalaman order book 10 bps relatif berimbang antara bid dan ask."
            _add_if_facts_present(
                factors,
                fact_ids_by_key,
                ("spread_bps", "imbalance_10bps"),
                rule_id="orderbook_liquidity_imbalance",
                description=desc,
                direction=FactorDirection.NEUTRAL,
            )

    if smc is not None:
        if smc.latest_structure_break is not None:
            brk_type = smc.latest_structure_break.event_type
            if brk_type in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH):
                desc = (
                    "SMC CHoCH bullish terkonfirmasi pada penutupan candle."
                    if brk_type is StructureEventType.CHOCH_BULLISH
                    else "SMC BOS bullish terkonfirmasi pada penutupan candle."
                )
                brk_dir = FactorDirection.SUPPORTS_BULLISH
            else:
                desc = (
                    "SMC CHoCH bearish terkonfirmasi pada penutupan candle."
                    if brk_type is StructureEventType.CHOCH_BEARISH
                    else "SMC BOS bearish terkonfirmasi pada penutupan candle."
                )
                brk_dir = FactorDirection.SUPPORTS_BEARISH
            _add_if_facts_present(
                factors,
                fact_ids_by_key,
                ("smc_latest_break",),
                rule_id="smc_structure_break",
                description=desc,
                direction=brk_dir,
            )

        if smc.dealing_zone is not None:
            if smc.dealing_zone is DealingZone.DISCOUNT:
                desc = "Harga berada di zona Discount (< 45% rentang swing terkonfirmasi)."
            elif smc.dealing_zone is DealingZone.PREMIUM:
                desc = "Harga berada di zona Premium (> 55% rentang swing terkonfirmasi)."
            else:
                desc = "Harga berada di area Equilibrium dealing range."
            _add_if_facts_present(
                factors,
                fact_ids_by_key,
                ("last_close", "smc_equilibrium"),
                rule_id="smc_dealing_zone",
                description=desc,
                direction=FactorDirection.NEUTRAL,
            )

        if smc.latest_sweep is not None:
            if smc.latest_sweep.event_type is StructureEventType.LIQUIDITY_SWEEP_BULLISH:
                desc = "Terdeteksi Liquidity Sweep di bawah swing low lalu menutup kembali di atas."
                sw_dir = FactorDirection.SUPPORTS_BULLISH
            else:
                desc = (
                    "Terdeteksi Liquidity Sweep di atas swing high lalu menutup kembali di bawah."
                )
                sw_dir = FactorDirection.SUPPORTS_BEARISH
            _add_if_facts_present(
                factors,
                fact_ids_by_key,
                ("smc_latest_sweep",),
                rule_id="smc_liquidity_sweep",
                description=desc,
                direction=sw_dir,
            )

    return tuple(factors)
