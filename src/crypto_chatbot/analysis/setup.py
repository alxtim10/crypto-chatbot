"""Deterministic conditional trade setup engine (Entry, SL, TP1-3) & settlement evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import NamedTuple

from crypto_chatbot.analysis.indicators import MIN_HISTORY_EMA_20, compute_technical_metrics
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import StructureAnalysis, analyze_market_structure
from crypto_chatbot.schemas.analysis import (
    DealingZone,
    DerivativesMetrics,
    MarketStructure,
    MtfRsiAlignment,
    Regime,
    SetupBias,
    SetupStatus,
    SmcSummary,
    StructureEventType,
    TechnicalMetrics,
    TradeSetup,
)
from crypto_chatbot.schemas.macro import MacroBenchmarkContext, PositioningRegime
from crypto_chatbot.schemas.market import MarketCandle, Timeframe
from crypto_chatbot.schemas.ml import CostAssumptions
from crypto_chatbot.schemas.news import SentimentLabel, SentimentMetrics
from crypto_chatbot.storage.repositories import TradeSetupRecord

SL_ATR_BUFFER = 0.5
MIN_SL_ATR = 0.5
MIN_LTF_SL_HTF_ATR = 0.35
MAX_SL_ATR = 3.5
MIN_RR_TP2 = 1.5
DEFAULT_TP1_R = 1.0
DEFAULT_TP2_R = 2.0
DEFAULT_TP3_R = 3.0
CROWDED_FUNDING_8H = 0.0010

SETUP_PARAMETERS: dict[str, int | float | str] = {
    "trade_setup_min_rr_tp2": MIN_RR_TP2,
    "trade_setup_sl_atr_buffer": SL_ATR_BUFFER,
    "trade_setup_tp_levels": 3,
}


def _net_reward_risk(
    *,
    entry_mid: float,
    target: float,
    stop_loss: float,
    cost_rate: float,
) -> float | None:
    """Reward:risk after subtracting round-trip execution cost from both reward and risk.

    Cost is applied in price terms as `cost_rate * entry_mid` on each of the reward (gross
    distance to the target) and the risk (gross distance to the stop). Returns `None` when
    the net risk is non-positive (cost dominates the stop distance).
    """
    cost_price = cost_rate * entry_mid
    gross_reward = abs(target - entry_mid)
    gross_risk = abs(entry_mid - stop_loss)
    net_reward = gross_reward - cost_price
    net_risk = gross_risk + cost_price
    if net_risk <= 0.0:
        return None
    net_rr = net_reward / net_risk
    return net_rr if net_rr > 0.0 else 0.0


class _LtfRefinement(NamedTuple):
    entry_low: float | None
    entry_high: float | None
    entry_source: str | None
    stop_loss: float | None
    invalidation_reason: str | None
    confirmation: str
    ltf_support: float | None = None
    ltf_resistance: float | None = None
    extra_risk_notes: tuple[str, ...] = ()
    has_counter_sweep: bool = False


def _wait_setup(
    reason: str,
    risk_notes: Sequence[str] = (),
    *,
    entry_timeframe: Timeframe | None = None,
    ltf_confirmation: str | None = None,
    htf_support: float | None = None,
    htf_resistance: float | None = None,
    ltf_support: float | None = None,
    ltf_resistance: float | None = None,
) -> TradeSetup:
    return TradeSetup(
        status=SetupStatus.WAIT,
        bias=SetupBias.NEUTRAL,
        entry_timeframe=entry_timeframe,
        ltf_confirmation=ltf_confirmation,
        htf_support=htf_support,
        htf_resistance=htf_resistance,
        ltf_support=ltf_support,
        ltf_resistance=ltf_resistance,
        wait_reason=reason,
        risk_notes=tuple(dict.fromkeys(risk_notes)),
    )


def _filter_causal_candles(
    candles: Sequence[MarketCandle],
    cutoff: datetime | None,
    *,
    use_available_time: bool,
) -> list[MarketCandle]:
    if cutoff is None:
        return list(candles)
    return [
        c for c in candles if (c.available_time if use_available_time else c.close_time) <= cutoff
    ]


def _evaluate_htf_filter_for_15m(
    *,
    bias: SetupBias,
    smc: SmcSummary,
    htf_candles_by_tf: Mapping[Timeframe, Sequence[MarketCandle]] | None,
    as_of: datetime | None,
    use_available_time: bool,
) -> tuple[str | None, str | None]:
    """Check HTF (4h/1h) trend alignment when analyzing 15m directly."""
    if not htf_candles_by_tf:
        return None, None

    chosen_tf: Timeframe | None = None
    chosen_candles: list[MarketCandle] = []
    for candidate_tf in (Timeframe.H4, Timeframe.H1):
        raw_seq = htf_candles_by_tf.get(candidate_tf)
        if not raw_seq:
            continue
        filtered = _filter_causal_candles(raw_seq, as_of, use_available_time=use_available_time)
        if len(filtered) >= MIN_HISTORY_EMA_20:
            chosen_tf = candidate_tf
            chosen_candles = filtered
            break

    if chosen_tf is None:
        return None, None

    htf_ind = compute_technical_metrics(chosen_candles)
    htf_regime = classify_regime(htf_ind.metrics)
    htf_struct = analyze_market_structure(chosen_candles)
    ref_ema = (
        htf_ind.metrics.ema_50 if htf_ind.metrics.ema_50 is not None else htf_ind.metrics.ema_20
    )
    adx = htf_ind.metrics.adx_14
    pdi = htf_ind.metrics.plus_di_14
    mdi = htf_ind.metrics.minus_di_14
    last_close = htf_ind.metrics.last_close

    adx_bull = adx is not None and adx >= 25.0 and pdi is not None and mdi is not None and pdi > mdi
    adx_bear = adx is not None and adx >= 25.0 and pdi is not None and mdi is not None and mdi > pdi
    htf_bull = htf_regime is Regime.TRENDING_UP or (
        ref_ema is not None
        and last_close > ref_ema
        and (adx_bull or htf_struct.market_structure is MarketStructure.BULLISH)
    )
    htf_bear = htf_regime is Regime.TRENDING_DOWN or (
        ref_ema is not None
        and last_close < ref_ema
        and (adx_bear or htf_struct.market_structure is MarketStructure.BEARISH)
    )

    brk = smc.latest_structure_break.event_type if smc.latest_structure_break else None
    swp = smc.latest_sweep.event_type if smc.latest_sweep else None
    bull_reversal = (
        swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH
        and brk is StructureEventType.CHOCH_BULLISH
    )
    bear_reversal = (
        swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH
        and brk is StructureEventType.CHOCH_BEARISH
    )

    if bias is SetupBias.LONG and htf_bear and not bull_reversal:
        return (
            f"Setup LONG 15m berlawanan dengan tren turun HTF ({chosen_tf.value}) "
            "tanpa konfirmasi Liquidity Sweep + CHoCH 15m.",
            None,
        )
    if bias is SetupBias.SHORT and htf_bull and not bear_reversal:
        return (
            f"Setup SHORT 15m berlawanan dengan tren naik HTF ({chosen_tf.value}) "
            "tanpa konfirmasi Liquidity Sweep + CHoCH 15m.",
            None,
        )

    if (bias is SetupBias.LONG and htf_bull) or (bias is SetupBias.SHORT and htf_bear):
        return None, f"Selaras tren/struktur HTF ({chosen_tf.value})"
    return None, f"HTF ({chosen_tf.value}) netral/transisi; fokus konfirmasi 15m"


def _extract_sr_levels(
    structure: StructureAnalysis,
    smc: SmcSummary,
) -> tuple[float | None, float | None]:
    sup = (
        smc.dealing_range_low
        if smc.dealing_range_low is not None
        else (structure.last_swing_lows[-1].level if structure.last_swing_lows else None)
    )
    res = (
        smc.dealing_range_high
        if smc.dealing_range_high is not None
        else (structure.last_swing_highs[-1].level if structure.last_swing_highs else None)
    )
    return sup, res


def _refine_with_ltf_15m(
    *,
    bias: SetupBias,
    last: float,
    htf_atr: float,
    primary_timeframe: Timeframe,
    ltf_candles: Sequence[MarketCandle],
    as_of: datetime | None,
    use_available_time: bool,
) -> _LtfRefinement | None:
    """Refine Entry Zone and Stop Loss on causal 15m structure aligned with HTF bias."""
    filtered = _filter_causal_candles(ltf_candles, as_of, use_available_time=use_available_time)
    if len(filtered) < MIN_HISTORY_EMA_20 or filtered[0].timeframe is not Timeframe.M15:
        return None

    ltf_ind = compute_technical_metrics(filtered)
    ltf_struct = analyze_market_structure(filtered)
    ltf_smc = analyze_smc(filtered, structure=ltf_struct)
    ltf_sup, ltf_res = _extract_sr_levels(ltf_struct, ltf_smc)

    ltf_brk = ltf_smc.latest_structure_break.event_type if ltf_smc.latest_structure_break else None
    ltf_swp = ltf_smc.latest_sweep.event_type if ltf_smc.latest_sweep else None
    ltf_atr = (
        ltf_ind.metrics.atr_14
        if ltf_ind.metrics.atr_14 is not None and ltf_ind.metrics.atr_14 > 0
        else htf_atr * 0.35
    )

    e_low: float | None = None
    e_high: float | None = None
    e_src: str | None = None
    sl_15m: float | None = None
    inv_15m: str | None = None
    extra_notes: list[str] = []

    if bias is SetupBias.LONG:
        has_counter_sweep = ltf_swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH
        if (
            ltf_swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH
            and ltf_brk is StructureEventType.CHOCH_BULLISH
        ):
            confirmation = "15m CHoCH BULLISH + Liquidity Sweep terkonfirmasi"
        elif ltf_brk in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH):
            ev_name = (
                "CHoCH BULLISH" if ltf_brk is StructureEventType.CHOCH_BULLISH else "BOS BULLISH"
            )
            confirmation = f"15m {ev_name} terkonfirmasi"
        elif ltf_struct.market_structure is MarketStructure.BULLISH:
            confirmation = "15m struktur bullish selaras bias HTF"
        elif (
            ltf_brk in (StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH)
            or ltf_struct.market_structure is MarketStructure.BEARISH
        ):
            confirmation = "15m masih koreksi berlawanan; tunggu CHoCH bullish 15m di zona entry"
            extra_notes.append(
                "Struktur 15m masih melemah; tunggu konfirmasi CHoCH bullish 15m sebelum entry."
            )
        else:
            confirmation = "15m konsolidasi di dekat area S/R; pantau reaksi di zona entry 15m"

        fvg = ltf_smc.active_bullish_fvg
        ob = ltf_smc.active_bullish_ob
        ema20 = ltf_ind.metrics.ema_20
        lows_below = [
            ev.level
            for ev in ltf_struct.last_swing_lows
            if 0 < ev.level < last and (last - ev.level) <= 1.8 * htf_atr
        ]
        if fvg is not None and 0 < fvg.zone_low <= last and (last - fvg.zone_high) <= 2.2 * htf_atr:
            e_low = fvg.zone_low
            e_high = max(fvg.zone_low, min(fvg.zone_high, last))
            e_src = f"Refinement 15m: Bullish FVG (bias {primary_timeframe.value})"
        elif ob is not None and 0 < ob.zone_low <= last and (last - ob.zone_high) <= 2.2 * htf_atr:
            e_low = ob.zone_low
            e_high = max(ob.zone_low, min(ob.zone_high, last))
            e_src = f"Refinement 15m: Bullish Order Block (bias {primary_timeframe.value})"
        elif lows_below:
            near_sup = max(lows_below)
            e_low = near_sup
            e_high = max(e_low, min(last, near_sup + 0.5 * ltf_atr))
            e_src = f"Refinement 15m: Area Support / Swing Low 15m (bias {primary_timeframe.value})"
        elif ema20 is not None and ema20 < last and (last - ema20) <= 1.8 * htf_atr:
            e_low = max(1e-8, ema20 - 0.2 * ltf_atr)
            e_high = max(e_low, min(ema20 + 0.2 * ltf_atr, last))
            e_src = f"Refinement 15m: Pullback EMA 20 15m (bias {primary_timeframe.value})"
        else:
            e_low = max(1e-8, last - 0.6 * ltf_atr)
            e_high = max(e_low, last - 0.15 * ltf_atr)
            e_src = f"Refinement 15m: Zona Pullback Presisi 15m (bias {primary_timeframe.value})"

        if e_low is not None and e_high is not None:
            e_mid = (e_low + e_high) / 2.0
            ltf_lows = [
                ev.level
                for ev in ltf_struct.last_swing_lows
                if ev.level < e_low and (e_mid - ev.level) <= 2.5 * htf_atr
            ]
            ltf_floor = max(ltf_lows) if ltf_lows else (e_low - 1.2 * ltf_atr)
            sl_15m = min(
                ltf_floor - SL_ATR_BUFFER * ltf_atr,
                e_low - 0.25 * ltf_atr,
                e_mid - MIN_LTF_SL_HTF_ATR * htf_atr,
            )
            if ltf_sup is not None and ltf_sup < e_low:
                sl_15m = min(sl_15m, ltf_sup - SL_ATR_BUFFER * ltf_atr)
            inv_15m = "Di bawah Support / Swing Low 15m terkonfirmasi + buffer ATR"
    else:
        has_counter_sweep = ltf_swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH
        if (
            ltf_swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH
            and ltf_brk is StructureEventType.CHOCH_BEARISH
        ):
            confirmation = "15m CHoCH BEARISH + Liquidity Sweep terkonfirmasi"
        elif ltf_brk in (StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH):
            ev_name = (
                "CHoCH BEARISH" if ltf_brk is StructureEventType.CHOCH_BEARISH else "BOS BEARISH"
            )
            confirmation = f"15m {ev_name} terkonfirmasi"
        elif ltf_struct.market_structure is MarketStructure.BEARISH:
            confirmation = "15m struktur bearish selaras bias HTF"
        elif (
            ltf_brk in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH)
            or ltf_struct.market_structure is MarketStructure.BULLISH
        ):
            confirmation = "15m masih rebound berlawanan; tunggu CHoCH bearish 15m di zona entry"
            extra_notes.append(
                "Struktur 15m masih rebound; tunggu konfirmasi CHoCH bearish 15m sebelum entry."
            )
        else:
            confirmation = "15m konsolidasi di dekat area S/R; pantau reaksi di zona entry 15m"

        fvg = ltf_smc.active_bearish_fvg
        ob = ltf_smc.active_bearish_ob
        ema20 = ltf_ind.metrics.ema_20
        highs_above = [
            ev.level
            for ev in ltf_struct.last_swing_highs
            if ev.level > last and (ev.level - last) <= 1.8 * htf_atr
        ]
        if fvg is not None and fvg.zone_high >= last and (fvg.zone_low - last) <= 2.2 * htf_atr:
            e_low = min(fvg.zone_high, max(fvg.zone_low, last))
            e_high = fvg.zone_high
            e_src = f"Refinement 15m: Bearish FVG (bias {primary_timeframe.value})"
        elif ob is not None and ob.zone_high >= last and (ob.zone_low - last) <= 2.2 * htf_atr:
            e_low = min(ob.zone_high, max(ob.zone_low, last))
            e_high = ob.zone_high
            e_src = f"Refinement 15m: Bearish Order Block (bias {primary_timeframe.value})"
        elif highs_above:
            near_res = min(highs_above)
            e_low = min(near_res, max(last, near_res - 0.5 * ltf_atr))
            e_high = near_res
            e_src = (
                f"Refinement 15m: Area Resistance / Swing High 15m (bias {primary_timeframe.value})"
            )
        elif ema20 is not None and ema20 > last and (ema20 - last) <= 1.8 * htf_atr:
            e_low = max(last, ema20 - 0.2 * ltf_atr)
            e_high = max(e_low, ema20 + 0.2 * ltf_atr)
            e_src = f"Refinement 15m: Reli EMA 20 15m (bias {primary_timeframe.value})"
        else:
            e_low = last + 0.15 * ltf_atr
            e_high = last + 0.6 * ltf_atr
            e_src = f"Refinement 15m: Zona Reli Presisi 15m (bias {primary_timeframe.value})"

        if e_low is not None and e_high is not None:
            e_mid = (e_low + e_high) / 2.0
            ltf_highs = [
                ev.level
                for ev in ltf_struct.last_swing_highs
                if ev.level > e_high and (ev.level - e_mid) <= 2.5 * htf_atr
            ]
            ltf_ceil = min(ltf_highs) if ltf_highs else (e_high + 1.2 * ltf_atr)
            sl_15m = max(
                ltf_ceil + SL_ATR_BUFFER * ltf_atr,
                e_high + 0.25 * ltf_atr,
                e_mid + MIN_LTF_SL_HTF_ATR * htf_atr,
            )
            if ltf_res is not None and ltf_res > e_high:
                sl_15m = max(sl_15m, ltf_res + SL_ATR_BUFFER * ltf_atr)
            inv_15m = "Di atas Resistance / Swing High 15m terkonfirmasi + buffer ATR"

    return _LtfRefinement(
        entry_low=e_low,
        entry_high=e_high,
        entry_source=e_src,
        stop_loss=sl_15m,
        invalidation_reason=inv_15m,
        confirmation=confirmation,
        ltf_support=ltf_sup,
        ltf_resistance=ltf_res,
        extra_risk_notes=tuple(extra_notes),
        has_counter_sweep=has_counter_sweep,
    )


def _collect_risk_notes(
    *,
    bias: SetupBias,
    smc: SmcSummary,
    derivatives: DerivativesMetrics | None,
    sentiment: SentimentMetrics | None = None,
) -> list[str]:
    notes: list[str] = []
    if bias is SetupBias.LONG and smc.dealing_zone is DealingZone.PREMIUM:
        notes.append(
            "Harga berada di zona Premium dealing range; hindari FOMO dan tunggu retracement "
            "ke zona entry."
        )
    elif bias is SetupBias.SHORT and smc.dealing_zone is DealingZone.DISCOUNT:
        notes.append(
            "Harga berada di zona Discount dealing range; waspada pantulan sebelum entry short."
        )

    if derivatives is not None:
        fr8 = derivatives.funding_rate_per_8h
        if fr8 is not None:
            if bias is SetupBias.LONG and fr8 >= CROWDED_FUNDING_8H:
                notes.append(
                    "Funding rate 8h sangat positif (crowded long; waspada risiko long squeeze)."
                )
            elif bias is SetupBias.SHORT and fr8 <= -CROWDED_FUNDING_8H:
                notes.append(
                    "Funding rate 8h sangat negatif (crowded short; waspada risiko short squeeze)."
                )
        if derivatives.spread_bps is not None and derivatives.spread_bps >= 10.0:
            notes.append("Spread order book agak lebar; perhatikan risiko slippage saat eksekusi.")
        if derivatives.imbalance_10bps is not None:
            if bias is SetupBias.LONG and derivatives.imbalance_10bps <= -0.35:
                notes.append(
                    "Order book 10 bps condong berat di sisi ask (tekanan jual jangka pendek)."
                )
            elif bias is SetupBias.SHORT and derivatives.imbalance_10bps >= 0.35:
                notes.append(
                    "Order book 10 bps condong tebal di sisi bid (dukungan beli jangka pendek)."
                )
    if sentiment is not None and sentiment.article_count >= 1 and sentiment.mean_confidence >= 0.50:
        if bias is SetupBias.LONG and sentiment.dominant_label is SentimentLabel.NEGATIVE:
            notes.append(
                "Sentimen berita 24h cenderung negatif (faktor pendukung non-probabilistik; "
                "waspadai headline berlawanan)."
            )
        elif bias is SetupBias.SHORT and sentiment.dominant_label is SentimentLabel.POSITIVE:
            notes.append(
                "Sentimen berita 24h cenderung positif (faktor pendukung non-probabilistik; "
                "waspadai headline berlawanan)."
            )
    return notes


def _resolve_bias(
    *,
    metrics: TechnicalMetrics,
    regime: Regime,
    structure: StructureAnalysis,
    smc: SmcSummary,
) -> tuple[SetupBias, str | None]:
    brk = smc.latest_structure_break.event_type if smc.latest_structure_break else None
    swp = smc.latest_sweep.event_type if smc.latest_sweep else None

    aligned_bull_sweep_choch = (
        swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH
        and brk is StructureEventType.CHOCH_BULLISH
    )
    aligned_bear_sweep_choch = (
        swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH
        and brk is StructureEventType.CHOCH_BEARISH
    )

    if regime is Regime.RANGING and not (aligned_bull_sweep_choch or aligned_bear_sweep_choch):
        return (
            SetupBias.NEUTRAL,
            "Rezim pasar sedang ranging/mendatar tanpa konfirmasi Liquidity Sweep + CHoCH.",
        )

    if (
        regime is Regime.HIGH_VOLATILITY
        and structure.market_structure is MarketStructure.MIXED
        and brk is None
    ):
        return (
            SetupBias.NEUTRAL,
            "Volatilitas tinggi dengan struktur pasar bercampur (risiko whipsaw tinggi).",
        )

    if aligned_bull_sweep_choch:
        return SetupBias.LONG, None
    if aligned_bear_sweep_choch:
        return SetupBias.SHORT, None

    score = 0
    if regime is Regime.TRENDING_UP:
        score += 2
    elif regime is Regime.TRENDING_DOWN:
        score -= 2

    if structure.market_structure is MarketStructure.BULLISH:
        score += 2
    elif structure.market_structure is MarketStructure.BEARISH:
        score -= 2

    if brk in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH):
        score += 2
    elif brk in (StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH):
        score -= 2

    if swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH:
        score += 1
    elif swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH:
        score -= 1

    if metrics.ema_50 is not None:
        if metrics.last_close > metrics.ema_50:
            score += 1
        elif metrics.last_close < metrics.ema_50:
            score -= 1

    if metrics.mtf_rsi_alignment is MtfRsiAlignment.ALIGNED_BULLISH:
        score += 1
    elif metrics.mtf_rsi_alignment is MtfRsiAlignment.ALIGNED_BEARISH:
        score -= 1

    # Explicit conflict guard when primary trend opposes recent structural CHoCH/break
    if (
        regime is Regime.TRENDING_UP and brk is StructureEventType.CHOCH_BEARISH and score > -2
    ) or (regime is Regime.TRENDING_DOWN and brk is StructureEventType.CHOCH_BULLISH and score < 2):
        return (
            SetupBias.NEUTRAL,
            "Tren utama berlawanan dengan CHoCH terbaru; tunggu konfirmasi struktur lanjutan.",
        )

    if score >= 2:
        return SetupBias.LONG, None
    if score <= -2:
        return SetupBias.SHORT, None

    return (
        SetupBias.NEUTRAL,
        "Sinyal tren, struktur SMC, dan momentum belum selaras (bias netral / WAIT).",
    )


def _build_htf_levels_and_sl(
    *,
    bias: SetupBias,
    metrics: TechnicalMetrics,
    structure: StructureAnalysis,
    smc: SmcSummary,
    last: float,
    atr: float,
) -> tuple[float, float, str, float, str] | None:
    if bias is SetupBias.LONG:
        entry_low, entry_high, entry_source = _select_long_entry(metrics, smc, last, atr)
        entry_mid = (entry_low + entry_high) / 2.0
        swing_lows = [ev.level for ev in structure.last_swing_lows if ev.level < entry_low]
        structural_floor = max(swing_lows) if swing_lows else (entry_low - 1.2 * atr)
        stop_loss = min(
            structural_floor - SL_ATR_BUFFER * atr,
            entry_low - 0.25 * atr,
            entry_mid - MIN_SL_ATR * atr,
        )
        if stop_loss <= 0:
            return None
        return (
            entry_low,
            entry_high,
            entry_source,
            stop_loss,
            "Di bawah Swing Low terkonfirmasi + buffer 0,5x ATR",
        )

    entry_low, entry_high, entry_source = _select_short_entry(metrics, smc, last, atr)
    entry_mid = (entry_low + entry_high) / 2.0
    swing_highs = [ev.level for ev in structure.last_swing_highs if ev.level > entry_high]
    structural_ceil = min(swing_highs) if swing_highs else (entry_high + 1.2 * atr)
    stop_loss = max(
        structural_ceil + SL_ATR_BUFFER * atr,
        entry_high + 0.25 * atr,
        entry_mid + MIN_SL_ATR * atr,
    )
    return (
        entry_low,
        entry_high,
        entry_source,
        stop_loss,
        "Di atas Swing High terkonfirmasi + buffer 0,5x ATR",
    )


def _evaluate_micro_5m_confirmation(
    *,
    bias: SetupBias,
    micro_candles: Sequence[MarketCandle] | None,
    as_of: datetime | None,
    use_available_time: bool,
) -> str | None:
    """Evaluate causal 5m micro-confirmation inside/near setup entry without tightening SL."""
    if not micro_candles or bias not in (SetupBias.LONG, SetupBias.SHORT):
        return None
    filtered = _filter_causal_candles(micro_candles, as_of, use_available_time=use_available_time)
    if len(filtered) < 15 or filtered[0].timeframe is not Timeframe.M5:
        return None

    m5_struct = analyze_market_structure(filtered)
    m5_smc = analyze_smc(filtered, structure=m5_struct)
    m5_brk = m5_smc.latest_structure_break.event_type if m5_smc.latest_structure_break else None
    m5_swp = m5_smc.latest_sweep.event_type if m5_smc.latest_sweep else None

    if bias is SetupBias.LONG:
        if (
            m5_swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH
            and m5_brk is StructureEventType.CHOCH_BULLISH
        ):
            return "5m CHoCH+Sweep searah (micro trigger kuat)"
        if m5_brk in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH):
            return "5m CHoCH/BOS searah terkonfirmasi"
        if m5_struct.market_structure is MarketStructure.BULLISH:
            return "5m struktur searah"
        if (
            m5_brk in (StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH)
            or m5_struct.market_structure is MarketStructure.BEARISH
        ):
            return "5m masih koreksi (tunggu CHoCH 5m di zona entry)"
        return "5m konsolidasi"

    if (
        m5_swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH
        and m5_brk is StructureEventType.CHOCH_BEARISH
    ):
        return "5m CHoCH+Sweep searah (micro trigger kuat)"
    if m5_brk in (StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH):
        return "5m CHoCH/BOS searah terkonfirmasi"
    if m5_struct.market_structure is MarketStructure.BEARISH:
        return "5m struktur searah"
    if (
        m5_brk in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH)
        or m5_struct.market_structure is MarketStructure.BULLISH
    ):
        return "5m masih koreksi (tunggu CHoCH 5m di zona entry)"
    return "5m konsolidasi"


def build_trade_setup(
    *,
    metrics: TechnicalMetrics,
    regime: Regime,
    structure: StructureAnalysis,
    smc: SmcSummary,
    derivatives: DerivativesMetrics | None = None,
    sentiment: SentimentMetrics | None = None,
    primary_timeframe: Timeframe = Timeframe.H4,
    ltf_candles: Sequence[MarketCandle] | None = None,
    micro_candles: Sequence[MarketCandle] | None = None,
    htf_candles_by_tf: Mapping[Timeframe, Sequence[MarketCandle]] | None = None,
    as_of: datetime | None = None,
    use_available_time: bool = False,
    cost_assumptions: CostAssumptions | None = None,
    macro_context: MacroBenchmarkContext | None = None,
) -> TradeSetup:
    """Compute 1 deterministic conditional trade setup (Entry, SL, TP1-3) or honest WAIT."""
    htf_sup, htf_res = _extract_sr_levels(structure, smc)
    ltf_sup: float | None = None
    ltf_res: float | None = None
    if primary_timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.D1) and ltf_candles:
        filtered_15m = _filter_causal_candles(
            ltf_candles,
            as_of,
            use_available_time=use_available_time,
        )
        if len(filtered_15m) >= 15 and filtered_15m[0].timeframe is Timeframe.M15:
            st_15m = analyze_market_structure(filtered_15m)
            smc_15m = analyze_smc(filtered_15m, structure=st_15m)
            ltf_sup, ltf_res = _extract_sr_levels(st_15m, smc_15m)

    atr = metrics.atr_14
    ema_50 = metrics.ema_50
    if atr is None or atr <= 0 or ema_50 is None:
        return _wait_setup(
            "Metrik ATR 14 atau EMA 50 belum tersedia untuk menghitung zona entry dan risiko.",
            entry_timeframe=primary_timeframe,
            htf_support=htf_sup,
            htf_resistance=htf_res,
            ltf_support=ltf_sup,
            ltf_resistance=ltf_res,
        )

    bias, wait_reason = _resolve_bias(
        metrics=metrics,
        regime=regime,
        structure=structure,
        smc=smc,
    )
    if bias is SetupBias.NEUTRAL or wait_reason is not None:
        return _wait_setup(
            wait_reason or "Kondisi pasar belum memiliki bias dominan.",
            entry_timeframe=primary_timeframe,
            htf_support=htf_sup,
            htf_resistance=htf_res,
            ltf_support=ltf_sup,
            ltf_resistance=ltf_res,
        )

    last = metrics.last_close
    risk_notes = _collect_risk_notes(
        bias=bias,
        smc=smc,
        derivatives=derivatives,
        sentiment=sentiment,
    )
    if macro_context is not None:
        if macro_context.dxy is not None and macro_context.dxy.macro_impact == "headwind":
            risk_notes.append(
                "Makro Headwind: Indeks Dolar AS (DXY) menguat dengan korelasi negatif."
            )
        if (
            macro_context.btc_benchmark is not None
            and macro_context.btc_benchmark.leader_alignment == "divergent_warning"
            and bias is SetupBias.LONG
        ):
            risk_notes.append(
                "Konflik Leader: Setup LONG pada altcoin bertentangan dengan "
                "tren bearish BTC benchmark."
            )
        if macro_context.positioning_regime == PositioningRegime.CROWDED_SQUEEZE_RISK:
            risk_notes.append(
                "Rezim Posisi: Risiko crowded squeeze terdeteksi pada pasar futures "
                "(funding/posisi ekstrem)."
            )

    # Filter 1: Overbought/Oversold Exhaustion at Dealing Range Extremes
    if (
        bias is SetupBias.LONG
        and metrics.rsi_14 is not None
        and metrics.rsi_14 >= 75.0
        and smc.dealing_zone is DealingZone.PREMIUM
    ):
        return _wait_setup(
            f"RSI 14 jenuh beli ekstrem ({metrics.rsi_14:.1f} >= 75) di zona Premium dealing "
            "range; waspada exhaustion pucuk, tunggu retracement ke zona lebih murah (WAIT).",
            risk_notes,
            entry_timeframe=primary_timeframe,
            htf_support=htf_sup,
            htf_resistance=htf_res,
            ltf_support=ltf_sup,
            ltf_resistance=ltf_res,
        )
    if (
        bias is SetupBias.SHORT
        and metrics.rsi_14 is not None
        and metrics.rsi_14 <= 25.0
        and smc.dealing_zone is DealingZone.DISCOUNT
    ):
        return _wait_setup(
            f"RSI 14 jenuh jual ekstrem ({metrics.rsi_14:.1f} <= 25) di zona Discount "
            "dealing range; waspada exhaustion dasar, tunggu pullback ke zona lebih tinggi (WAIT).",
            risk_notes,
            entry_timeframe=primary_timeframe,
            htf_support=htf_sup,
            htf_resistance=htf_res,
            ltf_support=ltf_sup,
            ltf_resistance=ltf_res,
        )

    # Filter 2 (HTF): Liquidity Sweep counter-trend + Adverse Order Book Imbalance
    htf_swp = smc.latest_sweep.event_type if smc.latest_sweep else None
    if derivatives is not None and derivatives.imbalance_10bps is not None:
        if (
            bias is SetupBias.LONG
            and htf_swp is StructureEventType.LIQUIDITY_SWEEP_BEARISH
            and derivatives.imbalance_10bps <= -0.30
        ):
            imb_val = derivatives.imbalance_10bps
            return _wait_setup(
                "Terdeteksi Liquidity Sweep bearish disertai tekanan order book tebal di sisi "
                f"ask (imbalance 10bps {imb_val:.2f} <= -0,30); waspada distribusi pucuk (WAIT).",
                risk_notes,
                entry_timeframe=primary_timeframe,
                htf_support=htf_sup,
                htf_resistance=htf_res,
                ltf_support=ltf_sup,
                ltf_resistance=ltf_res,
            )
        if (
            bias is SetupBias.SHORT
            and htf_swp is StructureEventType.LIQUIDITY_SWEEP_BULLISH
            and derivatives.imbalance_10bps >= 0.30
        ):
            imb_val = derivatives.imbalance_10bps
            return _wait_setup(
                "Terdeteksi Liquidity Sweep bullish disertai dukungan order book tebal di sisi "
                f"bid (imbalance 10bps +{imb_val:.2f} >= +0,30); waspada absorpsi dasar (WAIT).",
                risk_notes,
                entry_timeframe=primary_timeframe,
                htf_support=htf_sup,
                htf_resistance=htf_res,
                ltf_support=ltf_sup,
                ltf_resistance=ltf_res,
            )

    htf_conf_note: str | None = None
    if primary_timeframe is Timeframe.M15:
        htf_wait, htf_conf_note = _evaluate_htf_filter_for_15m(
            bias=bias,
            smc=smc,
            htf_candles_by_tf=htf_candles_by_tf,
            as_of=as_of,
            use_available_time=use_available_time,
        )
        if htf_wait is not None:
            return _wait_setup(
                htf_wait,
                risk_notes,
                entry_timeframe=Timeframe.M15,
                htf_support=htf_sup,
                htf_resistance=htf_res,
            )

    htf_levels = _build_htf_levels_and_sl(
        bias=bias,
        metrics=metrics,
        structure=structure,
        smc=smc,
        last=last,
        atr=atr,
    )
    if htf_levels is None:
        return _wait_setup(
            "Invalidasi Stop Loss tidak valid secara numerik.",
            risk_notes,
            entry_timeframe=primary_timeframe,
            htf_support=htf_sup,
            htf_resistance=htf_res,
            ltf_support=ltf_sup,
            ltf_resistance=ltf_res,
        )

    entry_low, entry_high, entry_source, stop_loss, invalidation_reason = htf_levels
    entry_tf = primary_timeframe
    ltf_confirmation: str | None = htf_conf_note

    if primary_timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.D1) and ltf_candles:
        ltf_ref = _refine_with_ltf_15m(
            bias=bias,
            last=last,
            htf_atr=atr,
            primary_timeframe=primary_timeframe,
            ltf_candles=ltf_candles,
            as_of=as_of,
            use_available_time=use_available_time,
        )
        if ltf_ref is not None:
            ltf_confirmation = ltf_ref.confirmation
            ltf_sup = ltf_ref.ltf_support or ltf_sup
            ltf_res = ltf_ref.ltf_resistance or ltf_res
            risk_notes.extend(ltf_ref.extra_risk_notes)

            # Filter 2 (LTF): 15m Counter-Liquidity Sweep + Adverse Order Book Imbalance
            if (
                derivatives is not None
                and derivatives.imbalance_10bps is not None
                and ltf_ref.has_counter_sweep
            ):
                if bias is SetupBias.LONG and derivatives.imbalance_10bps <= -0.30:
                    imb_val = derivatives.imbalance_10bps
                    return _wait_setup(
                        "Terdeteksi 15m Liquidity Sweep bearish disertai tekanan order book "
                        f"tebal di sisi ask (imbalance 10bps {imb_val:.2f} <= -0,30); "
                        "waspada jebakan pucuk (WAIT).",
                        risk_notes,
                        entry_timeframe=primary_timeframe,
                        ltf_confirmation=ltf_ref.confirmation,
                        htf_support=htf_sup,
                        htf_resistance=htf_res,
                        ltf_support=ltf_sup,
                        ltf_resistance=ltf_res,
                    )
                if bias is SetupBias.SHORT and derivatives.imbalance_10bps >= 0.30:
                    imb_val = derivatives.imbalance_10bps
                    return _wait_setup(
                        "Terdeteksi 15m Liquidity Sweep bullish disertai dukungan order book "
                        f"tebal di sisi bid (imbalance 10bps +{imb_val:.2f} >= +0,30); "
                        "waspada absorpsi dasar (WAIT).",
                        risk_notes,
                        entry_timeframe=primary_timeframe,
                        ltf_confirmation=ltf_ref.confirmation,
                        htf_support=htf_sup,
                        htf_resistance=htf_res,
                        ltf_support=ltf_sup,
                        ltf_resistance=ltf_res,
                    )
            if (
                ltf_ref.entry_low is not None
                and ltf_ref.entry_high is not None
                and ltf_ref.entry_source is not None
                and ltf_ref.stop_loss is not None
                and ltf_ref.invalidation_reason is not None
                and ltf_ref.stop_loss > 0
            ):
                cand_mid = (ltf_ref.entry_low + ltf_ref.entry_high) / 2.0
                cand_risk = abs(cand_mid - ltf_ref.stop_loss)
                if 0 < cand_risk <= MAX_SL_ATR * atr:
                    entry_low = ltf_ref.entry_low
                    entry_high = ltf_ref.entry_high
                    entry_source = ltf_ref.entry_source
                    stop_loss = ltf_ref.stop_loss
                    invalidation_reason = ltf_ref.invalidation_reason
                    entry_tf = Timeframe.M15

    micro_note = _evaluate_micro_5m_confirmation(
        bias=bias,
        micro_candles=micro_candles,
        as_of=as_of,
        use_available_time=use_available_time,
    )
    if micro_note is not None:
        ltf_confirmation = f"{ltf_confirmation} | {micro_note}" if ltf_confirmation else micro_note

    entry_mid = (entry_low + entry_high) / 2.0
    if bias is SetupBias.LONG:
        risk = entry_mid - stop_loss
        if risk > MAX_SL_ATR * atr:
            return _wait_setup(
                "Jarak Stop Loss ke swing invalidasi terlalu lebar (> 3,5x ATR); tunggu koreksi "
                "atau struktur lebih rapat.",
                risk_notes,
                entry_timeframe=entry_tf,
                ltf_confirmation=ltf_confirmation,
                htf_support=htf_sup,
                htf_resistance=htf_res,
                ltf_support=ltf_sup,
                ltf_resistance=ltf_res,
            )

        tp1, tp1_label, tp2, tp2_label, tp3, tp3_label = _select_long_targets(
            entry_mid=entry_mid,
            risk=risk,
            metrics=metrics,
            structure=structure,
            smc=smc,
            ltf_resistance=ltf_res,
        )
        rr_tp1 = (tp1 - entry_mid) / risk
        rr_tp2 = (tp2 - entry_mid) / risk
        rr_tp3 = (tp3 - entry_mid) / risk

        if (
            regime in (Regime.RANGING, Regime.TRANSITIONAL)
            and smc.dealing_range_high is not None
            and smc.dealing_range_high > entry_mid
            and (smc.dealing_range_high - entry_mid) / risk < MIN_RR_TP2
        ):
            return _wait_setup(
                "Rasio Risk:Reward ke batas swing high terdekat di bawah 1:1,5 (WAIT).",
                risk_notes,
                entry_timeframe=entry_tf,
                ltf_confirmation=ltf_confirmation,
                htf_support=htf_sup,
                htf_resistance=htf_res,
                ltf_support=ltf_sup,
                ltf_resistance=ltf_res,
            )
    else:
        risk = stop_loss - entry_mid
        if risk > MAX_SL_ATR * atr:
            return _wait_setup(
                "Jarak Stop Loss ke swing invalidasi terlalu lebar (> 3,5x ATR); tunggu reli "
                "atau struktur lebih rapat.",
                risk_notes,
                entry_timeframe=entry_tf,
                ltf_confirmation=ltf_confirmation,
                htf_support=htf_sup,
                htf_resistance=htf_res,
                ltf_support=ltf_sup,
                ltf_resistance=ltf_res,
            )

        tp1, tp1_label, tp2, tp2_label, tp3, tp3_label = _select_short_targets(
            entry_mid=entry_mid,
            risk=risk,
            metrics=metrics,
            structure=structure,
            smc=smc,
            ltf_support=ltf_sup,
        )
        rr_tp1 = (entry_mid - tp1) / risk
        rr_tp2 = (entry_mid - tp2) / risk
        rr_tp3 = (entry_mid - tp3) / risk

        if (
            regime in (Regime.RANGING, Regime.TRANSITIONAL)
            and smc.dealing_range_low is not None
            and smc.dealing_range_low < entry_mid
            and (entry_mid - smc.dealing_range_low) / risk < MIN_RR_TP2
        ):
            return _wait_setup(
                "Rasio Risk:Reward ke batas swing low terdekat di bawah 1:1,5 (WAIT).",
                risk_notes,
                entry_timeframe=entry_tf,
                ltf_confirmation=ltf_confirmation,
                htf_support=htf_sup,
                htf_resistance=htf_res,
                ltf_support=ltf_sup,
                ltf_resistance=ltf_res,
            )

    if rr_tp2 < MIN_RR_TP2:
        return _wait_setup(
            "Rasio Risk:Reward ke TP 2 di bawah batas minimum 1:1,5 (WAIT).",
            risk_notes,
            entry_timeframe=entry_tf,
            ltf_confirmation=ltf_confirmation,
            htf_support=htf_sup,
            htf_resistance=htf_res,
            ltf_support=ltf_sup,
            ltf_resistance=ltf_res,
        )

    costs = cost_assumptions or CostAssumptions()
    cost_rate = costs.round_trip_cost_rate
    net_rr_tp1 = _net_reward_risk(
        entry_mid=entry_mid, target=tp1, stop_loss=stop_loss, cost_rate=cost_rate
    )
    net_rr_tp2 = _net_reward_risk(
        entry_mid=entry_mid, target=tp2, stop_loss=stop_loss, cost_rate=cost_rate
    )
    net_rr_tp3 = _net_reward_risk(
        entry_mid=entry_mid, target=tp3, stop_loss=stop_loss, cost_rate=cost_rate
    )
    if net_rr_tp2 is not None and net_rr_tp2 < MIN_RR_TP2:
        risk_notes.append(
            "Rasio Risk:Reward ke TP 2 setelah biaya (fee + spread + slippage) turun di bawah "
            "1:1,5; perkecil ukuran posisi atau tunggu zona entry lebih presisi."
        )

    sl_dist_ratio = abs(entry_mid - stop_loss) / entry_mid
    sl_dist_pct = round(sl_dist_ratio * 100.0, 4)
    max_theor_lev = max(1, int(0.8 / sl_dist_ratio)) if sl_dist_ratio > 0 else 1

    return TradeSetup(
        status=SetupStatus.READY,
        bias=bias,
        entry_low=entry_low,
        entry_high=entry_high,
        entry_mid=entry_mid,
        entry_source=entry_source,
        entry_timeframe=entry_tf,
        ltf_confirmation=ltf_confirmation,
        htf_support=htf_sup,
        htf_resistance=htf_res,
        ltf_support=ltf_sup,
        ltf_resistance=ltf_res,
        stop_loss=stop_loss,
        invalidation_reason=invalidation_reason,
        stop_loss_distance_pct=sl_dist_pct,
        max_theoretical_leverage=max_theor_lev,
        tp1=tp1,
        tp1_label=tp1_label,
        tp2=tp2,
        tp2_label=tp2_label,
        tp3=tp3,
        tp3_label=tp3_label,
        rr_tp1=rr_tp1,
        rr_tp2=rr_tp2,
        rr_tp3=rr_tp3,
        net_rr_tp1=net_rr_tp1,
        net_rr_tp2=net_rr_tp2,
        net_rr_tp3=net_rr_tp3,
        cost_rate_round_trip=cost_rate,
        risk_notes=tuple(dict.fromkeys(risk_notes)),
    )


def _select_long_entry(
    metrics: TechnicalMetrics,
    smc: SmcSummary,
    last: float,
    atr: float,
) -> tuple[float, float, str]:
    fvg = smc.active_bullish_fvg
    if fvg is not None and fvg.zone_low > 0 and (last - fvg.zone_high) <= 2.5 * atr:
        high = max(fvg.zone_low, min(fvg.zone_high, last))
        return fvg.zone_low, high, "Pullback ke Bullish FVG"

    ob = smc.active_bullish_ob
    if ob is not None and ob.zone_low > 0 and (last - ob.zone_high) <= 2.5 * atr:
        high = max(ob.zone_low, min(ob.zone_high, last))
        return ob.zone_low, high, "Pullback ke Bullish Order Block"

    ema20 = metrics.ema_20
    if ema20 is not None and ema20 < last and (last - ema20) <= 2.0 * atr:
        low = max(1e-8, ema20 - 0.15 * atr)
        high = max(low, min(ema20 + 0.15 * atr, last))
        return low, high, "Pullback ke EMA 20"

    low = max(1e-8, last - 0.6 * atr)
    high = max(low, last - 0.15 * atr)
    return low, high, "Zona pullback berbasis ATR"


def _select_short_entry(
    metrics: TechnicalMetrics,
    smc: SmcSummary,
    last: float,
    atr: float,
) -> tuple[float, float, str]:
    fvg = smc.active_bearish_fvg
    if fvg is not None and (fvg.zone_low - last) <= 2.5 * atr:
        low = min(fvg.zone_high, max(fvg.zone_low, last))
        return low, fvg.zone_high, "Reli ke Bearish FVG"

    ob = smc.active_bearish_ob
    if ob is not None and (ob.zone_low - last) <= 2.5 * atr:
        low = min(ob.zone_high, max(ob.zone_low, last))
        return low, ob.zone_high, "Reli ke Bearish Order Block"

    ema20 = metrics.ema_20
    if ema20 is not None and ema20 > last and (ema20 - last) <= 2.0 * atr:
        low = max(last, ema20 - 0.15 * atr)
        high = max(low, ema20 + 0.15 * atr)
        return low, high, "Reli ke EMA 20"

    low = last + 0.15 * atr
    high = last + 0.6 * atr
    return low, high, "Zona reli pullback berbasis ATR"


def _select_long_targets(
    *,
    entry_mid: float,
    risk: float,
    metrics: TechnicalMetrics,
    structure: StructureAnalysis,
    smc: SmcSummary,
    ltf_resistance: float | None = None,
) -> tuple[float, str, float, str, float, str]:
    raw_candidates: list[tuple[float, str]] = []
    if ltf_resistance is not None:
        raw_candidates.append((ltf_resistance, "Resistance 15m"))
    for ev in structure.last_swing_highs:
        raw_candidates.append((ev.level, "Swing High"))
    if smc.dealing_range_high is not None:
        raw_candidates.append((smc.dealing_range_high, "Range High"))
    if metrics.prev_day_high is not None:
        raw_candidates.append((metrics.prev_day_high, "Prev Day High"))
    if metrics.prev_week_high is not None:
        raw_candidates.append((metrics.prev_week_high, "Prev Week High"))

    candidates = sorted(
        ((lvl, lbl) for lvl, lbl in raw_candidates if lvl > entry_mid + 0.6 * risk),
        key=lambda item: item[0],
    )

    tp1_match = next(
        ((lvl, lbl) for lvl, lbl in candidates if 0.8 * risk <= (lvl - entry_mid) <= 1.5 * risk),
        None,
    )
    if tp1_match is not None:
        tp1, tp1_label = tp1_match
    else:
        tp1 = entry_mid + DEFAULT_TP1_R * risk
        tp1_label = "Ekspansi 1,0R"

    tp2_match = next(
        (
            (lvl, lbl)
            for lvl, lbl in candidates
            if lvl >= max(tp1 + 0.3 * risk, entry_mid + MIN_RR_TP2 * risk)
            and (lvl - entry_mid) <= 3.0 * risk
        ),
        None,
    )
    if tp2_match is not None:
        tp2, tp2_label = tp2_match
    else:
        tp2 = max(tp1 + 0.8 * risk, entry_mid + DEFAULT_TP2_R * risk)
        tp2_label = "Ekspansi 2,0R"

    tp3_match = next(
        (
            (lvl, lbl)
            for lvl, lbl in candidates
            if lvl >= tp2 + 0.5 * risk and (lvl - entry_mid) <= 5.0 * risk
        ),
        None,
    )
    if tp3_match is not None:
        tp3, tp3_label = tp3_match
    else:
        tp3 = max(tp2 + risk, entry_mid + DEFAULT_TP3_R * risk)
        tp3_label = "Ekspansi 3,0R"

    return tp1, tp1_label, tp2, tp2_label, tp3, tp3_label


def _select_short_targets(
    *,
    entry_mid: float,
    risk: float,
    metrics: TechnicalMetrics,
    structure: StructureAnalysis,
    smc: SmcSummary,
    ltf_support: float | None = None,
) -> tuple[float, str, float, str, float, str]:
    raw_candidates: list[tuple[float, str]] = []
    if ltf_support is not None:
        raw_candidates.append((ltf_support, "Support 15m"))
    for ev in structure.last_swing_lows:
        raw_candidates.append((ev.level, "Swing Low"))
    if smc.dealing_range_low is not None:
        raw_candidates.append((smc.dealing_range_low, "Range Low"))
    if metrics.prev_day_low is not None:
        raw_candidates.append((metrics.prev_day_low, "Prev Day Low"))
    if metrics.prev_week_low is not None:
        raw_candidates.append((metrics.prev_week_low, "Prev Week Low"))

    candidates = sorted(
        ((lvl, lbl) for lvl, lbl in raw_candidates if 0 < lvl < entry_mid - 0.6 * risk),
        key=lambda item: item[0],
        reverse=True,
    )

    tp1_match = next(
        ((lvl, lbl) for lvl, lbl in candidates if 0.8 * risk <= (entry_mid - lvl) <= 1.5 * risk),
        None,
    )
    if tp1_match is not None:
        tp1, tp1_label = tp1_match
    else:
        tp1 = max(1e-8, entry_mid - DEFAULT_TP1_R * risk)
        tp1_label = "Ekspansi 1,0R"

    tp2_match = next(
        (
            (lvl, lbl)
            for lvl, lbl in candidates
            if lvl <= min(tp1 - 0.3 * risk, entry_mid - MIN_RR_TP2 * risk)
            and (entry_mid - lvl) <= 3.0 * risk
        ),
        None,
    )
    if tp2_match is not None:
        tp2, tp2_label = tp2_match
    else:
        tp2 = max(1e-8, min(tp1 - 0.8 * risk, entry_mid - DEFAULT_TP2_R * risk))
        tp2_label = "Ekspansi 2,0R"

    tp3_match = next(
        (
            (lvl, lbl)
            for lvl, lbl in candidates
            if lvl <= tp2 - 0.5 * risk and (entry_mid - lvl) <= 5.0 * risk
        ),
        None,
    )
    if tp3_match is not None:
        tp3, tp3_label = tp3_match
    else:
        tp3 = max(1e-8, min(tp2 - risk, entry_mid - DEFAULT_TP3_R * risk))
        tp3_label = "Ekspansi 3,0R"

    return tp1, tp1_label, tp2, tp2_label, tp3, tp3_label


def evaluate_setup_against_candles(
    setup: TradeSetupRecord,
    future_candles: Sequence[MarketCandle],
    *,
    max_bars: int = 24,
) -> tuple[str, datetime | None, float | None]:
    """Deterministically settle a logged setup against subsequent closed candles.

    Conservative tie-break: if any candle after/at trigger touches both SL and TP,
    SL takes precedence (`hit_sl`, `-1.0R`).
    """
    if setup.status != SetupStatus.READY.value:
        return "skipped_wait", None, None

    if (
        setup.entry_low is None
        or setup.entry_high is None
        or setup.entry_mid is None
        or setup.stop_loss is None
        or setup.tp1 is None
        or setup.tp2 is None
        or setup.tp3 is None
        or setup.bias not in (SetupBias.LONG.value, SetupBias.SHORT.value)
    ):
        return "skipped_wait", None, None

    candles = sorted(
        (c for c in future_candles if c.is_closed and c.open_time >= setup.as_of),
        key=lambda c: c.open_time,
    )[:max_bars]
    if not candles:
        return "pending", None, None

    risk = abs(setup.entry_mid - setup.stop_loss)
    if risk <= 0:
        return "skipped_wait", None, None

    is_long = setup.bias == SetupBias.LONG.value
    triggered = False
    highest_tp = 0
    tp_hit_times: dict[int, datetime] = {}

    for candle in candles:
        if not triggered:
            if candle.low <= setup.entry_high and candle.high >= setup.entry_low:
                triggered = True
            else:
                continue

        if is_long:
            if candle.low <= setup.stop_loss:
                return "hit_sl", candle.close_time, -1.0
            if candle.high >= setup.tp3:
                return "hit_tp3", candle.close_time, setup.rr_tp3
            if candle.high >= setup.tp2 and highest_tp < 2:
                highest_tp = 2
                tp_hit_times[2] = candle.close_time
            if candle.high >= setup.tp1 and highest_tp < 1:
                highest_tp = 1
                tp_hit_times[1] = candle.close_time
        else:
            if candle.high >= setup.stop_loss:
                return "hit_sl", candle.close_time, -1.0
            if candle.low <= setup.tp3:
                return "hit_tp3", candle.close_time, setup.rr_tp3
            if candle.low <= setup.tp2 and highest_tp < 2:
                highest_tp = 2
                tp_hit_times[2] = candle.close_time
            if candle.low <= setup.tp1 and highest_tp < 1:
                highest_tp = 1
                tp_hit_times[1] = candle.close_time

    if highest_tp == 2:
        return "hit_tp2", tp_hit_times[2], setup.rr_tp2
    if highest_tp == 1:
        return "hit_tp1", tp_hit_times[1], setup.rr_tp1

    if len(candles) >= max_bars:
        if not triggered:
            return "expired", candles[-1].close_time, 0.0
        last_close = candles[-1].close
        realized_r = (
            (last_close - setup.entry_mid) / risk
            if is_long
            else (setup.entry_mid - last_close) / risk
        )
        return "expired", candles[-1].close_time, realized_r

    if triggered:
        return "entry_triggered", None, None

    return "pending", None, None
