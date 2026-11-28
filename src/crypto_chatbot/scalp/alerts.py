"""SC-08 & SC-13: Scalp v1 Telegram alert renderer, position size calculator, and daily digest.

Also provides `make_scalp_run_id` and `evaluate_scalp_symbol` so `alerts run` routes cleanly
when `strategy == "scalp_v1"` while keeping swing MTF untouched.
"""

from __future__ import annotations

import hashlib
import html
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.alerts.config import WIB
from crypto_chatbot.bot.templates import format_timestamp_utc_wib
from crypto_chatbot.scalp.config import MacroEventWindow, ScalpConfig, ScalpPositionConfig
from crypto_chatbot.scalp.features import ScalpFeatureSnapshot, extract_scalp_features_pit
from crypto_chatbot.scalp.filters import ScalpFilterInputs, evaluate_scalp_filters
from crypto_chatbot.scalp.mode import ScalpMode, select_scalp_mode_from_snapshot
from crypto_chatbot.scalp.settlement import ScalpOutcomeStatus, ScalpSettlementOutcome
from crypto_chatbot.scalp.setups import (
    ScalpSetupResult,
    detect_valid_dealing_range,
    evaluate_range_fade_setup,
    evaluate_trend_pullback_setup,
)
from crypto_chatbot.schemas.analysis import AnalysisSnapshot, MarketStructure
from crypto_chatbot.schemas.derivatives import (
    ContractStatsObservation,
    FundingRateObservation,
    OrderBookSnapshot,
)
from crypto_chatbot.schemas.market import MarketCandle
from crypto_chatbot.schemas.news import FearGreedObservation, NormalizedNewsArticle
from crypto_chatbot.storage.repositories import ScalpDecisionRecord, TradeSetupRecord


def make_scalp_run_id(candle_close_time: datetime) -> str:
    """Deterministic idempotent run_id for 5m scalp cadence (`scalp:<5m close_time>`, §2)."""
    close_utc = candle_close_time.astimezone(UTC)
    return f"scalp:{close_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def make_scalp_decision_id(run_id: str, canonical_symbol: str) -> str:
    digest = hashlib.sha256(f"{run_id}|{canonical_symbol}".encode()).hexdigest()[:16]
    return f"scdec-{digest}"


class PositionSizeEstimate(BaseModel):
    """Position size calculated from owner's own config (§8 — calculator, not advice)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_asset: str
    qty_base: float = Field(gt=0.0)
    notional_usd: float = Field(gt=0.0)
    risk_usd: float = Field(gt=0.0)
    risk_per_trade_pct: float = Field(gt=0.0)


def compute_position_size(
    *,
    canonical_symbol: str,
    entry_price: float,
    stop_loss: float,
    position_config: ScalpPositionConfig,
) -> PositionSizeEstimate | None:
    """Compute position size from `.env` `equity_usd` & `risk_per_trade_pct`, or None if unset."""
    if position_config.equity_usd is None or position_config.risk_per_trade_pct is None:
        return None
    stop_dist = abs(entry_price - stop_loss)
    if entry_price <= 0.0 or stop_dist <= 0.0:
        raise ValueError("entry_price must be > 0 and stop_loss must differ from entry_price")
    risk_usd = position_config.equity_usd * (position_config.risk_per_trade_pct / 100.0)
    qty_base = risk_usd / stop_dist
    notional_usd = qty_base * entry_price
    base_asset = canonical_symbol.split("/", 1)[0].upper()
    return PositionSizeEstimate(
        base_asset=base_asset,
        qty_base=qty_base,
        notional_usd=notional_usd,
        risk_usd=risk_usd,
        risk_per_trade_pct=position_config.risk_per_trade_pct,
    )


def _fmt_px(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 1000.0:
        return f"{value:,.2f}"
    if abs_v >= 1.0:
        return f"{value:.2f}"
    if abs_v >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


def build_scalp_inline_keyboard(alert_id: str) -> dict[str, Any]:
    """SC-08 inline keyboard: [Pasang] [Skip] [Mute 1j] [Detail]."""
    return {
        "inline_keyboard": [
            [
                {"text": "Pasang", "callback_data": f"al:ps:{alert_id}"},
                {"text": "Skip", "callback_data": f"al:sk:{alert_id}"},
                {"text": "Mute 1j", "callback_data": f"al:m1:{alert_id}"},
                {"text": "Detail", "callback_data": f"al:dt:{alert_id}"},
            ]
        ]
    }


def render_scalp_alert_message(
    *,
    canonical_symbol: str,
    setup: ScalpSetupResult,
    regime_1h_label: str,
    structure_15m_label: str,
    features: ScalpFeatureSnapshot,
    position_config: ScalpPositionConfig,
    phase: str = "shadow",
    snapshot_id: str = "",
    params_sha256: str = "",
) -> str:
    """Render deterministic SC-08 Telegram HTML scalp alert message."""
    if (
        setup.direction is None
        or setup.entry_low is None
        or setup.entry_high is None
        or setup.entry_limit is None
        or setup.stop_loss is None
        or setup.tp1 is None
        or setup.tp2 is None
        or setup.risk_bps is None
        or setup.cost_breakdown is None
    ):
        raise ValueError("Cannot render scalp alert for non-READY setup")

    base = canonical_symbol.split("/", 1)[0].upper()
    is_long = setup.direction == "long"
    emoji = "🟢" if is_long else "🔴"
    dir_lbl = "LONG" if is_long else "SHORT"
    order_lbl = "LIMIT BUY" if is_long else "LIMIT SELL"
    mode_lbl = (
        "Trend Pullback"
        if setup.mode in (ScalpMode.TREND_PULLBACK_LONG, ScalpMode.TREND_PULLBACK_SHORT)
        else "Range Fade"
    )
    phase_tag = " [SHADOW]" if phase == "shadow" else ""

    valid_wib = (
        setup.entry_valid_until_utc.astimezone(WIB).strftime("%H:%M WIB")
        if setup.entry_valid_until_utc is not None
        else "-"
    )
    cost = setup.cost_breakdown
    sl_sign = "-" if is_long else "+"
    tp1_r = setup.gross_rr_tp1 or 1.0
    tp2_r = setup.gross_rr_tp2 or 1.8

    size = compute_position_size(
        canonical_symbol=canonical_symbol,
        entry_price=setup.entry_limit,
        stop_loss=setup.stop_loss,
        position_config=position_config,
    )
    size_line = (
        f"Ukuran (risk {size.risk_per_trade_pct:g}% dari modal di config): "
        f"≈ {size.qty_base:.2f} {html.escape(size.base_asset)} "
        f"≈ {size.notional_usd:.0f} USDT notional"
        if size is not None
        else "Ukuran: size: not configured"
    )

    taker_str = (
        f"{features.taker_buy_sell_ratio_5m_3bar:.2f}"
        if features.taker_buy_sell_ratio_5m_3bar is not None
        else "n/a"
    )
    funding_str = (
        f"{features.funding_rate_8h * 100:+.3f}%" if features.funding_rate_8h is not None else "n/a"
    )

    header_base = html.escape(base)
    header_mode = html.escape(mode_lbl)
    header_tag = html.escape(phase_tag)
    entry_rng = f"{_fmt_px(setup.entry_low)} - {_fmt_px(setup.entry_high)}"
    lines = [
        f"{emoji} <b>SCALP {dir_lbl} · {header_base} · {header_mode}</b>{header_tag}",
        f"<code>{order_lbl}  {entry_rng}</code>   (berlaku s/d {html.escape(valid_wib)})",
        f"SL   {_fmt_px(setup.stop_loss)}  ({sl_sign}{setup.risk_bps:.0f} bps)      "
        f"Biaya: {cost.expected_round_trip_bps:.1f} bps ({cost.cost_pct_of_r * 100:.0f}% dari 1R)",
        f"TP1  {_fmt_px(setup.tp1)}  ({tp1_r:.1f}R, 60%)    "
        f"TP2  {_fmt_px(setup.tp2)}  ({tp2_r:.1f}R, 40%)",
        f"Time stop: {setup.max_hold_minutes} menit setelah terisi",
        size_line,
        f"Konteks: 1h {html.escape(regime_1h_label)} · 15m {html.escape(structure_15m_label)} "
        f"· taker {html.escape(taker_str)} · funding {html.escape(funding_str)}",
    ]
    if snapshot_id or params_sha256:
        lines.append(
            f"<i>Ledger: {html.escape(snapshot_id)} | params={html.escape(params_sha256[:12])}</i>"
        )
    return "\n".join(lines)


def render_scalp_settlement_reply(
    *,
    canonical_symbol: str,
    snapshot_id: str,
    outcome: ScalpSettlementOutcome,
) -> str:
    """SC-08 thread reply from 1m settlement (filled/TP1/BE/TP2/SL/time stop/expired)."""
    labels = {
        ScalpOutcomeStatus.OPEN_FILLED: "TERISI (FILLED)",
        ScalpOutcomeStatus.TP1_THEN_BE: "TP1 → BREAK-EVEN (BE)",
        ScalpOutcomeStatus.TP1_THEN_TIME_STOP: "TP1 → TIME STOP",
        ScalpOutcomeStatus.TP2_HIT: "TP2 TERCAPAI",
        ScalpOutcomeStatus.SL_HIT: "STOP LOSS (SL)",
        ScalpOutcomeStatus.TIME_STOP: "TIME STOP (90m)",
        ScalpOutcomeStatus.EXPIRED_UNFILLED: "EXPIRED — TIDAK TERISI",
        ScalpOutcomeStatus.CANCELLED_TP1_BEFORE_FILL: "BATAL — TP1 TERSENTUH SEBELUM TERISI",
    }
    status_lbl = labels.get(outcome.status, outcome.status.value.upper())
    ts_part = (
        f"\nWaktu: {html.escape(format_timestamp_utc_wib(outcome.exit_time_utc))}"
        if outcome.exit_time_utc is not None
        else ""
    )
    r_part = (
        f" | Net R: {outcome.net_r:+.2f}R (biaya {outcome.cost_r:.2f}R)" if outcome.filled else ""
    )
    return (
        f"<b>Update Scalp · {html.escape(canonical_symbol)}</b>\n"
        f"Status: <b>{html.escape(status_lbl)}</b>{html.escape(r_part)}{ts_part}\n"
        f"<i>Ledger: {html.escape(snapshot_id)}</i>"
    )


class ScalpSymbolEvaluation(BaseModel):
    """Combined decision + setup + features output for one symbol in a 5m scalp run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: ScalpDecisionRecord
    setup: ScalpSetupResult | None = None
    features: ScalpFeatureSnapshot
    regime_1h_label: str = "UNKNOWN"
    structure_15m_label: str = "unknown"


def evaluate_scalp_symbol(
    *,
    run_id: str,
    canonical_symbol: str,
    venue: str,
    decision_time_utc: datetime,
    snapshot_1h: AnalysisSnapshot,
    candles_1h: Sequence[MarketCandle],
    candles_15m: Sequence[MarketCandle],
    candles_5m: Sequence[MarketCandle],
    structure_15m: MarketStructure | None,
    config: ScalpConfig,
    params_sha256: str | None = None,
    orderbook: OrderBookSnapshot | None = None,
    rolling_median_spread_bps: float | None = None,
    contract_stats_5m: Sequence[ContractStatsObservation] = (),
    funding_observations: Sequence[FundingRateObservation] = (),
    news_articles: Sequence[NormalizedNewsArticle] = (),
    fear_greed_observations: Sequence[FearGreedObservation] = (),
    macro_events: Sequence[MacroEventWindow] = (),
    liq_p99_30d_usd: float | None = None,
    news_p99_30d_threshold: float | None = None,
    realized_r_today_utc: float = 0.0,
    last_sl_time_for_symbol_utc: datetime | None = None,
    open_taken_trades_count: int = 0,
) -> ScalpSymbolEvaluation:
    """Run the full `scalp_v1` decision pipeline for one symbol and emit a PIT decision."""
    ts = decision_time_utc.astimezone(UTC)
    features = extract_scalp_features_pit(
        decision_time_utc=ts,
        canonical_symbol=canonical_symbol,
        contract_stats_5m=contract_stats_5m,
        funding_observations=funding_observations,
        news_articles=news_articles,
        fear_greed_observations=fear_greed_observations,
    )

    atr_1h = snapshot_1h.technical.atr_14 if snapshot_1h.technical is not None else None
    est_rt_bps = (
        config.costs.maker_fee_bps
        + config.costs.taker_fee_bps
        + (rolling_median_spread_bps or config.costs.fallback_spread_bps) / 2.0
        + config.costs.fallback_taker_slippage_bps
    )
    dealing_range = detect_valid_dealing_range(
        candles_1h=candles_1h,
        candles_15m=candles_15m,
        atr_1h=atr_1h,
        expected_round_trip_cost_bps=est_rt_bps,
        config=config,
    )

    mode_res = select_scalp_mode_from_snapshot(
        snapshot_1h,
        structure_15m=structure_15m,
        has_valid_range=dealing_range is not None,
        range_low=dealing_range.range_low if dealing_range else None,
        range_high=dealing_range.range_high if dealing_range else None,
        config=config,
    )

    regime_lbl = snapshot_1h.regime.value.upper() if snapshot_1h.regime is not None else "UNKNOWN"
    struct_lbl = structure_15m.value if structure_15m is not None else "unknown"
    decision_id = make_scalp_decision_id(run_id, canonical_symbol)

    # Extract recent 5m liquidation totals for cascade filter
    recent_liq = tuple(
        max(o.long_liq_usd or 0.0, o.short_liq_usd or 0.0)
        for o in sorted(
            (o for o in contract_stats_5m if o.available_time.astimezone(UTC) <= ts),
            key=lambda o: o.observed_time,
        )[-config.filters.liq_cascade_block_bars :]
    )

    if mode_res.mode is ScalpMode.NO_TRADE:
        rec = ScalpDecisionRecord(
            decision_id=decision_id,
            run_id=run_id,
            strategy="scalp_v1",
            venue=venue,
            canonical_symbol=canonical_symbol,
            decision_time=ts,
            mode=mode_res.mode.value,
            decision_status="NO_TRADE",
            direction=None,
            reason_codes=(mode_res.reason_code,),
            mode_inputs_json=mode_res.inputs.model_dump(mode="json"),
            features_json=features.model_dump(mode="json"),
            params_sha256=params_sha256,
            created_at=ts,
        )
        return ScalpSymbolEvaluation(
            decision=rec,
            setup=None,
            features=features,
            regime_1h_label=regime_lbl,
            structure_15m_label=struct_lbl,
        )

    # Evaluate setup for active mode
    if mode_res.mode in (ScalpMode.TREND_PULLBACK_LONG, ScalpMode.TREND_PULLBACK_SHORT):
        setup = evaluate_trend_pullback_setup(
            mode=mode_res.mode,
            canonical_symbol=canonical_symbol,
            candles_15m=candles_15m,
            candles_5m=candles_5m,
            decision_time_utc=ts,
            config=config,
            orderbook=orderbook,
            rolling_median_spread_bps=rolling_median_spread_bps,
            funding_rate_8h=features.funding_rate_8h,
        )
    elif mode_res.mode is ScalpMode.RANGE_FADE and dealing_range is not None:
        setup = evaluate_range_fade_setup(
            canonical_symbol=canonical_symbol,
            dealing_range=dealing_range,
            candles_5m=candles_5m,
            decision_time_utc=ts,
            config=config,
            orderbook=orderbook,
            rolling_median_spread_bps=rolling_median_spread_bps,
            funding_rate_8h=features.funding_rate_8h,
        )
    else:
        setup = None

    # Check §6 blocking filters (direction-aware when setup provides direction)
    flt = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=ts,
            canonical_symbol=canonical_symbol,
            direction=setup.direction if setup is not None else None,
            macro_events=tuple(macro_events),
            recent_5m_liq_usd=recent_liq,
            liq_p99_30d_usd=liq_p99_30d_usd,
            funding_rate_8h=features.funding_rate_8h,
            news_count_30m=features.news_count_30m,
            news_p99_30d_threshold=news_p99_30d_threshold,
            realized_r_today_utc=realized_r_today_utc,
            last_sl_time_for_symbol_utc=last_sl_time_for_symbol_utc,
            open_taken_trades_count=open_taken_trades_count,
        ),
        config=config.filters,
    )

    if flt.blocked:
        rec = ScalpDecisionRecord(
            decision_id=decision_id,
            run_id=run_id,
            strategy="scalp_v1",
            venue=venue,
            canonical_symbol=canonical_symbol,
            decision_time=ts,
            mode=ScalpMode.NO_TRADE.value,
            decision_status="NO_TRADE",
            direction=setup.direction if setup is not None else None,
            reason_codes=flt.reason_codes,
            mode_inputs_json=mode_res.inputs.model_dump(mode="json"),
            features_json=features.model_dump(mode="json"),
            cost_breakdown_json=(
                setup.cost_breakdown.model_dump(mode="json")
                if setup is not None and setup.cost_breakdown is not None
                else None
            ),
            setup_json=setup.model_dump(mode="json") if setup is not None else None,
            params_sha256=params_sha256,
            created_at=ts,
        )
        return ScalpSymbolEvaluation(
            decision=rec,
            setup=setup,
            features=features,
            regime_1h_label=regime_lbl,
            structure_15m_label=struct_lbl,
        )

    if setup is None or setup.status != "READY":
        reason = setup.reason_code if setup is not None else "NO_SETUP"
        rec = ScalpDecisionRecord(
            decision_id=decision_id,
            run_id=run_id,
            strategy="scalp_v1",
            venue=venue,
            canonical_symbol=canonical_symbol,
            decision_time=ts,
            mode=mode_res.mode.value,
            decision_status="WAIT",
            direction=setup.direction if setup is not None else None,
            reason_codes=(reason,),
            mode_inputs_json=mode_res.inputs.model_dump(mode="json"),
            features_json=features.model_dump(mode="json"),
            setup_json=setup.model_dump(mode="json") if setup is not None else None,
            params_sha256=params_sha256,
            created_at=ts,
        )
        return ScalpSymbolEvaluation(
            decision=rec,
            setup=setup,
            features=features,
            regime_1h_label=regime_lbl,
            structure_15m_label=struct_lbl,
        )

    rec = ScalpDecisionRecord(
        decision_id=decision_id,
        run_id=run_id,
        strategy="scalp_v1",
        venue=venue,
        canonical_symbol=canonical_symbol,
        decision_time=ts,
        mode=mode_res.mode.value,
        decision_status="READY",
        direction=setup.direction,
        reason_codes=(setup.reason_code,),
        mode_inputs_json=mode_res.inputs.model_dump(mode="json"),
        features_json=features.model_dump(mode="json"),
        cost_breakdown_json=(
            setup.cost_breakdown.model_dump(mode="json")
            if setup.cost_breakdown is not None
            else None
        ),
        setup_json=setup.model_dump(mode="json"),
        params_sha256=params_sha256,
        created_at=ts,
    )
    return ScalpSymbolEvaluation(
        decision=rec,
        setup=setup,
        features=features,
        regime_1h_label=regime_lbl,
        structure_15m_label=struct_lbl,
    )


class ScalpDigestSummary(BaseModel):
    """SC-13 daily digest summary per mode, per symbol, stage failures, filters fired, fill rate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    period_hours: int = 24
    total_decisions: int = 0
    ready_count: int = 0
    wait_count: int = 0
    no_trade_count: int = 0
    by_mode: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_symbol: dict[str, dict[str, int]] = Field(default_factory=dict)
    stage_failures: dict[str, int] = Field(default_factory=dict)
    filters_fired: dict[str, int] = Field(default_factory=dict)
    setups_total: int = 0
    setups_filled: int = 0
    fill_rate: float = 0.0
    realized_net_r: float = 0.0


def build_scalp_digest_summary(
    *,
    decisions: Sequence[ScalpDecisionRecord],
    setups: Sequence[TradeSetupRecord],
    period_hours: int = 24,
) -> ScalpDigestSummary:
    """Aggregate SC-13 daily digest metrics from persisted scalp decisions and setups."""
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    stage_failures: Counter[str] = Counter()
    filters_fired: Counter[str] = Counter()

    ready_n = 0
    wait_n = 0
    no_trade_n = 0

    for d in decisions:
        st = d.decision_status
        if st == "READY":
            ready_n += 1
        elif st == "WAIT":
            wait_n += 1
        else:
            no_trade_n += 1

        by_mode[d.mode][st] += 1
        by_symbol[d.canonical_symbol][st] += 1
        for code in d.reason_codes:
            if code.startswith("FILTER_"):
                filters_fired[code] += 1
            elif st != "READY":
                stage_failures[code] += 1

    scalp_setups = [s for s in setups if s.strategy == "scalp_v1"]
    filled_statuses = {
        ScalpOutcomeStatus.OPEN_FILLED.value,
        ScalpOutcomeStatus.SL_HIT.value,
        ScalpOutcomeStatus.TP1_THEN_BE.value,
        ScalpOutcomeStatus.TP1_THEN_TIME_STOP.value,
        ScalpOutcomeStatus.TP2_HIT.value,
        ScalpOutcomeStatus.TIME_STOP.value,
    }
    filled = [s for s in scalp_setups if s.outcome_status in filled_statuses]
    fill_rate = (len(filled) / len(scalp_setups)) if scalp_setups else 0.0
    net_r = sum(float(s.realized_r) for s in scalp_setups if s.realized_r is not None)

    return ScalpDigestSummary(
        period_hours=period_hours,
        total_decisions=len(decisions),
        ready_count=ready_n,
        wait_count=wait_n,
        no_trade_count=no_trade_n,
        by_mode={k: dict(v) for k, v in sorted(by_mode.items())},
        by_symbol={k: dict(v) for k, v in sorted(by_symbol.items())},
        stage_failures=dict(stage_failures.most_common(10)),
        filters_fired=dict(filters_fired.most_common(10)),
        setups_total=len(scalp_setups),
        setups_filled=len(filled),
        fill_rate=fill_rate,
        realized_net_r=net_r,
    )


def render_scalp_digest(summary: ScalpDigestSummary) -> str:
    """Render deterministic SC-13 daily Telegram HTML digest."""
    lines = [
        f"<b>Ringkasan Harian Scalp v1 ({summary.period_hours}j terakhir)</b>",
        f"Keputusan: {summary.total_decisions} "
        f"(READY {summary.ready_count} · WAIT {summary.wait_count} "
        f"· NO_TRADE {summary.no_trade_count})",
        f"Fill rate: {summary.setups_filled}/{summary.setups_total} "
        f"({summary.fill_rate * 100:.1f}%) | Realized Net R: {summary.realized_net_r:+.2f}R",
    ]
    if summary.by_mode:
        lines.append("\n<b>Per Mode:</b>")
        for mode, counts in summary.by_mode.items():
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"• {html.escape(mode)}: {html.escape(parts)}")
    if summary.by_symbol:
        lines.append("\n<b>Per Simbol:</b>")
        for sym, counts in summary.by_symbol.items():
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"• {html.escape(sym)}: {html.escape(parts)}")
    if summary.filters_fired:
        lines.append("\n<b>Filter Terpicu:</b>")
        for code, cnt in summary.filters_fired.items():
            lines.append(f"• {html.escape(code)}: {cnt}x")
    if summary.stage_failures:
        lines.append("\n<b>Stage Failures Teratas:</b>")
        for code, cnt in list(summary.stage_failures.items())[:5]:
            lines.append(f"• {html.escape(code)}: {cnt}x")
    return "\n".join(lines)
