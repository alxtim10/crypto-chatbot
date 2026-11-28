"""Meta-labeling dataset builder and experiment runner for deterministic setups (PRD v1.7 §6.1)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.analysis.backtest import (
    BacktestConfig,
    _compute_signed_funding_r,
    _simulate_setup_v17,
)
from crypto_chatbot.analysis.indicators import compute_technical_metrics
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.setup import build_trade_setup
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.ml.evaluation import (
    PromotionV2Config,
    compute_brier_skill_score,
    compute_ece,
    decide_promotion_v2,
)
from crypto_chatbot.ml.features import extract_latest_production_features
from crypto_chatbot.ml.models import (
    CalibratedLogisticPipeline,
    NaiveBaseRateModel,
    select_l2_reg_grid,
)
from crypto_chatbot.ml.statistics import deflated_sharpe_ratio
from crypto_chatbot.ml.validation import (
    assert_no_fold_leakage,
    compute_uniqueness_weights,
    generate_cpcv_paths,
    generate_purged_walk_forward_splits,
    split_dev_and_holdout,
)
from crypto_chatbot.schemas.analysis import (
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    TradeSetup,
)
from crypto_chatbot.schemas.market import MarketCandle, Timeframe
from crypto_chatbot.schemas.ml import (
    CostAssumptions,
    FeatureRow,
    MlExperimentConfig,
)
from crypto_chatbot.schemas.news import SentimentObservation

META_MODEL_VERSION = "meta-l2logreg-platt-v2.0.0"
META_TARGET_NAME = "meta_setup_net_positive"
LEGACY_TARGET_EXPERIMENT = "direction-24h-legacy"
MIN_META_DEV_EVENTS = 300

SIGNED_BASE_FEATURES: tuple[str, ...] = (
    "ret_1",
    "ret_6",
    "ret_20",
    "close_vs_ema20",
    "ema20_vs_ema50",
    "rsi_14_norm",
    "macd_hist_pct",
    "di_diff_norm",
    "cvd_ratio_20",
    "sentiment_net_score",
)

SETUP_META_FEATURES: tuple[str, ...] = (
    "score_aligned",
    "risk_atr",
    "sl_pct",
    "exec_cost_r",
    "net_rr_tp2",
    "entry_src_fvg",
    "entry_src_ob",
    "entry_src_ema20",
    "entry_src_atr",
    "regime_trending_up",
    "regime_trending_down",
    "regime_ranging",
    "regime_high_volatility",
    "regime_compression",
    "structure_aligned",
    "funding_8h_aligned",
)


def align_features_for_direction(
    base_features: Mapping[str, float | None],
    *,
    bias: SetupBias,
    setup: TradeSetup,
    atr_14: float | None,
    regime: Regime,
    market_structure: MarketStructure,
    cost_assumptions: CostAssumptions,
    funding_8h: float | None = None,
    raw_score: float = 0.0,
) -> dict[str, float | None]:
    """Convert signed features to direction-aligned values and append setup features."""
    d = 1.0 if bias is SetupBias.LONG else -1.0
    out: dict[str, float | None] = {}
    for name, val in base_features.items():
        if name in SIGNED_BASE_FEATURES:
            out[name] = None if val is None else float(val) * d
        elif name == "taker_buy_ratio":
            out[name] = None if val is None else (float(val) - 0.5) * d
        else:
            out[name] = val

    e_mid = setup.entry_mid or 1.0
    sl = setup.stop_loss or e_mid
    risk = max(abs(e_mid - sl), 1e-12)
    exec_costs = cost_assumptions.model_copy(update={"funding_bps_per_24h": 0.0})
    exec_cost_r = (exec_costs.round_trip_cost_rate * e_mid) / risk
    tp2 = setup.tp2 if setup.tp2 is not None else e_mid
    net_rr_tp2 = (
        float(setup.net_rr_tp2)
        if setup.net_rr_tp2 is not None
        else (abs(tp2 - e_mid) / risk) - exec_cost_r
    )

    src = (setup.entry_source or "").lower()
    out["score_aligned"] = (raw_score * d) / 9.0
    out["risk_atr"] = (risk / atr_14) if (atr_14 is not None and atr_14 > 0.0) else None
    out["sl_pct"] = risk / e_mid
    out["exec_cost_r"] = exec_cost_r
    out["net_rr_tp2"] = net_rr_tp2
    out["entry_src_fvg"] = 1.0 if "fvg" in src else 0.0
    out["entry_src_ob"] = 1.0 if "ob" in src or "order_block" in src else 0.0
    out["entry_src_ema20"] = 1.0 if "ema" in src else 0.0
    out["entry_src_atr"] = 1.0 if "atr" in src else 0.0

    out["regime_trending_up"] = 1.0 if regime is Regime.TRENDING_UP else 0.0
    out["regime_trending_down"] = 1.0 if regime is Regime.TRENDING_DOWN else 0.0
    out["regime_ranging"] = 1.0 if regime is Regime.RANGING else 0.0
    out["regime_high_volatility"] = 1.0 if regime is Regime.HIGH_VOLATILITY else 0.0
    out["regime_compression"] = 1.0 if regime is Regime.TRANSITIONAL else 0.0

    struct_ok = (bias is SetupBias.LONG and market_structure is MarketStructure.BULLISH) or (
        bias is SetupBias.SHORT and market_structure is MarketStructure.BEARISH
    )
    out["structure_aligned"] = 1.0 if struct_ok else 0.0
    out["funding_8h_aligned"] = None if funding_8h is None else float(funding_8h) * d
    return out


class MetaLabelEvent(BaseModel):
    """Single filled deterministic setup event with causal exit time and aligned features."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    bias: SetupBias
    row: FeatureRow
    net_r_1x: float
    net_r_15x: float
    exec_cost_r: float = Field(ge=0.0)


class MetaLabelDataset(BaseModel):
    """Pooled chronological meta-label dataset across symbols for one timeframe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe
    symbols: tuple[str, ...]
    events: tuple[MetaLabelEvent, ...]
    total_ready_setups: int = Field(ge=0)
    filled_events_count: int = Field(ge=0)
    no_fill_tp1_first_count: int = Field(ge=0)
    no_fill_expired_count: int = Field(ge=0)
    no_fill_other_count: int = Field(ge=0)
    dataset_sha256: str


class MetaLabelExperimentResult(BaseModel):
    """Outcome of a meta-labeling experiment with Promotion Gates v2 and INSUFFICIENT_DATA guard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["READY", "INSUFFICIENT_DATA"]
    model_version: str = META_MODEL_VERSION
    target_name: str = META_TARGET_NAME
    timeframe: Timeframe
    symbols: tuple[str, ...]
    total_ready_setups: int = Field(ge=0)
    filled_events_count: int = Field(ge=0)
    no_fill_excluded_count: int = Field(ge=0)
    dev_events_count: int = Field(ge=0)
    holdout_events_count: int = Field(ge=0)
    selected_l2_reg: float | None = None
    promoted: bool = False
    rejection_reasons: tuple[str, ...] = ()
    dataset_sha256: str


def _latest_point_in_time_funding(
    funding_history: Sequence[tuple[datetime, float]] | None,
    as_of: datetime,
) -> float | None:
    if not funding_history:
        return None
    as_of_utc = as_of.astimezone(UTC)
    eligible = [rate for ts, rate in funding_history if ts.astimezone(UTC) <= as_of_utc]
    return float(eligible[-1]) if eligible else None


def build_meta_label_dataset(
    candles_by_symbol: Mapping[str, Sequence[MarketCandle]],
    *,
    timeframe: Timeframe,
    config: BacktestConfig | None = None,
    tau_r: float = 0.0,
    funding_by_symbol: Mapping[str, Sequence[tuple[datetime, float]]] | None = None,
    sentiment_by_symbol: Mapping[str, Sequence[SentimentObservation]] | None = None,
) -> MetaLabelDataset:
    """Build pooled point-in-time meta-label events from the deterministic §7 backtest engine."""
    cfg = config or BacktestConfig(
        warmup_bars=600,
        entry_expiry_bars=6,
        max_hold_bars=24,
        require_ema200=True,
    )
    all_events: list[MetaLabelEvent] = []
    total_ready = 0
    no_fill_tp1 = 0
    no_fill_exp = 0
    no_fill_other = 0
    symbols_sorted = tuple(sorted(candles_by_symbol.keys()))

    for symbol in symbols_sorted:
        closed = sorted(
            (c for c in candles_by_symbol[symbol] if c.is_closed),
            key=lambda c: c.open_time,
        )
        if len(closed) <= cfg.warmup_bars + 2:
            continue
        sym_funding = funding_by_symbol.get(symbol) if funding_by_symbol else None
        sym_sent = sentiment_by_symbol.get(symbol) if sentiment_by_symbol else None
        feat_cfg = MlExperimentConfig(
            canonical_symbol=symbol,
            timeframe=timeframe,
            cost_assumptions=cfg.cost_assumptions,
        )

        idx = cfg.warmup_bars
        while idx < len(closed) - 1:
            win_start = max(0, idx + 1 - cfg.max_lookback_bars)
            history = closed[win_start : idx + 1]
            bar_candle = history[-1]
            as_of = bar_candle.close_time

            ind = compute_technical_metrics(history)
            reg = classify_regime(ind.metrics)
            if reg is None:
                idx += cfg.step_bars
                continue
            struct = analyze_market_structure(history)
            smc = analyze_smc(history, structure=struct, atr_14=ind.metrics.atr_14)
            setup = build_trade_setup(
                metrics=ind.metrics,
                regime=reg,
                structure=struct,
                smc=smc,
                primary_timeframe=timeframe,
                as_of=as_of,
                cost_assumptions=cfg.cost_assumptions,
                require_ema200=cfg.require_ema200,
            )

            if setup.status is not SetupStatus.READY or setup.bias not in (
                SetupBias.LONG,
                SetupBias.SHORT,
            ):
                idx += cfg.step_bars
                continue

            total_ready += 1
            sim = _simulate_setup_v17(
                setup,
                closed[idx + 1 :],
                start_bar_index=idx,
                max_hold_bars=cfg.max_hold_bars,
                entry_expiry_bars=cfg.entry_expiry_bars,
                flat_boundary=cfg.flat_boundary,
            )
            if (
                sim is None
                or sim.cancel_reason is not None
                or sim.fill_time is None
                or sim.exit_time is None
            ):
                if sim is not None and sim.cancel_reason == "NO_FILL_TP1_FIRST":
                    no_fill_tp1 += 1
                elif sim is not None and sim.cancel_reason == "NO_FILL_EXPIRED":
                    no_fill_exp += 1
                else:
                    no_fill_other += 1
                idx += cfg.step_bars
                continue

            if setup.entry_mid is None or setup.stop_loss is None:
                idx += cfg.step_bars
                continue
            risk = abs(setup.entry_mid - setup.stop_loss)
            exec_costs = cfg.cost_assumptions.model_copy(update={"funding_bps_per_24h": 0.0})
            exec_cost_r = (exec_costs.round_trip_cost_rate * sim.fill_price) / risk
            held_hours = sim.bars_held * (timeframe.duration.total_seconds() / 3600.0)
            signed_fund_r, fund_assumed = _compute_signed_funding_r(
                is_long=(setup.bias is SetupBias.LONG),
                fill_time=sim.fill_time,
                exit_time=sim.exit_time,
                fill_price=sim.fill_price,
                risk=risk,
                remaining_history=sim.remaining_history,
                funding_history=sym_funding,
            )
            if fund_assumed:
                f_rate = (held_hours / 24.0) * (cfg.cost_assumptions.funding_bps_per_24h / 10_000.0)
                funding_r_signed = -((f_rate * sim.fill_price) / risk)
            else:
                funding_r_signed = signed_fund_r

            net_r_1x = sim.gross_r - exec_cost_r + funding_r_signed
            net_r_15x = sim.gross_r - 1.5 * exec_cost_r + funding_r_signed
            label = 1 if net_r_1x > tau_r else 0

            try:
                base_row = extract_latest_production_features(
                    history,
                    config=feat_cfg,
                    sentiment_observations=sym_sent,
                )
            except ValueError:
                idx = max(idx + cfg.step_bars, sim.exit_bar_idx + cfg.cooldown_bars)
                continue

            pit_funding = _latest_point_in_time_funding(sym_funding, as_of)
            aligned = align_features_for_direction(
                base_row.features,
                bias=setup.bias,
                setup=setup,
                atr_14=ind.metrics.atr_14,
                regime=reg,
                market_structure=struct.market_structure,
                cost_assumptions=cfg.cost_assumptions,
                funding_8h=pit_funding,
            )
            exit_ts = sim.exit_time
            if exit_ts <= as_of:
                idx = max(idx + cfg.step_bars, sim.exit_bar_idx + cfg.cooldown_bars)
                continue

            meta_row = FeatureRow(
                prediction_time=as_of,
                available_time=as_of,
                label_end_time=exit_ts,
                regime=reg,
                features=aligned,
                forward_return=round(net_r_1x, 6),
                label=label,
            )
            all_events.append(
                MetaLabelEvent(
                    symbol=symbol,
                    timeframe=timeframe,
                    bias=setup.bias,
                    row=meta_row,
                    net_r_1x=round(net_r_1x, 6),
                    net_r_15x=round(net_r_15x, 6),
                    exec_cost_r=round(exec_cost_r, 6),
                )
            )
            idx = max(idx + cfg.step_bars, sim.exit_bar_idx + cfg.cooldown_bars)

    all_events.sort(key=lambda ev: (ev.row.prediction_time, ev.symbol))
    digest_payload = [
        {
            "s": ev.symbol,
            "pt": ev.row.prediction_time.isoformat(),
            "et": ev.row.label_end_time.isoformat(),
            "y": ev.row.label,
            "r": round(ev.net_r_1x, 6),
        }
        for ev in all_events
    ]
    dataset_sha = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return MetaLabelDataset(
        timeframe=timeframe,
        symbols=symbols_sorted,
        events=tuple(all_events),
        total_ready_setups=total_ready,
        filled_events_count=len(all_events),
        no_fill_tp1_first_count=no_fill_tp1,
        no_fill_expired_count=no_fill_exp,
        no_fill_other_count=no_fill_other,
        dataset_sha256=dataset_sha,
    )


def _events_to_matrix(
    events: Sequence[MetaLabelEvent],
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [
                np.nan if ev.row.features.get(name) is None else float(ev.row.features[name] or 0.0)
                for name in feature_names
            ]
            for ev in events
        ],
        dtype=np.float64,
    )
    y = np.array([ev.row.label for ev in events], dtype=np.float64)
    return x, y


def run_meta_labeling_experiment(
    dataset: MetaLabelDataset,
    *,
    min_dev_events: int = MIN_META_DEV_EVENTS,
    holdout_fraction: float = 0.20,
    entry_expiry_bars: int = 6,
    max_hold_bars: int = 24,
    base_rate_delta: float = 0.05,
    n_splits: int = 4,
    promotion_config: PromotionV2Config | None = None,
) -> MetaLabelExperimentResult:
    """Run purged walk-forward meta-labeling with uniqueness weights & Promotion Gates v2."""
    no_fill_total = (
        dataset.no_fill_tp1_first_count
        + dataset.no_fill_expired_count
        + dataset.no_fill_other_count
    )
    rows = tuple(ev.row for ev in dataset.events)
    embargo_bars = max(3, entry_expiry_bars + max_hold_bars)

    if len(rows) < min_dev_events:
        return MetaLabelExperimentResult(
            status="INSUFFICIENT_DATA",
            timeframe=dataset.timeframe,
            symbols=dataset.symbols,
            total_ready_setups=dataset.total_ready_setups,
            filled_events_count=dataset.filled_events_count,
            no_fill_excluded_count=no_fill_total,
            dev_events_count=len(rows),
            holdout_events_count=0,
            promoted=False,
            rejection_reasons=(
                f"INSUFFICIENT_DATA: filled events ({len(rows)}) below minimum ({min_dev_events}).",
            ),
            dataset_sha256=dataset.dataset_sha256,
        )

    try:
        dev_rows, holdout_rows = split_dev_and_holdout(
            rows,
            holdout_fraction=holdout_fraction,
            embargo_bars=embargo_bars,
        )
    except ValueError as exc:
        return MetaLabelExperimentResult(
            status="INSUFFICIENT_DATA",
            timeframe=dataset.timeframe,
            symbols=dataset.symbols,
            total_ready_setups=dataset.total_ready_setups,
            filled_events_count=dataset.filled_events_count,
            no_fill_excluded_count=no_fill_total,
            dev_events_count=0,
            holdout_events_count=0,
            promoted=False,
            rejection_reasons=(f"INSUFFICIENT_DATA: {exc}",),
            dataset_sha256=dataset.dataset_sha256,
        )

    if len(dev_rows) < min_dev_events:
        return MetaLabelExperimentResult(
            status="INSUFFICIENT_DATA",
            timeframe=dataset.timeframe,
            symbols=dataset.symbols,
            total_ready_setups=dataset.total_ready_setups,
            filled_events_count=dataset.filled_events_count,
            no_fill_excluded_count=no_fill_total,
            dev_events_count=len(dev_rows),
            holdout_events_count=len(holdout_rows),
            promoted=False,
            rejection_reasons=(
                f"INSUFFICIENT_DATA: dev events ({len(dev_rows)}) below minimum "
                f"({min_dev_events}).",
            ),
            dataset_sha256=dataset.dataset_sha256,
        )

    row_to_event = {id(ev.row): ev for ev in dataset.events}
    dev_events = tuple(row_to_event[id(r)] for r in dev_rows)
    holdout_events = tuple(row_to_event[id(r)] for r in holdout_rows)

    feature_names = tuple(sorted(dev_rows[0].features.keys()))
    x_dev, y_dev = _events_to_matrix(dev_events, feature_names)
    x_hold, y_hold = _events_to_matrix(holdout_events, feature_names)
    w_dev = compute_uniqueness_weights(dev_rows, normalize=True)
    w_hold = compute_uniqueness_weights(holdout_rows, normalize=True)

    splits = generate_purged_walk_forward_splits(
        dev_rows,
        n_splits=n_splits,
        embargo_bars=embargo_bars,
    )
    assert_no_fold_leakage(dev_rows, splits, min_embargo_bars=embargo_bars)

    best_lam, _lam_scores = select_l2_reg_grid(
        x_dev,
        y_dev,
        feature_names=feature_names,
        pred_times_train=[r.prediction_time for r in dev_rows],
        label_end_times_train=[r.label_end_time for r in dev_rows],
        sample_weight=w_dev,
    )

    oof_y: list[float] = []
    oof_pm: list[float] = []
    oof_pn: list[float] = []
    oof_w: list[float] = []
    oof_sel_1x: list[float] = []
    oof_sel_15x: list[float] = []
    oof_all_1x: list[float] = []
    fold_stats: list[tuple[float, float]] = []

    for split in splits:
        tr = list(split.train_indices)
        va = list(split.val_indices)
        naive = NaiveBaseRateModel().fit(y_dev[tr])
        pn_va = naive.predict_proba(len(va))

        pipe = CalibratedLogisticPipeline(
            feature_names=feature_names,
            l2_reg=best_lam,
        ).fit(
            x_dev[tr],
            y_dev[tr],
            pred_times_train=[dev_rows[i].prediction_time for i in tr],
            label_end_times_train=[dev_rows[i].label_end_time for i in tr],
            sample_weight=w_dev[tr],
        )
        pm_va = pipe.predict_proba(x_dev[va])
        w_va = w_dev[va]
        norm_w_va = w_va / float(np.sum(w_va))
        naive_bs = float(np.sum(norm_w_va * (pn_va - y_dev[va]) ** 2))
        model_bs = float(np.sum(norm_w_va * (pm_va - y_dev[va]) ** 2))
        fold_bss = compute_brier_skill_score(model_bs, naive_bs)

        thresh = pipe.base_rate_ + base_rate_delta
        fold_sel_r: list[float] = []
        for loc_i, orig_i in enumerate(va):
            ev = dev_events[orig_i]
            oof_y.append(float(y_dev[orig_i]))
            oof_pm.append(float(pm_va[loc_i]))
            oof_pn.append(float(pn_va[loc_i]))
            oof_w.append(float(w_dev[orig_i]))
            oof_all_1x.append(ev.net_r_1x)
            if float(pm_va[loc_i]) >= thresh:
                fold_sel_r.append(ev.net_r_1x)
                oof_sel_1x.append(ev.net_r_1x)
                oof_sel_15x.append(ev.net_r_15x)

        fold_exp = float(np.mean(fold_sel_r)) if fold_sel_r else -1.0
        fold_stats.append((fold_bss, fold_exp))

    # Final model fit on full dev set -> evaluate untouched holdout
    final_naive = NaiveBaseRateModel().fit(y_dev)
    pn_hold = final_naive.predict_proba(len(holdout_rows))
    final_pipe = CalibratedLogisticPipeline(
        feature_names=feature_names,
        l2_reg=best_lam,
    ).fit(
        x_dev,
        y_dev,
        pred_times_train=[r.prediction_time for r in dev_rows],
        label_end_times_train=[r.label_end_time for r in dev_rows],
        sample_weight=w_dev,
    )
    pm_hold = final_pipe.predict_proba(x_hold)
    norm_w_hold = w_hold / float(np.sum(w_hold))
    hold_naive_bs = float(np.sum(norm_w_hold * (pn_hold - y_hold) ** 2))
    hold_model_bs = float(np.sum(norm_w_hold * (pm_hold - y_hold) ** 2))
    hold_bss = compute_brier_skill_score(hold_model_bs, hold_naive_bs)
    hold_ece = compute_ece(y_hold, pm_hold, n_bins=5, sample_weight=w_hold)
    hold_thresh = final_pipe.base_rate_ + base_rate_delta
    hold_sel_r = [
        holdout_events[i].net_r_1x for i in range(len(holdout_events)) if pm_hold[i] >= hold_thresh
    ]
    hold_sel_exp = float(np.mean(hold_sel_r)) if hold_sel_r else -1.0

    # DSR on selected OOS R series
    dsr_val: float | None = None
    if len(oof_sel_1x) >= 3:
        dsr_res = deflated_sharpe_ratio(
            np.asarray(oof_sel_1x, dtype=np.float64),
            trials=4,  # 4 lambda grid points
            trial_sharpes=[0.0, 0.0, 0.0, 0.0],
            obs_frequency="per_trade",
            trial_frequencies=["per_trade"] * 4,
        )
        dsr_val = dsr_res.deflated_sharpe_ratio

    # CPCV profitable path fraction
    cpcv_frac: float | None = None
    try:
        paths = generate_cpcv_paths(dev_rows, n_groups=6, test_groups_per_path=2, embargo_bars=3)
        profitable_paths = 0
        for p in paths:
            p_tr = list(p.train_indices)
            p_te = list(p.test_indices)
            if not p_tr or not p_te:
                continue
            cp_pipe = CalibratedLogisticPipeline(
                feature_names=feature_names,
                l2_reg=best_lam,
            ).fit(
                x_dev[p_tr],
                y_dev[p_tr],
                pred_times_train=[dev_rows[i].prediction_time for i in p_tr],
                label_end_times_train=[dev_rows[i].label_end_time for i in p_tr],
                sample_weight=w_dev[p_tr],
            )
            cp_pm = cp_pipe.predict_proba(x_dev[p_te])
            cp_th = cp_pipe.base_rate_ + base_rate_delta
            cp_sel = [
                dev_events[orig_i].net_r_1x for j, orig_i in enumerate(p_te) if cp_pm[j] >= cp_th
            ]
            if cp_sel and float(np.mean(cp_sel)) > 0.0:
                profitable_paths += 1
        cpcv_frac = profitable_paths / len(paths) if paths else None
    except ValueError:
        cpcv_frac = None

    decision = decide_promotion_v2(
        oof_y_true=oof_y,
        oof_p_model=oof_pm,
        oof_p_naive=oof_pn,
        oof_selected_r_1x=oof_sel_1x,
        oof_selected_r_15x=oof_sel_15x,
        oof_all_setup_expectancy=float(np.mean(oof_all_1x)) if oof_all_1x else 0.0,
        fold_bss_and_expectancy=fold_stats,
        holdout_selected_trades=len(hold_sel_r),
        holdout_bss=hold_bss,
        holdout_selected_expectancy=hold_sel_exp,
        holdout_weighted_ece=hold_ece,
        dsr=dsr_val,
        cpcv_profitable_fraction=cpcv_frac,
        oof_sample_weight=oof_w,
        config=promotion_config,
    )

    return MetaLabelExperimentResult(
        status="READY",
        timeframe=dataset.timeframe,
        symbols=dataset.symbols,
        total_ready_setups=dataset.total_ready_setups,
        filled_events_count=dataset.filled_events_count,
        no_fill_excluded_count=no_fill_total,
        dev_events_count=len(dev_rows),
        holdout_events_count=len(holdout_rows),
        selected_l2_reg=best_lam,
        promoted=decision.promoted,
        rejection_reasons=decision.rejection_reasons,
        dataset_sha256=dataset.dataset_sha256,
    )
