"""15-minute multi-timeframe top-down alert evaluator (4h/1h bias -> 15m setup -> 5m confirm).

Consumes the `AnalysisSnapshot` and closed-candle batches built by the exact same
`AnalysisService.analyze_with_batches` path used by `analyze`. Zero duplicate
regime/structure/setup calculations and zero LLM calls.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.alerts.config import MtfTimeframesConfig
from crypto_chatbot.analysis.scanner import ScannedSetupItem
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DealingZone,
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    TradeSetup,
)
from crypto_chatbot.schemas.market import MarketCandle, QualityStatus, Timeframe
from crypto_chatbot.storage.repositories import AlertSymbolState

MTF_ENGINE_VERSION = "mtf-15m-v2"
MIN_REGIME_BARS = 114  # ATR(14) + ATR percentile(100) required by classify_regime

StageName = Literal["data", "bias", "setup", "confirm", "cost_gate", "error"]


class MixedVenueError(ValueError):
    """Raised when timeframes in a single MTF bundle originate from different venues."""


def make_mtf_run_id(candle_close_time: datetime) -> str:
    """Build deterministic idempotent run_id for the 15m MTF cadence (`mtf:<15m close_time>`)."""
    close_utc = candle_close_time.astimezone(UTC)
    return f"mtf:{close_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def assert_single_venue_bundle(batches: Mapping[Timeframe, OhlcvBatch]) -> str:
    """Verify every non-empty timeframe batch and candle belongs to ONE venue; return that venue."""
    venues: set[str] = set()
    for tf, batch in batches.items():
        if not batch.candles:
            continue
        meta_venue = batch.metadata.venue.strip().lower()
        if meta_venue:
            venues.add(meta_venue)
        for candle in batch.candles:
            cv = candle.venue.strip().lower()
            if cv:
                venues.add(cv)
        if len(venues) > 1:
            raise MixedVenueError(
                f"Mixed venues detected across MTF bundle at {tf.value}: {sorted(venues)}"
            )
    if not venues:
        raise ValueError("MTF candle bundle contains no closed candles")
    return next(iter(venues))


def _filter_closed_candles(
    candles: Sequence[MarketCandle],
    cutoff: datetime,
) -> list[MarketCandle]:
    cutoff_utc = cutoff.astimezone(UTC)
    return [
        c
        for c in candles
        if c.is_closed
        and c.close_time.astimezone(UTC) <= cutoff_utc
        and c.available_time.astimezone(UTC) <= cutoff_utc
    ]


class MtfBiasEvaluation(BaseModel):
    """Result of the HTF (4h + 1h) top-down bias gate derived from `AnalysisSnapshot`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agreed_direction: Literal["long", "short"] | None = None
    missing_data: bool = False
    per_tf_bar_counts: dict[str, int] = Field(default_factory=dict)
    per_tf_summary: dict[str, str] = Field(default_factory=dict)
    per_tf_regime: dict[str, Regime | None] = Field(default_factory=dict)
    per_tf_structure: dict[str, MarketStructure | None] = Field(default_factory=dict)
    per_tf_adx_14: dict[str, float | None] = Field(default_factory=dict)
    reason: str


def evaluate_htf_bias_from_snapshot(
    snapshot: AnalysisSnapshot,
    bias_timeframes: Sequence[Timeframe],
    *,
    batches: Mapping[Timeframe, OhlcvBatch] | None = None,
) -> MtfBiasEvaluation:
    """Read HTF regime + structure directly from `snapshot.mtf_overview` (built by `analyze`).

    If any bias timeframe lacks sufficient bars to classify regime or structure (`None`),
    returns `missing_data=True` with explicit bar counts (stage `"data"`), and NEVER emits
    `"unknown"` in a bias reason.
    """
    ov_by_tf = {ov.timeframe: ov for ov in snapshot.mtf_overview}
    directions: list[Literal["long", "short"] | None] = []
    summaries: dict[str, str] = {}
    regimes: dict[str, Regime | None] = {}
    structures: dict[str, MarketStructure | None] = {}
    adx_map: dict[str, float | None] = {}
    bar_counts: dict[str, int] = {}
    missing_tfs: list[str] = []

    cutoff = snapshot.as_of.astimezone(UTC)
    for tf in bias_timeframes:
        batch = batches.get(tf) if batches is not None else None
        bar_cnt = len(_filter_closed_candles(batch.candles, cutoff)) if batch is not None else 0
        bar_counts[tf.value] = bar_cnt

        ov = ov_by_tf.get(tf)
        if ov is None or ov.regime is None or ov.market_structure is None:
            regimes[tf.value] = ov.regime if ov is not None else None
            structures[tf.value] = ov.market_structure if ov is not None else None
            adx_map[tf.value] = ov.adx_14 if ov is not None else None
            missing_tfs.append(f"{tf.value}={bar_cnt} bar")
            directions.append(None)
            continue

        summaries[tf.value] = f"{ov.regime.value}/{ov.market_structure.value}"
        regimes[tf.value] = ov.regime
        structures[tf.value] = ov.market_structure
        adx_map[tf.value] = ov.adx_14

        if ov.regime is Regime.TRENDING_UP and ov.market_structure is MarketStructure.BULLISH:
            directions.append("long")
        elif ov.regime is Regime.TRENDING_DOWN and ov.market_structure is MarketStructure.BEARISH:
            directions.append("short")
        else:
            directions.append(None)

    if missing_tfs:
        return MtfBiasEvaluation(
            agreed_direction=None,
            missing_data=True,
            per_tf_bar_counts=bar_counts,
            per_tf_summary=summaries,
            per_tf_regime=regimes,
            per_tf_structure=structures,
            per_tf_adx_14=adx_map,
            reason=(
                f"Data HTF belum cukup untuk klasifikasi rezim "
                f"({', '.join(missing_tfs)}; butuh >={MIN_REGIME_BARS} bar tertutup)"
            ),
        )

    summary_str = ", ".join(f"{k}={v}" for k, v in summaries.items())
    if directions and all(d == "long" for d in directions):
        return MtfBiasEvaluation(
            agreed_direction="long",
            missing_data=False,
            per_tf_bar_counts=bar_counts,
            per_tf_summary=summaries,
            per_tf_regime=regimes,
            per_tf_structure=structures,
            per_tf_adx_14=adx_map,
            reason=f"HTF selaras LONG ({summary_str})",
        )
    if directions and all(d == "short" for d in directions):
        return MtfBiasEvaluation(
            agreed_direction="short",
            missing_data=False,
            per_tf_bar_counts=bar_counts,
            per_tf_summary=summaries,
            per_tf_regime=regimes,
            per_tf_structure=structures,
            per_tf_adx_14=adx_map,
            reason=f"HTF selaras SHORT ({summary_str})",
        )
    return MtfBiasEvaluation(
        agreed_direction=None,
        missing_data=False,
        per_tf_bar_counts=bar_counts,
        per_tf_summary=summaries,
        per_tf_regime=regimes,
        per_tf_structure=structures,
        per_tf_adx_14=adx_map,
        reason=f"Bias HTF belum sepakat ({summary_str})",
    )


def classify_setup_wait_stage(wait_reason: str | None) -> Literal["cost_gate", "setup"]:
    """Distinguish cost-gate rejections from structural/setup rejections."""
    if not wait_reason:
        return "setup"
    low = wait_reason.lower()
    cost_markers = (
        "biaya eksekusi",
        "exec_cost",
        "execution cost",
        "cost_gate",
        "r:r bersih",
        "net r:r",
        "net_rr",
    )
    if any(marker in low for marker in cost_markers):
        return "cost_gate"
    return "setup"


def evaluate_5m_confirmation(
    candles_5m: Sequence[MarketCandle],
    *,
    direction: Literal["long", "short"],
    entry_low: float,
    entry_high: float,
    stop_loss: float,
    cutoff: datetime,
) -> tuple[bool, bool, str]:
    """Evaluate confirmation on the LAST CLOSED 5m candle.

    Returns `(confirmed, invalidated, detail)`.
    """
    closed = _filter_closed_candles(candles_5m, cutoff)
    if len(closed) < 2:
        return False, False, f"Candle tutup 5m belum cukup ({len(closed)} bar)"

    last = closed[-1]
    prev = closed[-2]
    ts_lbl = last.close_time.astimezone(UTC).strftime("%H:%M UTC")

    if direction == "long":
        if last.close <= stop_loss:
            return False, True, f"Close 5m ({ts_lbl}) menembus SL long; setup batal"
        bullish_bar = last.close > last.open
        broke_prev_high = last.close > prev.high
        zone_reject = last.low <= entry_high and last.close >= entry_low
        if bullish_bar and (broke_prev_high or zone_reject):
            trigger = "break high 5m" if broke_prev_high else "rejeksi zona entry 5m"
            return True, False, f"Konfirmasi 5m LONG ({trigger}, close {ts_lbl})"
        return False, False, f"Menunggu konfirmasi candle tutup 5m LONG (terakhir {ts_lbl})"

    # Short direction
    if last.close >= stop_loss:
        return False, True, f"Close 5m ({ts_lbl}) menembus SL short; setup batal"
    bearish_bar = last.close < last.open
    broke_prev_low = last.close < prev.low
    zone_reject = last.high >= entry_low and last.close <= entry_high
    if bearish_bar and (broke_prev_low or zone_reject):
        trigger = "break low 5m" if broke_prev_low else "rejeksi zona entry 5m"
        return True, False, f"Konfirmasi 5m SHORT ({trigger}, close {ts_lbl})"
    return False, False, f"Menunggu konfirmasi candle tutup 5m SHORT (terakhir {ts_lbl})"


class ArmedSetupSnapshot(BaseModel):
    """Serializable 15m setup awaiting 5m confirmation within `setup_expiry_minutes`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    canonical_symbol: str
    timeframe: Timeframe = Timeframe.M15
    venue: str
    direction: Literal["long", "short"]
    as_of: datetime
    last_close: float
    dealing_zone: str | None = None
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    rr_tp1: float | None = None
    rr_tp2: float | None = None
    rr_tp3: float | None = None
    htf_bias_summary: str = ""
    entry_source: str | None = None

    def to_scanned_item(self, *, confirmation_detail: str) -> ScannedSetupItem:
        dz: DealingZone | None = None
        if self.dealing_zone:
            try:
                dz = DealingZone(self.dealing_zone)
            except ValueError:
                dz = None
        return ScannedSetupItem(
            symbol=self.canonical_symbol,
            timeframe=self.timeframe,
            venue=self.venue,
            status=SetupStatus.READY,
            bias=SetupBias.LONG if self.direction == "long" else SetupBias.SHORT,
            last_close=self.last_close,
            entry_low=self.entry_low,
            entry_high=self.entry_high,
            stop_loss=self.stop_loss,
            tp1=self.tp1,
            tp2=self.tp2,
            tp3=self.tp3,
            rr_tp2=self.rr_tp2,
            dealing_zone=dz,
            ltf_confirmation=f"{self.htf_bias_summary} | {confirmation_detail}".strip(" |"),
            snapshot_id=self.snapshot_id,
            as_of=self.as_of,
        )


class MtfSymbolDecision(BaseModel):
    """Per-symbol outcome of one 15m top-down MTF evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str
    venue: str
    setup_timeframe: Timeframe
    setup_close_time: datetime
    outcome: Literal["ready", "armed", "wait", "expired"]
    promoted_from_armed: bool = False
    direction: Literal["long", "short"] | None = None
    item: ScannedSetupItem | None = None
    trade_setup: TradeSetup | None = None
    armed_setup: ArmedSetupSnapshot | None = None
    expired_armed_setup: ArmedSetupSnapshot | None = None
    failed_stage: StageName | None = None
    failed_reason: str | None = None


def _make_mtf_snapshot_id(
    *,
    canonical_symbol: str,
    venue: str,
    timeframe: Timeframe,
    setup_close_time: datetime,
    direction: str,
) -> str:
    raw = (
        f"mtf|{venue}|{canonical_symbol}|{timeframe.value}|"
        f"{setup_close_time.astimezone(UTC).isoformat()}|{direction}"
    )
    return f"mtf-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _build_armed_from_trade_setup(
    *,
    canonical_symbol: str,
    venue: str,
    setup_tf: Timeframe,
    setup_close_time: datetime,
    last_close: float,
    dealing_zone: DealingZone | None,
    direction: Literal["long", "short"],
    trade_setup: TradeSetup,
    htf_bias_summary: str,
) -> ArmedSetupSnapshot:
    if (
        trade_setup.entry_low is None
        or trade_setup.entry_high is None
        or trade_setup.stop_loss is None
    ):
        raise ValueError("trade_setup must have entry_low, entry_high, and stop_loss")
    snap_id = _make_mtf_snapshot_id(
        canonical_symbol=canonical_symbol,
        venue=venue,
        timeframe=setup_tf,
        setup_close_time=setup_close_time,
        direction=direction,
    )
    return ArmedSetupSnapshot(
        snapshot_id=snap_id,
        canonical_symbol=canonical_symbol,
        timeframe=setup_tf,
        venue=venue,
        direction=direction,
        as_of=setup_close_time,
        last_close=last_close,
        dealing_zone=dealing_zone.value if dealing_zone is not None else None,
        entry_low=trade_setup.entry_low,
        entry_high=trade_setup.entry_high,
        stop_loss=trade_setup.stop_loss,
        tp1=trade_setup.tp1,
        tp2=trade_setup.tp2,
        tp3=trade_setup.tp3,
        rr_tp1=trade_setup.rr_tp1,
        rr_tp2=trade_setup.rr_tp2,
        rr_tp3=trade_setup.rr_tp3,
        htf_bias_summary=htf_bias_summary,
        entry_source=trade_setup.entry_source,
    )


def evaluate_symbol_mtf(
    *,
    snapshot: AnalysisSnapshot,
    batches: Mapping[Timeframe, OhlcvBatch],
    mtf_config: MtfTimeframesConfig,
    prev_state: AlertSymbolState | None,
    now_utc: datetime,
) -> MtfSymbolDecision:
    """Evaluate one symbol using the shared `AnalysisSnapshot` from `analyze_with_batches`."""
    # 1. Strict single-venue validation across all timeframes in the bundle.
    venue = assert_single_venue_bundle(batches)
    canonical_symbol = snapshot.canonical_symbol
    setup_tf = mtf_config.setup
    confirm_tf = mtf_config.confirm
    setup_close_time = snapshot.as_of.astimezone(UTC)

    setup_batch = batches.get(setup_tf)
    setup_bar_cnt = (
        len(_filter_closed_candles(setup_batch.candles, setup_close_time))
        if setup_batch is not None
        else 0
    )
    if (
        snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA
        or snapshot.technical is None
        or snapshot.trade_setup is None
    ):
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="wait",
            failed_stage="data",
            failed_reason=(
                f"Data {setup_tf.value} belum cukup ({setup_bar_cnt} bar tertutup; "
                f"status={snapshot.quality_status.value})"
            ),
        )

    confirm_batch = batches.get(confirm_tf)
    confirm_candles = (
        _filter_closed_candles(confirm_batch.candles, setup_close_time)
        if confirm_batch is not None
        else []
    )

    # 2. Check existing ARMED setup first (expiry / 5m confirmation promotion).
    expired_armed: ArmedSetupSnapshot | None = None
    if (
        prev_state is not None
        and prev_state.last_status == "armed"
        and prev_state.armed_setup_json is not None
        and prev_state.armed_expires_at is not None
    ):
        armed = ArmedSetupSnapshot.model_validate(prev_state.armed_setup_json)
        if now_utc >= prev_state.armed_expires_at.astimezone(UTC):
            expired_armed = armed
        else:
            bias_check = evaluate_htf_bias_from_snapshot(
                snapshot,
                mtf_config.bias,
                batches=batches,
            )
            if bias_check.missing_data:
                return MtfSymbolDecision(
                    canonical_symbol=canonical_symbol,
                    venue=venue,
                    setup_timeframe=setup_tf,
                    setup_close_time=setup_close_time,
                    outcome="armed",
                    direction=armed.direction,
                    armed_setup=armed,
                    failed_stage="data",
                    failed_reason=bias_check.reason,
                )
            if bias_check.agreed_direction != armed.direction:
                return MtfSymbolDecision(
                    canonical_symbol=canonical_symbol,
                    venue=venue,
                    setup_timeframe=setup_tf,
                    setup_close_time=setup_close_time,
                    outcome="wait",
                    failed_stage="bias",
                    failed_reason=(
                        f"Setup ARMED {armed.direction.upper()} dibatalkan: {bias_check.reason}"
                    ),
                )
            confirmed, invalidated, detail = evaluate_5m_confirmation(
                confirm_candles,
                direction=armed.direction,
                entry_low=armed.entry_low,
                entry_high=armed.entry_high,
                stop_loss=armed.stop_loss,
                cutoff=setup_close_time,
            )
            if invalidated:
                return MtfSymbolDecision(
                    canonical_symbol=canonical_symbol,
                    venue=venue,
                    setup_timeframe=setup_tf,
                    setup_close_time=setup_close_time,
                    outcome="wait",
                    failed_stage="setup",
                    failed_reason=detail,
                )
            if confirmed:
                item = armed.to_scanned_item(confirmation_detail=detail)
                return MtfSymbolDecision(
                    canonical_symbol=canonical_symbol,
                    venue=venue,
                    setup_timeframe=setup_tf,
                    setup_close_time=setup_close_time,
                    outcome="ready",
                    promoted_from_armed=True,
                    direction=armed.direction,
                    item=item,
                    armed_setup=armed,
                )
            return MtfSymbolDecision(
                canonical_symbol=canonical_symbol,
                venue=venue,
                setup_timeframe=setup_tf,
                setup_close_time=setup_close_time,
                outcome="armed",
                direction=armed.direction,
                armed_setup=armed,
                failed_stage="confirm",
                failed_reason=detail,
            )

    # 3. Stage 1 — HTF Bias from `snapshot.mtf_overview` (identical to `analyze`).
    bias_eval = evaluate_htf_bias_from_snapshot(
        snapshot,
        mtf_config.bias,
        batches=batches,
    )
    if bias_eval.missing_data:
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="expired" if expired_armed is not None else "wait",
            expired_armed_setup=expired_armed,
            failed_stage="data",
            failed_reason=bias_eval.reason,
        )
    if bias_eval.agreed_direction is None:
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="expired" if expired_armed is not None else "wait",
            expired_armed_setup=expired_armed,
            failed_stage="bias",
            failed_reason=bias_eval.reason,
        )

    htf_dir = bias_eval.agreed_direction

    # 4. Stage 2 — 15m Setup directly from `snapshot.trade_setup` (built by `analyze`).
    trade_setup = snapshot.trade_setup
    if trade_setup.status is not SetupStatus.READY:
        stage = classify_setup_wait_stage(trade_setup.wait_reason)
        reason = trade_setup.wait_reason or "Setup 15m belum memenuhi syarat entry/SL/TP"
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="expired" if expired_armed is not None else "wait",
            expired_armed_setup=expired_armed,
            failed_stage=stage,
            failed_reason=reason,
        )

    expected_bias = SetupBias.LONG if htf_dir == "long" else SetupBias.SHORT
    if trade_setup.bias is not expected_bias:
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="expired" if expired_armed is not None else "wait",
            expired_armed_setup=expired_armed,
            failed_stage="setup",
            failed_reason=(
                f"Setup 15m ({trade_setup.bias.value.upper()}) berlawanan dengan "
                f"bias HTF ({htf_dir.upper()})"
            ),
        )

    if (
        trade_setup.entry_low is None
        or trade_setup.entry_high is None
        or trade_setup.stop_loss is None
    ):
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="expired" if expired_armed is not None else "wait",
            expired_armed_setup=expired_armed,
            failed_stage="setup",
            failed_reason="Zona entry atau SL pada setup 15m tidak lengkap",
        )

    armed_snap = _build_armed_from_trade_setup(
        canonical_symbol=canonical_symbol,
        venue=venue,
        setup_tf=setup_tf,
        setup_close_time=setup_close_time,
        last_close=snapshot.technical.last_close,
        dealing_zone=snapshot.smc.dealing_zone if snapshot.smc is not None else None,
        direction=htf_dir,
        trade_setup=trade_setup,
        htf_bias_summary=bias_eval.reason,
    )

    # 5. Stage 3 — Confirmation on the last CLOSED 5m candle.
    confirmed, invalidated, confirm_detail = evaluate_5m_confirmation(
        confirm_candles,
        direction=htf_dir,
        entry_low=armed_snap.entry_low,
        entry_high=armed_snap.entry_high,
        stop_loss=armed_snap.stop_loss,
        cutoff=setup_close_time,
    )
    if invalidated:
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="expired" if expired_armed is not None else "wait",
            expired_armed_setup=expired_armed,
            failed_stage="setup",
            failed_reason=confirm_detail,
        )

    if confirmed:
        item = armed_snap.to_scanned_item(confirmation_detail=confirm_detail)
        return MtfSymbolDecision(
            canonical_symbol=canonical_symbol,
            venue=venue,
            setup_timeframe=setup_tf,
            setup_close_time=setup_close_time,
            outcome="ready",
            promoted_from_armed=False,
            direction=htf_dir,
            item=item,
            trade_setup=trade_setup,
            armed_setup=armed_snap,
            expired_armed_setup=expired_armed,
        )

    return MtfSymbolDecision(
        canonical_symbol=canonical_symbol,
        venue=venue,
        setup_timeframe=setup_tf,
        setup_close_time=setup_close_time,
        outcome="armed",
        direction=htf_dir,
        trade_setup=trade_setup,
        armed_setup=armed_snap,
        expired_armed_setup=expired_armed,
        failed_stage="confirm",
        failed_reason=f"Setup 15m ARMED ({htf_dir.upper()}); {confirm_detail}",
    )


def build_armed_setup_dict(armed: ArmedSetupSnapshot) -> dict[str, Any]:
    """Serialize `ArmedSetupSnapshot` to JSON-compatible dict for `alert_symbol_states`."""
    return armed.model_dump(mode="json")
