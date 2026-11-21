"""Deterministic analysis service: snapshot assembly, hashing, and recomputation."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from datetime import datetime

from crypto_chatbot.analysis.factors import build_factors
from crypto_chatbot.analysis.failed_breakout import detect_failed_breakouts
from crypto_chatbot.analysis.indicators import (
    CALC_VERSION,
    INDICATOR_PARAMETERS,
    compute_technical_metrics,
)
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    MIN_FULL_HISTORY_CANDLES,
    MarketDataProvider,
    OhlcvBatch,
)
from crypto_chatbot.market_data.quality import (
    determine_quality_status,
    is_candle_stale,
)
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    InputRange,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    ProviderMetadata,
    QualityFlag,
    QualityStatus,
    Timeframe,
)
from crypto_chatbot.storage.repositories import MarketRepository

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
) -> str:
    payload = {
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
    quote_currency: str,
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
                source=source,
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

    return tuple(facts), fact_ids_by_key


def _build_limitations(
    *,
    batch: OhlcvBatch,
    merged_flags: Sequence[QualityFlag],
    status: QualityStatus,
    dropped_mtf: Sequence[Timeframe],
) -> tuple[str, ...]:
    notes: list[str] = []
    if batch.error_message:
        notes.append(batch.error_message)
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
        f"Sumber data terbatas pada spot {batch.metadata.venue} untuk {batch.canonical_symbol}."
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
        indicator_parameters=dict(INDICATOR_PARAMETERS),
        technical=None,
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
        ),
    )


def build_analysis_snapshot(
    batch: OhlcvBatch,
    *,
    mtf_batches: Mapping[Timeframe, OhlcvBatch] | None = None,
    use_available_time: bool = False,
) -> AnalysisSnapshot:
    """Assemble a complete deterministic `AnalysisSnapshot` from a normalized `OhlcvBatch`."""
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

    if mtf_batches is not None:
        mtf_candles = {batch.timeframe: batch.candles}
        for tf, mtf_batch in mtf_batches.items():
            if tf is batch.timeframe:
                continue
            if not mtf_batch.candles or (set(mtf_batch.quality_flags) & _BLOCKING_BATCH_FLAGS):
                dropped_mtf.append(tf)
                continue
            mtf_candles[tf] = mtf_batch.candles
            mtf_hashes[tf.value] = compute_candles_sha256(mtf_batch.candles)

    ind_result = compute_technical_metrics(
        batch.candles,
        mtf_candles=mtf_candles,
        use_available_time=use_available_time,
    )
    regime = classify_regime(ind_result.metrics)
    structure = analyze_market_structure(batch.candles)
    failed_breakouts = detect_failed_breakouts(batch.candles)

    merged_flags = tuple(dict.fromkeys([*batch.quality_flags, *ind_result.quality_flags]))
    final_status = determine_quality_status(len(batch.candles), merged_flags)
    if regime is None or final_status is QualityStatus.INSUFFICIENT_DATA:
        return _insufficient_snapshot(
            batch=batch,
            input_range=input_range,
            as_of=as_of,
            flags=merged_flags,
            dropped_mtf=dropped_mtf,
        )

    recent_events, latest_sh, latest_sl, latest_fb = _select_recent_events(
        structure.pivots,
        failed_breakouts,
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
        quote_currency=batch.metadata.quote_currency,
    )

    recent_cutoff = as_of - 10 * batch.timeframe.duration
    factors = build_factors(
        metrics=ind_result.metrics,
        regime=regime,
        structure=structure,
        failed_breakouts=failed_breakouts,
        recent_cutoff_time=recent_cutoff,
        fact_ids_by_key=fact_ids_by_key,
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
        indicator_parameters=dict(INDICATOR_PARAMETERS),
        technical=ind_result.metrics,
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
    """Orchestrates market-data fetching, deterministic snapshot creation, and persistence."""

    def __init__(
        self,
        provider: MarketDataProvider,
        repository: MarketRepository | None = None,
    ) -> None:
        self._provider = provider
        self._repository = repository

    async def analyze(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        limit: int = DEFAULT_CANDLE_LIMIT,
        include_mtf: bool = True,
        persist: bool = True,
    ) -> AnalysisSnapshot:
        """Fetch market data, build an `AnalysisSnapshot`, and optionally persist it."""
        mapping = resolve_mvp_spot_symbol(symbol)
        primary = await self._provider.get_closed_ohlcv(
            mapping.canonical_symbol,
            timeframe,
            limit=limit,
        )

        mtf_batches: dict[Timeframe, OhlcvBatch] | None = None
        if (
            include_mtf
            and primary.candles
            and determine_quality_status(len(primary.candles), primary.quality_flags)
            is not QualityStatus.INSUFFICIENT_DATA
        ):
            other_tfs = [
                tf for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1) if tf is not timeframe
            ]
            fetched = await asyncio.gather(
                *(
                    self._provider.get_closed_ohlcv(mapping.canonical_symbol, tf, limit=100)
                    for tf in other_tfs
                )
            )
            mtf_batches = dict(zip(other_tfs, fetched, strict=True))

        snapshot = build_analysis_snapshot(primary, mtf_batches=mtf_batches)

        if persist and self._repository is not None:
            if primary.candles:
                self._repository.save_candles(primary.candles)
            if mtf_batches is not None:
                for mtf_batch in mtf_batches.values():
                    if mtf_batch.candles:
                        self._repository.save_candles(mtf_batch.candles)
            self._repository.save_snapshot(snapshot)

        return snapshot

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
        if stored.technical is not None and stored.technical.mtf_rsi_alignment is not None:
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
                    limit=100,
                )
                if mtf_candles:
                    mtf_batches[tf] = _batch_from_stored_candles(
                        mtf_candles,
                        as_of=stored.as_of,
                        is_primary=False,
                    )

        return build_analysis_snapshot(primary_batch, mtf_batches=mtf_batches)
