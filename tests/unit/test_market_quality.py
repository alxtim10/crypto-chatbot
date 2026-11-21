from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from crypto_chatbot.market_data.base import (
    DEFAULT_BACKFILL_AVAILABILITY_LAG,
    DEFAULT_FRESHNESS_GRACE,
    MIN_DEGRADED_HISTORY_CANDLES,
    MIN_FULL_HISTORY_CANDLES,
)
from crypto_chatbot.market_data.quality import (
    determine_quality_status,
    is_candle_stale,
    normalize_ohlcv_rows,
)
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.market import QualityFlag, QualityStatus, Timeframe

WIB = timezone(timedelta(hours=7))


def _row(
    open_time: datetime,
    *,
    open_p: float = 100.0,
    high_p: float = 105.0,
    low_p: float = 95.0,
    close_p: float = 102.0,
    vol: float = 50.0,
) -> list[float]:
    return [open_time.timestamp() * 1000.0, open_p, high_p, low_p, close_p, vol]


def _make_series(
    count: int,
    *,
    end_close_time: datetime,
    timeframe: Timeframe = Timeframe.H4,
) -> list[list[float]]:
    start_open = end_close_time - count * timeframe.duration
    return [
        _row(
            start_open + idx * timeframe.duration,
            open_p=100.0 + idx,
            high_p=105.0 + idx,
            low_p=95.0 + idx,
            close_p=102.0 + idx,
            vol=50.0 + idx,
        )
        for idx in range(count)
    ]


def test_excludes_open_candle_and_populates_metadata() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    received = datetime(2026, 11, 1, 13, 0, tzinfo=UTC)
    rows = [
        _row(datetime(2026, 11, 1, 4, 0, tzinfo=UTC)),
        _row(datetime(2026, 11, 1, 8, 0, tzinfo=UTC)),
        # Open candle (12:00..16:00 UTC) when received_time is 13:00 UTC
        _row(datetime(2026, 11, 1, 12, 0, tzinfo=UTC)),
    ]

    batch = normalize_ohlcv_rows(
        rows,
        symbol=symbol,
        timeframe=Timeframe.H4,
        received_time=received,
    )

    assert len(batch.candles) == 2
    assert batch.excluded_open_count == 1
    assert batch.invalid_row_count == 0
    assert QualityFlag.INVALID_OHLC not in batch.quality_flags
    assert QualityFlag.STALE not in batch.quality_flags
    assert batch.metadata.base_unit == "BTC"
    assert batch.metadata.quote_currency == "USDT"
    for candle in batch.candles:
        assert candle.is_closed is True
        assert candle.base_unit == "BTC"
        assert candle.quote_currency == "USDT"
        assert candle.available_time == received


def test_available_time_live_vs_backfill() -> None:
    symbol = resolve_mvp_spot_symbol("ETH")
    open_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    close_time = open_time + Timeframe.H1.duration
    received = close_time + timedelta(seconds=12)

    live_batch = normalize_ohlcv_rows(
        [_row(open_time)],
        symbol=symbol,
        timeframe=Timeframe.H1,
        received_time=received,
        is_backfill=False,
    )
    assert live_batch.candles[0].available_time == received

    backfill_batch = normalize_ohlcv_rows(
        [_row(open_time)],
        symbol=symbol,
        timeframe=Timeframe.H1,
        received_time=received,
        is_backfill=True,
    )
    assert (
        backfill_batch.candles[0].available_time == close_time + DEFAULT_BACKFILL_AVAILABILITY_LAG
    )
    assert QualityFlag.STALE not in backfill_batch.quality_flags


@pytest.mark.parametrize(
    ("timeframe", "elapsed", "expected_stale"),
    [
        (Timeframe.H1, timedelta(hours=1) + DEFAULT_FRESHNESS_GRACE, False),
        (Timeframe.H1, timedelta(hours=1) + DEFAULT_FRESHNESS_GRACE + timedelta(seconds=1), True),
        (Timeframe.H4, timedelta(hours=4) + DEFAULT_FRESHNESS_GRACE, False),
        (Timeframe.H4, timedelta(hours=4) + DEFAULT_FRESHNESS_GRACE + timedelta(seconds=1), True),
        (Timeframe.D1, timedelta(days=1) + DEFAULT_FRESHNESS_GRACE, False),
        (Timeframe.D1, timedelta(days=1) + DEFAULT_FRESHNESS_GRACE + timedelta(seconds=1), True),
    ],
)
def test_staleness_boundary_1x_plus_grace(
    timeframe: Timeframe,
    elapsed: timedelta,
    expected_stale: bool,
) -> None:
    close_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    now = close_time + elapsed
    assert is_candle_stale(close_time, timeframe, now) is expected_stale


def test_identical_duplicates_handled_deterministically() -> None:
    symbol = resolve_mvp_spot_symbol("SOL")
    t1 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 11, 1, 1, 0, tzinfo=UTC)
    received = datetime(2026, 11, 1, 2, 1, tzinfo=UTC)

    batch = normalize_ohlcv_rows(
        [_row(t2, close_p=103.0), _row(t1, close_p=101.0), _row(t1, close_p=101.0)],
        symbol=symbol,
        timeframe=Timeframe.H1,
        received_time=received,
    )

    assert [c.open_time for c in batch.candles] == [t1, t2]
    assert batch.duplicate_count == 1
    assert QualityFlag.DUPLICATE in batch.quality_flags
    assert QualityFlag.INVALID_OHLC not in batch.quality_flags


def test_conflicting_duplicates_flag_invalid_ohlc_and_stay_order_independent() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    t1 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    received = datetime(2026, 11, 1, 1, 1, tzinfo=UTC)

    row_a = _row(t1, close_p=101.0)
    row_b = _row(t1, close_p=104.0)

    batch_ab = normalize_ohlcv_rows(
        [row_a, row_b],
        symbol=symbol,
        timeframe=Timeframe.H1,
        received_time=received,
    )
    batch_ba = normalize_ohlcv_rows(
        [row_b, row_a],
        symbol=symbol,
        timeframe=Timeframe.H1,
        received_time=received,
    )

    assert batch_ab.candles == batch_ba.candles
    assert batch_ab.duplicate_count == 1
    assert QualityFlag.DUPLICATE in batch_ab.quality_flags
    assert QualityFlag.INVALID_OHLC in batch_ab.quality_flags


def test_gap_detection_flags_without_interpolating() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
    received = datetime(2026, 11, 1, 12, 1, tzinfo=UTC)

    batch = normalize_ohlcv_rows(
        [_row(t0), _row(t2)],
        symbol=symbol,
        timeframe=Timeframe.H4,
        received_time=received,
    )

    assert len(batch.candles) == 2
    assert QualityFlag.GAP in batch.quality_flags
    assert batch.gap_intervals == ((datetime(2026, 11, 1, 4, 0, tzinfo=UTC), t2),)


def test_misaligned_candle_stride_flags_invalid_ohlc() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    t_half = datetime(2026, 11, 1, 2, 0, tzinfo=UTC)
    received = datetime(2026, 11, 1, 6, 1, tzinfo=UTC)

    batch = normalize_ohlcv_rows(
        [_row(t0), _row(t_half)],
        symbol=symbol,
        timeframe=Timeframe.H4,
        received_time=received,
    )

    assert QualityFlag.INVALID_OHLC in batch.quality_flags


def test_invalid_rows_rejected_without_silent_repair() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    received = datetime(2026, 11, 1, 8, 1, tzinfo=UTC)

    rows = [
        _row(t0),
        [t1.timestamp() * 1000.0, 100.0, 90.0, 95.0, 98.0, 10.0],  # high < open
        [t1.timestamp() * 1000.0, float("nan"), 105.0, 95.0, 100.0, 10.0],
        [t1.timestamp() * 1000.0, 100.0, 105.0],  # short row
        _row(t1),
    ]

    batch = normalize_ohlcv_rows(
        rows,
        symbol=symbol,
        timeframe=Timeframe.H4,
        received_time=received,
    )

    assert batch.invalid_row_count == 3
    assert len(batch.candles) == 2
    assert QualityFlag.INVALID_OHLC in batch.quality_flags


def test_determine_quality_status_and_full_600_candle_batch() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    received = datetime(2026, 11, 1, 12, 1, tzinfo=UTC)
    rows = _make_series(
        MIN_FULL_HISTORY_CANDLES,
        end_close_time=datetime(2026, 11, 1, 12, 0, tzinfo=UTC),
    )

    batch = normalize_ohlcv_rows(
        rows,
        symbol=symbol,
        timeframe=Timeframe.H4,
        received_time=received,
    )

    assert len(batch.candles) == MIN_FULL_HISTORY_CANDLES
    assert batch.quality_flags == ()
    assert determine_quality_status(len(batch.candles), batch.quality_flags) is QualityStatus.READY

    assert (
        determine_quality_status(MIN_DEGRADED_HISTORY_CANDLES - 1, ())
        is QualityStatus.INSUFFICIENT_DATA
    )
    assert (
        determine_quality_status(MIN_DEGRADED_HISTORY_CANDLES, (QualityFlag.SHORT_HISTORY,))
        is QualityStatus.DEGRADED
    )
    assert (
        determine_quality_status(MIN_FULL_HISTORY_CANDLES, (QualityFlag.ZERO_VOLUME,))
        is QualityStatus.DEGRADED
    )
    for blocking in (
        QualityFlag.STALE,
        QualityFlag.GAP,
        QualityFlag.INVALID_OHLC,
        QualityFlag.SYMBOL_MISMATCH,
        QualityFlag.PROVIDER_ERROR,
    ):
        assert (
            determine_quality_status(MIN_FULL_HISTORY_CANDLES, (blocking,))
            is QualityStatus.INSUFFICIENT_DATA
        )


def test_rejects_naive_datetimes_and_normalizes_non_utc() -> None:
    symbol = resolve_mvp_spot_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    received_wib = datetime(2026, 11, 1, 8, 1, tzinfo=WIB)  # 01:01 UTC

    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_ohlcv_rows(
            [_row(t0)],
            symbol=symbol,
            timeframe=Timeframe.H1,
            received_time=datetime(2026, 11, 1, 1, 1),
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        is_candle_stale(datetime(2026, 11, 1, 1, 0), Timeframe.H1, received_wib)

    batch = normalize_ohlcv_rows(
        [_row(t0)],
        symbol=symbol,
        timeframe=Timeframe.H1,
        received_time=received_wib,
    )
    assert batch.received_time == datetime(2026, 11, 1, 1, 1, tzinfo=UTC)
