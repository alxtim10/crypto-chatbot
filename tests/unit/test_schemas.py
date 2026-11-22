from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from crypto_chatbot.schemas import (
    AnalysisResponse,
    AnalysisSnapshot,
    AnalyzeCommandArgs,
    CompareCommandArgs,
    Fact,
    FactorDirection,
    FactorItem,
    InputRange,
    LlmAnalysisNarration,
    MarketCandle,
    MarketStructure,
    MarketType,
    MtfRsiAlignment,
    NarrationStatus,
    ProviderMetadata,
    QualityFlag,
    QualityStatus,
    Regime,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
    Timeframe,
    WatchlistAction,
    WatchlistCommandArgs,
)

WIB = timezone(timedelta(hours=7))
SHA256_SAMPLE = "a" * 64


def _valid_candle_kwargs(**overrides: Any) -> dict[str, Any]:
    open_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    close_time = open_time + timedelta(hours=4)
    base: dict[str, Any] = {
        "provider": "ccxt",
        "venue": "binance",
        "canonical_symbol": "BTC/USDT",
        "provider_symbol": "BINANCE:BTCUSDT",
        "market_type": MarketType.SPOT,
        "timeframe": Timeframe.H4,
        "open_time": open_time,
        "close_time": close_time,
        "received_time": close_time + timedelta(seconds=5),
        "available_time": close_time + timedelta(seconds=5),
        "open": 90_000.0,
        "high": 91_500.0,
        "low": 89_500.0,
        "close": 91_000.0,
        "base_volume": 1234.5,
        "quote_volume": 111_000_000.0,
        "base_unit": "BTC",
        "quote_currency": "USDT",
        "is_closed": True,
    }
    base.update(overrides)
    return base


def _valid_snapshot(**overrides: Any) -> AnalysisSnapshot:
    as_of = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    facts = (
        Fact(
            fact_id="F1",
            label="Last close",
            value=91_000.0,
            unit="USDT",
            display_format="price",
            source="binance:BTC/USDT:4h",
            as_of=as_of,
        ),
        Fact(
            fact_id="F2",
            label="EMA 200",
            value=85_000.0,
            unit="USDT",
            display_format="price",
            source="binance:BTC/USDT:4h",
            as_of=as_of,
        ),
    )
    payload: dict[str, Any] = {
        "snapshot_id": "snap-001",
        "provider": "ccxt",
        "venue": "binance",
        "canonical_symbol": "BTC/USDT",
        "provider_symbol": "BINANCE:BTCUSDT",
        "market_type": MarketType.SPOT,
        "timeframe": Timeframe.H4,
        "as_of": as_of,
        "input_range": InputRange(
            first_open_time=as_of - timedelta(hours=4 * 1000),
            last_open_time=as_of - timedelta(hours=4),
            candle_count=1000,
            sha256=SHA256_SAMPLE,
        ),
        "calc_version": "1.0.0",
        "indicator_parameters": {"ema_fast": 20, "ema_mid": 50, "ema_slow": 200, "rsi": 14},
        "technical": TechnicalMetrics(
            last_close=91_000.0,
            ema_200=85_000.0,
            rsi_14=61.5,
            mtf_rsi_alignment=MtfRsiAlignment.ALIGNED_BULLISH,
            prev_day_high=92_000.0,
            prev_day_low=88_000.0,
            prev_day_close=90_500.0,
        ),
        "facts": facts,
        "regime": Regime.TRENDING_UP,
        "market_structure": MarketStructure.BULLISH,
        "events": (
            StructureEvent(
                event_type=StructureEventType.SWING_HIGH,
                event_time=as_of - timedelta(hours=16),
                confirmation_time=as_of - timedelta(hours=4),
                level=91_500.0,
            ),
        ),
        "factors": (
            FactorItem(
                rule_id="price_above_ema200",
                description="Close is above EMA 200",
                direction=FactorDirection.SUPPORTS_BULLISH,
                fact_ids=("F1", "F2"),
            ),
        ),
        "quality_status": QualityStatus.READY,
        "quality_flags": (),
        "limitations": ("Spot Binance OHLCV only.",),
    }
    payload.update(overrides)
    return AnalysisSnapshot(**payload)


def test_candle_normalizes_non_utc_timestamps_to_utc() -> None:
    open_wib = datetime(2026, 11, 1, 7, 0, tzinfo=WIB)
    close_wib = datetime(2026, 11, 1, 11, 0, tzinfo=WIB)
    candle = MarketCandle(
        **_valid_candle_kwargs(
            open_time=open_wib,
            close_time=close_wib,
            received_time=close_wib + timedelta(seconds=10),
            available_time=close_wib + timedelta(seconds=10),
        )
    )

    assert candle.open_time == datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    assert candle.open_time.tzinfo is UTC
    assert candle.close_time == datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    assert candle.close_time.tzinfo is UTC


@pytest.mark.parametrize(
    "naive_value",
    [
        datetime(2026, 11, 1, 0, 0),
        "2026-11-01T00:00:00",
    ],
)
def test_candle_rejects_naive_timestamps(naive_value: Any) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        MarketCandle(**_valid_candle_kwargs(open_time=naive_value))


def test_candle_rejects_mismatched_close_time() -> None:
    open_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    binance_minus_1ms = open_time + timedelta(hours=4) - timedelta(milliseconds=1)

    with pytest.raises(ValidationError, match="close_time"):
        MarketCandle(**_valid_candle_kwargs(close_time=binance_minus_1ms))


def test_candle_rejects_closed_before_close_time() -> None:
    open_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    close_time = open_time + timedelta(hours=4)

    with pytest.raises(ValidationError, match="is_closed"):
        MarketCandle(
            **_valid_candle_kwargs(
                received_time=close_time - timedelta(seconds=1),
                is_closed=True,
            )
        )


def test_candle_rejects_available_time_before_close_time() -> None:
    open_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    close_time = open_time + timedelta(hours=4)

    with pytest.raises(ValidationError, match="available_time"):
        MarketCandle(**_valid_candle_kwargs(available_time=close_time - timedelta(seconds=1)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"high": 90_500.0},  # high < close (91_000)
        {"high": 89_900.0},  # high < open (90_000)
        {"low": 90_500.0},  # low > open (90_000)
        {"low": 91_200.0},  # low > close (91_000)
        {"open": 0.0},
        {"close": -10.0},
        {"base_volume": -1.0},
        {"high": float("nan")},
        {"close": float("inf")},
    ],
)
def test_candle_rejects_invalid_ohlc_and_volume(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MarketCandle(**_valid_candle_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"market_type": MarketType.SPOT, "canonical_symbol": "BTC/USDT:USDT"},
        {"market_type": MarketType.SPOT, "provider_symbol": "BINANCE:BTCUSDT.P"},
        {"market_type": MarketType.PERPETUAL, "canonical_symbol": "BTC/USDT"},
        {"timeframe": "5m"},
        {"market_type": "futures"},
        {"quality_flags": ("UNKNOWN_FLAG",)},
    ],
)
def test_candle_rejects_symbol_mixing_and_invalid_enums(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MarketCandle(**_valid_candle_kwargs(**overrides))


def test_candle_and_provider_metadata_round_trip() -> None:
    candle = MarketCandle(**_valid_candle_kwargs())
    assert MarketCandle.model_validate_json(candle.model_dump_json()) == candle

    meta = ProviderMetadata(
        provider="ccxt",
        venue="binance",
        market_type=MarketType.SPOT,
        base_unit="BTC",
        quote_currency="USDT",
    )
    assert ProviderMetadata.model_validate_json(meta.model_dump_json()) == meta


def test_snapshot_validation_and_round_trip() -> None:
    snapshot = _valid_snapshot()
    assert AnalysisSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot

    insufficient = AnalysisSnapshot(
        snapshot_id="snap-err",
        provider="ccxt",
        venue="binance",
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        market_type=MarketType.SPOT,
        timeframe=Timeframe.H4,
        as_of=datetime(2026, 11, 1, 4, 0, tzinfo=UTC),
        calc_version="1.0.0",
        quality_status=QualityStatus.INSUFFICIENT_DATA,
        quality_flags=(QualityFlag.STALE,),
        limitations=("Latest closed candle is stale.",),
    )
    assert AnalysisSnapshot.model_validate_json(insufficient.model_dump_json()) == insufficient


def test_snapshot_rejects_blocking_flags_on_ready() -> None:
    with pytest.raises(ValidationError, match="blocking flags"):
        _valid_snapshot(quality_flags=(QualityFlag.STALE,))


def test_snapshot_rejects_lookahead_event_confirmation() -> None:
    as_of = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    future_event = StructureEvent(
        event_type=StructureEventType.SWING_HIGH,
        event_time=as_of - timedelta(hours=4),
        confirmation_time=as_of + timedelta(hours=4),
        level=92_000.0,
    )
    with pytest.raises(ValidationError, match="lookahead"):
        _valid_snapshot(events=(future_event,))


def test_snapshot_rejects_unknown_factor_fact_id() -> None:
    bad_factor = FactorItem(
        rule_id="bad_ref",
        description="References missing fact",
        direction=FactorDirection.NEUTRAL,
        fact_ids=("F99",),
    )
    with pytest.raises(ValidationError, match="unknown fact_ids"):
        _valid_snapshot(factors=(bad_factor,))


def test_factor_item_cannot_be_probabilistic() -> None:
    with pytest.raises(ValidationError):
        FactorItem.model_validate(
            {
                "rule_id": "rule1",
                "description": "Not a probability",
                "direction": "neutral",
                "fact_ids": ("F1",),
                "is_probabilistic": True,
            }
        )


def test_llm_narration_and_analysis_response_round_trip() -> None:
    snapshot = _valid_snapshot()
    narration = LlmAnalysisNarration(
        summary="Harga terakhir {F1} berada di atas EMA 200 ({F2}).",
        supporting_factors=("Harga di atas EMA 200 ({F2}).",),
        contradicting_factors=(),
        bullish_scenario="Selama bertahan di atas {F2}, tren naik tetap terjaga.",
        bearish_scenario="Penutupan kembali di bawah {F2} melemahkan struktur saat ini.",
        limitations=("Data berasal dari Binance spot saja.",),
        fact_ids=("F1", "F2"),
    )
    response = AnalysisResponse(
        snapshot=snapshot,
        narration_status=NarrationStatus.GENERATED,
        narration=narration,
    )

    assert AnalysisResponse.model_validate_json(response.model_dump_json()) == response


def test_llm_narration_rejects_market_condition_and_undeclared_placeholders() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LlmAnalysisNarration.model_validate(
            {
                "summary": "Ringkasan {F1}.",
                "market_condition": "TRENDING_UP",
                "bullish_scenario": "Skenario naik.",
                "bearish_scenario": "Skenario turun.",
                "fact_ids": ["F1"],
            }
        )

    with pytest.raises(ValidationError, match="fact_ids"):
        LlmAnalysisNarration(
            summary="Harga {F1} dan EMA {F2}.",
            bullish_scenario="Skenario naik.",
            bearish_scenario="Skenario turun.",
            fact_ids=("F1",),
        )


def test_analysis_response_rejects_unknown_snapshot_fact_id() -> None:
    snapshot = _valid_snapshot()
    narration = LlmAnalysisNarration(
        summary="Harga {F9}.",
        bullish_scenario="Skenario naik.",
        bearish_scenario="Skenario turun.",
        fact_ids=("F9",),
    )
    with pytest.raises(ValidationError, match="unknown snapshot fact_ids"):
        AnalysisResponse(
            snapshot=snapshot,
            narration_status=NarrationStatus.GENERATED,
            narration=narration,
        )


def test_command_argument_schemas() -> None:
    analyze = AnalyzeCommandArgs(symbol="btc", timeframe="4H")  # type: ignore[arg-type]
    assert analyze.symbol == "BTC/USDT"
    assert analyze.timeframe is Timeframe.H4
    assert AnalyzeCommandArgs.model_validate_json(analyze.model_dump_json()) == analyze

    compare = CompareCommandArgs(symbols=("btc", "ETH/USDT"), timeframe="1d")  # type: ignore[arg-type]
    assert compare.symbols == ("BTC/USDT", "ETH/USDT")
    assert compare.timeframe is Timeframe.D1
    assert CompareCommandArgs.model_validate_json(compare.model_dump_json()) == compare

    wl_add = WatchlistCommandArgs(action="ADD", symbol="sol")  # type: ignore[arg-type]
    assert wl_add.action is WatchlistAction.ADD
    assert wl_add.symbol == "SOL/USDT"
    assert WatchlistCommandArgs.model_validate_json(wl_add.model_dump_json()) == wl_add

    wl_list = WatchlistCommandArgs()
    assert wl_list.action is WatchlistAction.LIST
    assert wl_list.symbol is None

    alt_m15 = AnalyzeCommandArgs(symbol="doge", timeframe="15m")  # type: ignore[arg-type]
    assert alt_m15.symbol == "DOGE/USDT"
    assert alt_m15.timeframe is Timeframe.M15


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (AnalyzeCommandArgs, {"symbol": "BTC perpetual", "timeframe": "4h"}),
        (AnalyzeCommandArgs, {"symbol": "BTC", "timeframe": "5m"}),
        (AnalyzeCommandArgs, {"symbol": "BTC/EUR", "timeframe": "4h"}),
        (CompareCommandArgs, {"symbols": ("BTC", "BTC/USDT"), "timeframe": "1d"}),
        (CompareCommandArgs, {"symbols": ("BTC",), "timeframe": "1d"}),
        (WatchlistCommandArgs, {"action": "add"}),
        (WatchlistCommandArgs, {"action": "list", "symbol": "BTC"}),
    ],
)
def test_command_argument_schemas_reject_invalid_inputs(
    cls: type[AnalyzeCommandArgs | CompareCommandArgs | WatchlistCommandArgs],
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        cls.model_validate(kwargs)
