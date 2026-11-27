from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from crypto_chatbot import cli
from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.config import Settings
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.gate_candles import normalize_gate_candlesticks
from crypto_chatbot.market_data.resampling import (
    compare_resampled_with_native,
    resample_candles,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.market import MarketCandle, MarketType, QualityFlag, Timeframe


def _make_1h_candles(
    count: int,
    *,
    start: datetime | None = None,
) -> list[MarketCandle]:
    symbol = resolve_default_perp_symbol("BTC")
    t0 = start or datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    received = t0 + timedelta(hours=count + 1)
    candles: list[MarketCandle] = []
    for i in range(count):
        open_t = t0 + timedelta(hours=i)
        close_t = open_t + timedelta(hours=1)
        base_p = 60000.0 + i * 5.0
        candles.append(
            MarketCandle(
                provider="gate_rest",
                venue="gate",
                canonical_symbol=symbol.canonical_symbol,
                provider_symbol="GATE:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                timeframe=Timeframe.H1,
                open_time=open_t,
                close_time=close_t,
                received_time=received,
                available_time=close_t,
                open=base_p,
                high=base_p + 15.0 + (i % 4),
                low=base_p - 12.0 - (i % 3),
                close=base_p + 3.0,
                raw_contracts=10_000.0 + i * 100.0,
                base_volume=1.0 + i * 0.01,
                quote_volume=(1.0 + i * 0.01) * (base_p + 3.0),
                taker_buy_base_volume=0.6 + i * 0.005,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return candles


def test_resample_utc_anchored_complete_buckets_only() -> None:
    # 2 full days (48h) + 2 extra hours -> 48/4 = 12 full 4h bars (incomplete 13th dropped!)
    # and 48/24 = 2 full 1d bars
    base_50 = _make_1h_candles(50)
    b4 = resample_candles(base_50, target_timeframe=Timeframe.H4)
    assert len(b4.candles) == 12
    assert b4.candles[0].open_time == datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    assert b4.candles[0].open == base_50[0].open
    assert b4.candles[0].high == max(c.high for c in base_50[:4])
    assert b4.candles[0].low == min(c.low for c in base_50[:4])
    assert b4.candles[0].close == base_50[3].close
    assert b4.candles[0].base_volume == pytest.approx(sum(c.base_volume for c in base_50[:4]))
    assert b4.candles[0].taker_buy_base_volume == pytest.approx(
        sum(c.taker_buy_base_volume or 0.0 for c in base_50[:4])
    )

    bd = resample_candles(base_50, target_timeframe=Timeframe.D1)
    assert len(bd.candles) == 2
    assert bd.candles[0].open_time == datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    assert bd.candles[0].close == base_50[23].close
    assert bd.candles[1].open_time == datetime(2026, 11, 2, 0, 0, tzinfo=UTC)


def test_resample_prefix_invariance() -> None:
    full = _make_1h_candles(60)
    as_of_48 = full[47].close_time
    prefix_48 = full[:48]

    res_prefix = resample_candles(prefix_48, target_timeframe=Timeframe.H4, as_of=as_of_48)
    res_with_future = resample_candles(full, target_timeframe=Timeframe.H4, as_of=as_of_48)
    assert res_prefix.candles == res_with_future.candles
    assert len(res_prefix.candles) == 12


def test_parity_report_flags_resample_mismatch_over_5_bps() -> None:
    base = _make_1h_candles(24)  # 6 x 4h bars
    resampled = resample_candles(base, target_timeframe=Timeframe.H4).candles
    native_exact = [c.model_copy(update={"provider": "gate_rest"}) for c in resampled]

    ok_summary = compare_resampled_with_native(
        resampled,
        native_exact,
        base_timeframe=Timeframe.H1,
        target_timeframe=Timeframe.H4,
        max_bars=200,
    )
    assert ok_summary.matched_bars == 6
    assert ok_summary.max_close_diff_bps == 0.0
    assert ok_summary.mismatch_flagged is False
    assert QualityFlag.RESAMPLE_MISMATCH not in ok_summary.quality_flags

    # Perturb one native close by +10 bps (> 5 bps threshold)
    perturbed = list(native_exact)
    perturbed[-1] = perturbed[-1].model_copy(
        update={"close": perturbed[-1].close * 1.001, "high": perturbed[-1].high * 1.002}
    )
    bad_summary = compare_resampled_with_native(
        resampled,
        perturbed,
        base_timeframe=Timeframe.H1,
        target_timeframe=Timeframe.H4,
        max_bars=200,
    )
    assert bad_summary.mismatch_flagged is True
    assert bad_summary.max_close_diff_bps > 5.0
    assert QualityFlag.RESAMPLE_MISMATCH in bad_summary.quality_flags


@pytest.mark.asyncio
async def test_analysis_service_resample_from_1h_builds_4h_and_1d() -> None:
    # 24 * 120 = 2880 hours (enough for 120d 1d + 720x 4h bars)
    base = _make_1h_candles(24 * 110)

    class _Fake1hProvider:
        def __init__(self) -> None:
            self.tfs_called: list[Timeframe] = []

        async def get_closed_ohlcv(
            self,
            symbol: str,
            timeframe: Timeframe,
            limit: int = DEFAULT_CANDLE_LIMIT,
        ) -> OhlcvBatch:
            del symbol
            self.tfs_called.append(timeframe)
            if timeframe in (Timeframe.H4, Timeframe.D1):
                raise AssertionError(f"expected H1 fetch when resample_from=1h, got {timeframe}")
            if timeframe is not Timeframe.H1:
                raise KeyError(timeframe)
            selected = tuple(base[-min(len(base), limit) :])
            return OhlcvBatch(
                metadata=normalize_gate_candlesticks(
                    [],
                    symbol=resolve_default_perp_symbol("BTC"),
                    provider_symbol="GATE:BTCUSDT.P",
                    timeframe=Timeframe.H1,
                    quanto_multiplier=Decimal("0.0001"),
                    received_time=base[-1].received_time,
                ).metadata,
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                timeframe=Timeframe.H1,
                received_time=base[-1].received_time,
                candles=selected,
            )

        async def close(self) -> None:
            return None

    prov = _Fake1hProvider()
    service = AnalysisService(provider=prov, resample_from=Timeframe.H1)
    snap = await service.analyze(
        "BTC",
        Timeframe.H4,
        limit=DEFAULT_CANDLE_LIMIT,
        include_mtf=True,
        include_derivatives=False,
        persist=False,
    )
    assert snap.timeframe is Timeframe.H4
    assert snap.technical is not None
    assert Timeframe.H1 in prov.tfs_called
    assert Timeframe.H4 not in prov.tfs_called
    assert Timeframe.D1 not in prov.tfs_called


def test_settings_resample_from_validation() -> None:
    s = Settings(
        _env_file=None,
        telegram_bot_token=SecretStr("123:abc"),
        gemini_api_key=SecretStr("gem-key"),
        gemini_model="gemini-2.5-flash",
        database_url=SecretStr("postgresql+psycopg://u:p@localhost:5432/db"),
        allowed_telegram_user_ids=(111,),
        resample_from="1H",
    )
    assert s.resample_from == "1h"


def test_cli_resample_parity_with_offline_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _make_1h_candles(800)
    b4 = resample_candles(base, target_timeframe=Timeframe.H4)
    bd = resample_candles(base, target_timeframe=Timeframe.D1)

    class _MockClient:
        async def get_quanto_multiplier(self, contract: str) -> Decimal:
            del contract
            return Decimal("0.0001")

        async def fetch_candlesticks(
            self,
            contract: str,
            *,
            interval: str,
            limit: int | None = None,
            from_time: int | None = None,
            to_time: int | None = None,
        ) -> list[dict[str, object]]:
            del contract, from_time, to_time, limit
            return []

        async def close(self) -> None:
            return None

    class _MockProvider:
        def __init__(self, **kw: object) -> None:
            del kw

        async def get_closed_ohlcv_history(
            self, symbol: str, tf: Timeframe, limit: int = 100
        ) -> OhlcvBatch:
            del symbol, limit
            if tf is Timeframe.H1:
                return OhlcvBatch(
                    metadata=b4.metadata.model_copy(update={"provider": "gate_rest"}),
                    canonical_symbol=b4.canonical_symbol,
                    provider_symbol=b4.provider_symbol,
                    timeframe=Timeframe.H1,
                    received_time=base[-1].received_time,
                    candles=tuple(base),
                )
            if tf is Timeframe.H4:
                return b4
            return bd

        async def close(self) -> None:
            return None

    import crypto_chatbot.market_data.gate_candles as gc

    monkeypatch.setattr(gc, "GateFuturesMarketDataProvider", _MockProvider)
    # Keep httpx unused cleanly
    assert httpx is not None

    out_path = tmp_path / "parity.json"
    rc = cli.main(
        ["resample-parity", "--symbol", "BTC", "--bars", "200", "--out", str(out_path), "--json"]
    )
    assert rc == 0
    report_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert report_data["targets"][0]["max_close_diff_bps"] == 0.0
    assert report_data["targets"][0]["mismatch_flagged"] is False
    assert "BTC/USDT:USDT" in capsys.readouterr().out
