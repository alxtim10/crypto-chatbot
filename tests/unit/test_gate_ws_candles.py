from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from crypto_chatbot.config import Settings
from crypto_chatbot.market_data.gate_candles import (
    GateFuturesMarketDataProvider,
    GateFuturesRestClient,
)
from crypto_chatbot.market_data.gate_ws_candles import (
    GateClosedCandleStreamAssembler,
    GateWsCandlesRunner,
    build_gate_ws_subscribe_message,
)
from crypto_chatbot.schemas.market import MarketCandle, Timeframe


def _ws_frame(
    *,
    t: int,
    o: str,
    h: str,
    l_price: str,
    c: str,
    v: int,
    a: str,
    w: bool,
    n: str = "1m_BTC_USDT",
) -> dict[str, Any]:
    return {
        "channel": "futures.candlesticks",
        "event": "update",
        "result": [
            {
                "t": t,
                "o": o,
                "h": h,
                "l": l_price,
                "c": c,
                "v": v,
                "a": a,
                "n": n,
                "w": w,
            }
        ],
    }


def test_ws_assembler_suppresses_forming_and_emits_closed_once() -> None:
    assembler = GateClosedCandleStreamAssembler()
    mults = {"BTC_USDT": Decimal("0.0001")}
    t0 = 1795774800
    rx0 = datetime.fromtimestamp(t0 + 20, tz=UTC)

    # 1. Forming bar (w=False) must NOT emit
    out1 = assembler.ingest_payload(
        _ws_frame(
            t=t0,
            o="61630.5",
            h="61642.9",
            l_price="61607.5",
            c="61611.5",
            v=74005,
            a="456040.56613",
            w=False,
        ),
        multipliers=mults,
        received_time=rx0,
    )
    assert out1 == []

    # 2. Explicit window close (w=True) emits closed candle with unit normalization
    rx1 = datetime.fromtimestamp(t0 + 60, tz=UTC)
    out2 = assembler.ingest_payload(
        _ws_frame(
            t=t0,
            o="61630.5",
            h="61650.0",
            l_price="61607.5",
            c="61645.0",
            v=80000,
            a="493000.12",
            w=True,
        ),
        multipliers=mults,
        received_time=rx1,
    )
    assert len(out2) == 1
    c = out2[0]
    assert c.provider == "gate_ws"
    assert c.venue == "gate"
    assert c.timeframe is Timeframe.M1
    assert c.raw_contracts == 80000.0
    assert c.base_volume == pytest.approx(8.0)  # 80000 * 0.0001
    assert c.quote_volume == pytest.approx(493000.12)
    assert c.available_time == c.close_time

    # 3. Duplicate w=True frame for same t0 is deduplicated
    out_dup = assembler.ingest_payload(
        _ws_frame(
            t=t0,
            o="61630.5",
            h="61650.0",
            l_price="61607.5",
            c="61645.0",
            v=80000,
            a="493000.12",
            w=True,
        ),
        multipliers=mults,
        received_time=rx1,
    )
    assert out_dup == []

    # 4. Forming t1 followed by rollover t2 closes t1 without needing w=True on t1
    t1 = t0 + 60
    t2 = t0 + 120
    assert (
        assembler.ingest_payload(
            _ws_frame(
                t=t1,
                o="61645.0",
                h="61660.0",
                l_price="61640.0",
                c="61655.0",
                v=10000,
                a="61655.0",
                w=False,
            ),
            multipliers=mults,
            received_time=datetime.fromtimestamp(t1 + 45, tz=UTC),
        )
        == []
    )
    out_rollover = assembler.ingest_payload(
        _ws_frame(
            t=t2,
            o="61655.0",
            h="61665.0",
            l_price="61650.0",
            c="61658.0",
            v=1000,
            a="6165.8",
            w=False,
        ),
        multipliers=mults,
        received_time=datetime.fromtimestamp(t2 + 1, tz=UTC),
    )
    assert len(out_rollover) == 1
    assert out_rollover[0].open_time == datetime.fromtimestamp(t1, tz=UTC)
    assert out_rollover[0].base_volume == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_ws_runner_subscribes_backfills_gaps_and_persists_closed() -> None:
    t0 = 1795774800

    def _rest_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v4/futures/usdt/contracts":
            return httpx.Response(
                200,
                json=[{"name": "BTC_USDT", "quanto_multiplier": "0.0001"}],
            )
        if request.url.path == "/api/v4/futures/usdt/candlesticks":
            return httpx.Response(
                200,
                json=[
                    {
                        "t": t0 - 120,
                        "o": "61600",
                        "h": "61610",
                        "l": "61590",
                        "c": "61605",
                        "v": 5000,
                        "sum": "30802.5",
                    },
                    {
                        "t": t0 - 60,
                        "o": "61605",
                        "h": "61620",
                        "l": "61600",
                        "c": "61615",
                        "v": 6000,
                        "sum": "36969.0",
                    },
                ],
            )
        return httpx.Response(404)

    rest_client = GateFuturesRestClient(transport=httpx.MockTransport(_rest_handler))
    now_fixed = datetime.fromtimestamp(t0 + 120, tz=UTC)
    provider = GateFuturesMarketDataProvider(client=rest_client, clock=lambda: now_fixed)
    delivered: list[MarketCandle] = []

    class _FakeWs:
        def __init__(self, frames: Sequence[dict[str, Any]]) -> None:
            self.sent: list[dict[str, Any]] = []
            self._frames = list(frames)

        async def __aenter__(self) -> _FakeWs:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args
            return None

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        def __aiter__(self) -> AsyncIterator[str]:
            async def _gen() -> AsyncIterator[str]:
                for frame in self._frames:
                    yield json.dumps(frame)

            return _gen()

    fake_ws = _FakeWs(
        [
            _ws_frame(
                t=t0,
                o="61630.5",
                h="61642.9",
                l_price="61607.5",
                c="61611.5",
                v=74005,
                a="456040.56613",
                w=True,
            )
        ]
    )

    def _connect(url: str, **kwargs: object) -> _FakeWs:
        del url, kwargs
        return fake_ws

    runner = GateWsCandlesRunner(
        symbols=("BTC",),
        timeframes=("1m",),
        rest_client=rest_client,
        market_provider=provider,
        on_candles=lambda batch: delivered.extend(batch),
        ws_connect=_connect,
        clock=lambda: now_fixed,
    )
    try:
        sub_msg = build_gate_ws_subscribe_message("BTC_USDT", "1m", now=now_fixed)
        assert sub_msg["channel"] == "futures.candlesticks"
        assert sub_msg["payload"] == ["1m", "BTC_USDT"]

        await runner.run_once()
        assert len(fake_ws.sent) == 1
        assert fake_ws.sent[0]["payload"] == ["1m", "BTC_USDT"]
        # 2 closed REST gap-fill candles + 1 WS closed candle
        assert any(c.provider == "gate_rest" for c in delivered)
        assert any(c.provider == "gate_ws" for c in delivered)
        ws_bars = [c for c in delivered if c.provider == "gate_ws"]
        assert len(ws_bars) == 1
        assert ws_bars[0].base_volume == pytest.approx(7.4005)
    finally:
        await runner.stop()
        await provider.close()
        await rest_client.close()


def test_settings_gate_ws_defaults_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "111")
    monkeypatch.setenv("GATE_WS_ENABLED", "true")
    monkeypatch.setenv("GATE_WS_SYMBOLS", "btc, eth")
    monkeypatch.setenv("GATE_WS_TIMEFRAMES", "1m, 5m")
    s = Settings(_env_file=None)
    assert s.gate_ws_enabled is True
    assert s.gate_ws_symbols == ("BTC", "ETH")
    assert s.gate_ws_timeframes == ("1m", "5m")
    # Default flag stays off unless explicitly enabled
    monkeypatch.delenv("GATE_WS_ENABLED", raising=False)
    s_default = Settings(
        _env_file=None,
        telegram_bot_token=SecretStr("123:abc"),
        gemini_api_key=SecretStr("gem-key"),
        gemini_model="gemini-2.5-flash",
        database_url=SecretStr("postgresql+psycopg://u:p@localhost:5432/db"),
        allowed_telegram_user_ids=(111,),
    )
    assert s_default.gate_ws_enabled is False
    assert isinstance(timedelta(minutes=1), timedelta)
