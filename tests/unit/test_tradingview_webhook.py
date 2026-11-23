"""Phase 10 tests: TradingView AI Agent Lens webhook auth, deduplication, and comparison."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from crypto_chatbot.analysis.service import build_analysis_snapshot
from crypto_chatbot.bot.handlers import SecretRedactingFilter
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.market_data.tradingview import (
    DEFAULT_WEBHOOK_PORT,
    TradingViewWebhookError,
    TradingViewWebhookProcessor,
    TradingViewWebhookServer,
    compare_packet_with_exchange_candles,
    parse_ai_agent_lens_packet,
)
from crypto_chatbot.schemas.market import MarketType, QualityFlag, QualityStatus, Timeframe
from crypto_chatbot.schemas.tradingview import (
    AI_AGENT_LENS_PINNED_COMMIT,
    AI_AGENT_LENS_SCHEMA_VERSION,
    TRADINGVIEW_PUBLISHED_WEBHOOK_IPS,
    TradingViewEventType,
)

TV_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tradingview_ai_agent_lens_sample.json"
)
CANDLE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
ANCHOR_NOW = datetime(2026, 10, 14, 16, 0, 1, tzinfo=UTC)
WEBHOOK_SECRET = "tv-super-secret-body-token-987654"


def _sample_payload(**overrides: Any) -> dict[str, Any]:
    raw = json.loads(TV_FIXTURE.read_text(encoding="utf-8"))
    raw["secret"] = WEBHOOK_SECRET
    raw.update(overrides)
    return dict(raw)


def _body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def test_webhook_rejects_missing_wrong_secret_non_allowlisted_ip_size_and_rate() -> None:
    now_mono = [100.0]
    processor = TradingViewWebhookProcessor(
        shared_secret=SecretStr(WEBHOOK_SECRET),
        max_body_bytes=2048,
        rate_limit_per_minute=2,
        clock=lambda: ANCHOR_NOW,
        monotonic_clock=lambda: now_mono[0],
    )
    allowed_ip = TRADINGVIEW_PUBLISHED_WEBHOOK_IPS[0]
    assert DEFAULT_WEBHOOK_PORT == 443

    # 1. Non-allowlisted IP rejected (403)
    res_ip = processor.process_request(_body(_sample_payload()), source_ip="203.0.113.10")
    assert res_ip.status_code == 403
    assert res_ip.accepted is False
    assert res_ip.reason == "ip_not_allowlisted"

    # 2. Missing body secret rejected (401)
    no_secret = _sample_payload()
    no_secret.pop("secret", None)
    res_missing = processor.process_request(_body(no_secret), source_ip=allowed_ip)
    assert res_missing.status_code == 401
    assert res_missing.accepted is False
    assert res_missing.reason == "invalid_secret"

    # 3. Wrong body secret rejected (401)
    res_wrong = processor.process_request(
        _body(_sample_payload(secret="wrong-secret-value")),
        source_ip=allowed_ip,
    )
    assert res_wrong.status_code == 401
    assert res_wrong.accepted is False
    assert res_wrong.reason == "invalid_secret"

    # 4. Rate limit enforced after 2 requests from same IP in a fresh 60s window
    now_mono[0] = 170.0
    res_ok = processor.process_request(_body(_sample_payload()), source_ip=allowed_ip)
    assert res_ok.status_code == 200
    assert res_ok.accepted is True
    assert res_ok.duplicate is False
    assert res_ok.packet is not None
    assert "secret" not in res_ok.packet.sanitized_payload

    res_dup = processor.process_request(_body(_sample_payload()), source_ip=allowed_ip)
    assert res_dup.status_code == 200
    assert res_dup.accepted is True
    assert res_dup.duplicate is True

    # 3rd call in same 60s window hits rate limit (429)
    res_rate = processor.process_request(_body(_sample_payload()), source_ip=allowed_ip)
    assert res_rate.status_code == 429
    assert res_rate.accepted is False
    assert res_rate.reason == "rate_limited"

    # Advance clock past rate limit window -> oversized body rejected (413)
    now_mono[0] = 240.0
    oversized = _body(_sample_payload(note="x" * 4096))
    res_big = processor.process_request(oversized, source_ip=allowed_ip)
    assert res_big.status_code == 413
    assert res_big.accepted is False
    assert res_big.reason == "payload_too_large"


def test_webhook_deduplication_schema_version_and_symbol_mapping() -> None:
    processor = TradingViewWebhookProcessor(
        shared_secret=SecretStr(WEBHOOK_SECRET),
        clock=lambda: ANCHOR_NOW,
    )
    ip = TRADINGVIEW_PUBLISHED_WEBHOOK_IPS[1]

    # Unknown / missing schema_version rejected (422)
    res_bad_schema = processor.process_request(
        _body(_sample_payload(schema_version="ai-agent-lens/v2")),
        source_ip=ip,
    )
    assert res_bad_schema.status_code == 422
    assert res_bad_schema.reason == "unknown_schema_version"

    res_no_schema = processor.process_request(
        _body({k: v for k, v in _sample_payload().items() if k != "schema_version"}),
        source_ip=ip,
    )
    assert res_no_schema.status_code == 422
    assert res_no_schema.reason == "unknown_schema_version"

    # Probabilistic fields are rejected to keep rule scores separate from ML probability
    res_prob = processor.process_request(
        _body(_sample_payload(ml_probability=0.82)),
        source_ip=ip,
    )
    assert res_prob.status_code == 422
    assert res_prob.reason == "probabilistic_score_rejected"

    # First packet accepted; exact same symbol+timeframe+event+candle_time duplicate ignored
    first = processor.process_request(_body(_sample_payload()), source_ip=ip)
    assert first.status_code == 200
    assert first.duplicate is False
    assert first.packet is not None
    assert first.packet.schema_version == AI_AGENT_LENS_SCHEMA_VERSION
    assert first.packet.attribution_commit == AI_AGENT_LENS_PINNED_COMMIT
    assert first.packet.is_probabilistic is False

    second = processor.process_request(
        _body(_sample_payload(volume=99999.0)),
        source_ip=ip,
    )
    assert second.status_code == 200
    assert second.accepted is True
    assert second.duplicate is True
    assert second.idempotency_key == first.idempotency_key
    assert second.reason == "duplicate_ignored"

    # Perpetual mapping & spot/perpetual conflict guard
    perp_pkt = parse_ai_agent_lens_packet(
        _sample_payload(symbol="BINANCE:BTCUSDT.P", timeframe="240", event_type="mtf_rsi_shift"),
        received_time=ANCHOR_NOW,
    )
    assert perp_pkt.market_type is MarketType.PERPETUAL
    assert perp_pkt.canonical_symbol == "BTC/USDT:USDT"
    assert perp_pkt.timeframe is Timeframe.H4
    assert perp_pkt.event_type is TradingViewEventType.MTF_RSI_SHIFT

    with pytest.raises(TradingViewWebhookError, match="perpetual"):
        parse_ai_agent_lens_packet(
            _sample_payload(symbol="BINANCE:BTCUSDT.P", market_type="spot"),
            received_time=ANCHOR_NOW,
        )


def test_tradingview_exchange_comparison_and_snapshot_integration() -> None:
    raw_candles = json.loads(CANDLE_FIXTURE.read_text(encoding="utf-8"))["raw_ohlcv"]
    batch = normalize_ohlcv_rows(
        raw_candles,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=ANCHOR_NOW,
        now=ANCHOR_NOW,
    )

    # Baseline snapshot fact IDs remain unchanged when tradingview_packet is None
    base_snap = build_analysis_snapshot(batch)
    assert base_snap.tradingview is None

    # 1. Exact matching packet -> clean READY snapshot with TradingView facts & non-prob factor
    exact_pkt = parse_ai_agent_lens_packet(_sample_payload(), received_time=ANCHOR_NOW)
    exact_cmp = compare_packet_with_exchange_candles(exact_pkt, batch.candles, as_of=ANCHOR_NOW)
    assert exact_cmp.exchange_candle_found is True
    assert exact_cmp.max_deviation_bps is not None
    assert exact_cmp.max_deviation_bps < 25.0
    assert exact_cmp.quality_flags == ()

    snap_tv = build_analysis_snapshot(batch, tradingview_packet=exact_pkt)
    assert snap_tv.quality_status is QualityStatus.READY
    assert snap_tv.tradingview is not None
    assert snap_tv.tradingview.is_probabilistic is False
    assert snap_tv.tradingview.rule_score == pytest.approx(42.5)
    assert snap_tv.facts[: len(base_snap.facts)] == base_snap.facts
    assert any(
        f.rule_id == "tv_ai_agent_lens_score" and not f.is_probabilistic for f in snap_tv.factors
    )

    # 2. Moderate mismatch (> 50 bps) -> PRICE_MISMATCH / DEGRADED
    mod_pkt = parse_ai_agent_lens_packet(
        _sample_payload(close=66_100.0, high=66_200.0),
        received_time=ANCHOR_NOW,
    )
    snap_mod = build_analysis_snapshot(batch, tradingview_packet=mod_pkt)
    assert snap_mod.quality_status is QualityStatus.DEGRADED
    assert QualityFlag.PRICE_MISMATCH in snap_mod.quality_flags

    # 3. Severe mismatch (> 200 bps) -> SEVERE_PRICE_MISMATCH / INSUFFICIENT_DATA
    sev_pkt = parse_ai_agent_lens_packet(
        _sample_payload(close=68_000.0, high=68_500.0),
        received_time=ANCHOR_NOW,
    )
    snap_sev = build_analysis_snapshot(batch, tradingview_packet=sev_pkt)
    assert snap_sev.quality_status is QualityStatus.INSUFFICIENT_DATA
    assert QualityFlag.SEVERE_PRICE_MISMATCH in snap_sev.quality_flags

    # 4. Stale packet detection (> timeframe + 5m grace)
    stale_pkt = parse_ai_agent_lens_packet(
        _sample_payload(),
        received_time=ANCHOR_NOW,
        now=ANCHOR_NOW + timedelta(hours=5),
    )
    assert QualityFlag.STALE in stale_pkt.quality_flags


def test_secret_redacting_filter_scrubs_webhook_body_and_query_secrets() -> None:
    body_secret = "tvBodySecretValue9988"
    query_secret = "tvQuerySecretValue1122"
    flt = SecretRedactingFilter(secrets=[WEBHOOK_SECRET])

    rec_query = logging.LogRecord(
        name="tradingview",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Rejected TradingView request to /webhook/tradingview?webhook_secret=%s&src=tv",
        args=(query_secret,),
        exc_info=None,
    )
    assert flt.filter(rec_query) is True
    assert query_secret not in rec_query.getMessage()
    assert "[REDACTED]" in rec_query.getMessage()

    rec_body = logging.LogRecord(
        name="tradingview",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg='Invalid payload: {"secret": "%s", "symbol": "BTCUSDT"}',
        args=(body_secret,),
        exc_info=None,
    )
    assert flt.filter(rec_body) is True
    assert body_secret not in rec_body.getMessage()
    assert WEBHOOK_SECRET not in flt.redact_text(f"secret={WEBHOOK_SECRET}")

    # Query-string secrets rejected explicitly by processor
    proc = TradingViewWebhookProcessor(
        shared_secret=SecretStr(WEBHOOK_SECRET),
        allowed_ips=("127.0.0.1",),
        clock=lambda: ANCHOR_NOW,
    )
    q_res = proc.process_request(
        _body(_sample_payload()),
        source_ip="127.0.0.1",
        query_string=f"secret={WEBHOOK_SECRET}",
    )
    assert q_res.status_code == 400
    assert q_res.reason == "query_secret_not_allowed"


@pytest.mark.asyncio
async def test_webhook_http_server_framing_in_process() -> None:
    class _FakeWriter:
        def __init__(self, peername: tuple[str, int]) -> None:
            self._peername = peername
            self.buffer = bytearray()
            self.closed = False

        def get_extra_info(self, name: str) -> Any:
            return self._peername if name == "peername" else None

        def write(self, data: bytes) -> None:
            self.buffer.extend(data)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return None

    processor = TradingViewWebhookProcessor(
        shared_secret=SecretStr(WEBHOOK_SECRET),
        allowed_ips=("127.0.0.1", "::1"),
        clock=lambda: ANCHOR_NOW,
    )
    server = TradingViewWebhookServer(processor, host="127.0.0.1", port=443)

    async def _send(
        method: str,
        path: str,
        raw_body: bytes,
    ) -> tuple[int, dict[str, Any]]:
        reader = asyncio.StreamReader()
        req = (
            f"{method} {path} HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(raw_body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + raw_body
        reader.feed_data(req)
        reader.feed_eof()
        writer = _FakeWriter(("127.0.0.1", 50000))
        await server._handle_client(reader, writer)  # type: ignore[arg-type]
        assert writer.closed is True
        head, _, body = bytes(writer.buffer).partition(b"\r\n\r\n")
        status_code = int(head.decode("ascii").split(" ", 2)[1])
        return status_code, json.loads(body.decode("utf-8"))

    st_ok, d_ok = await _send("POST", "/webhook/tradingview", _body(_sample_payload()))
    assert st_ok == 200
    assert d_ok["accepted"] is True
    assert d_ok["duplicate"] is False

    st_dup, d_dup = await _send("POST", "/webhook/tradingview", _body(_sample_payload()))
    assert st_dup == 200
    assert d_dup["duplicate"] is True

    st_bad, _ = await _send("POST", "/webhook/tradingview", _body(_sample_payload(secret="wrong")))
    assert st_bad == 401

    st_405, _ = await _send("GET", "/webhook/tradingview", b"")
    assert st_405 == 405

    st_404, _ = await _send("POST", "/other", b"{}")
    assert st_404 == 404


@pytest.mark.asyncio
async def test_webhook_http_server_accepts_valid_and_rejects_bad_requests() -> None:
    processor = TradingViewWebhookProcessor(
        shared_secret=SecretStr(WEBHOOK_SECRET),
        allowed_ips=("127.0.0.1", "::1"),
        clock=lambda: ANCHOR_NOW,
    )
    server = TradingViewWebhookServer(processor, host="127.0.0.1", port=0)
    try:
        await server.start()
    except PermissionError:
        pytest.skip("loopback TCP socket bind blocked by sandbox")
    try:

        async def _post(
            raw_body: bytes, path: str = "/webhook/tradingview"
        ) -> tuple[int, dict[str, Any]]:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            except PermissionError:
                pytest.skip("loopback TCP socket connect blocked by sandbox")
            req = (
                f"POST {path} HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(raw_body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii") + raw_body
            writer.write(req)
            await writer.drain()
            raw_resp = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, _, body = raw_resp.partition(b"\r\n\r\n")
            status_code = int(head.decode("ascii").split(" ", 2)[1])
            return status_code, json.loads(body.decode("utf-8"))

        st1, d1 = await _post(_body(_sample_payload()))
        assert st1 == 200
        assert d1["accepted"] is True
        assert d1["duplicate"] is False

        st2, d2 = await _post(_body(_sample_payload()))
        assert st2 == 200
        assert d2["duplicate"] is True

        st3, d3 = await _post(_body(_sample_payload(secret="wrong")))
        assert st3 == 401
        assert d3["accepted"] is False
    finally:
        await server.stop()
