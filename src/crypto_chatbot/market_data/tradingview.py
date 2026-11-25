"""TradingView AI Agent Lens (`ai-agent-lens/v1`) webhook adapter, verification, and server."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import ssl
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from crypto_chatbot.market_data.base import DEFAULT_FRESHNESS_GRACE
from crypto_chatbot.market_data.derivatives_quality import (
    PRICE_MISMATCH_BLOCK_BPS,
    PRICE_MISMATCH_DEGRADE_BPS,
)
from crypto_chatbot.market_data.quality import is_candle_stale
from crypto_chatbot.market_data.symbols import SymbolError, resolve_symbol
from crypto_chatbot.schemas.market import MarketCandle, MarketType, QualityFlag, Timeframe
from crypto_chatbot.schemas.tradingview import (
    AI_AGENT_LENS_PINNED_COMMIT,
    AI_AGENT_LENS_SCHEMA_VERSION,
    FORBIDDEN_SECRET_PAYLOAD_KEYS,
    TRADINGVIEW_PUBLISHED_WEBHOOK_IPS,
    TradingViewEventType,
    TradingViewExchangeComparison,
    TradingViewPacket,
)

logger = logging.getLogger(__name__)

DEFAULT_WEBHOOK_PORT = 443
DEFAULT_MAX_WEBHOOK_BODY_BYTES = 32_768
DEFAULT_WEBHOOK_RATE_LIMIT_PER_MINUTE = 60

_TV_TIMEFRAME_MAP: dict[str, Timeframe] = {
    "15": Timeframe.M15,
    "15m": Timeframe.M15,
    "60": Timeframe.H1,
    "1h": Timeframe.H1,
    "240": Timeframe.H4,
    "4h": Timeframe.H4,
    "d": Timeframe.D1,
    "1d": Timeframe.D1,
}

_PROBABILISTIC_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {"ml_probability", "win_probability", "probability", "success_probability"}
)


class TradingViewPacketStore(Protocol):
    """Minimal persistence protocol for deduplicating verified TradingView packets."""

    def save_packet(self, packet: TradingViewPacket) -> bool: ...


class TradingViewWebhookError(ValueError):
    """Structured rejection error carrying HTTP status code and non-secret reason."""

    def __init__(self, message: str, *, status_code: int, reason: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


class WebhookResult(BaseModel):
    """Result of handling one TradingView webhook request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status_code: int
    accepted: bool
    duplicate: bool = False
    idempotency_key: str | None = None
    reason: str | None = None
    packet: TradingViewPacket | None = None


def _require_utc(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise TradingViewWebhookError(
            f"{label} must be timezone-aware UTC",
            status_code=422,
            reason="naive_timestamp",
        )
    return dt.astimezone(UTC)


def _parse_tv_timestamp(raw: Any, label: str) -> datetime:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not math.isfinite(float(raw)):
            raise TradingViewWebhookError(
                f"{label} must be a finite timestamp",
                status_code=422,
                reason="invalid_timestamp",
            )
        seconds = float(raw) / 1000.0 if abs(float(raw)) > 1e11 else float(raw)
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = datetime.fromisoformat(raw.strip())
        except ValueError:
            raise TradingViewWebhookError(
                f"{label} is not a valid ISO-8601 timestamp",
                status_code=422,
                reason="invalid_timestamp",
            ) from None
        return _require_utc(parsed, label)
    raise TradingViewWebhookError(
        f"missing or invalid {label}",
        status_code=422,
        reason="invalid_timestamp",
    )


def _to_finite_float(val: Any, label: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TradingViewWebhookError(
            f"{label} must be numeric",
            status_code=422,
            reason="invalid_ohlc",
        )
    num = float(val)
    if not math.isfinite(num):
        raise TradingViewWebhookError(
            f"{label} must be finite",
            status_code=422,
            reason="invalid_ohlc",
        )
    return num


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _sanitize_value(v)
            for k, v in value.items()
            if str(k).strip().lower() not in FORBIDDEN_SECRET_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def sanitize_tradingview_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy a TradingView JSON payload while stripping all credential fields."""
    cleaned = _sanitize_value(payload)
    return dict(cleaned) if isinstance(cleaned, Mapping) else {}


def compute_tradingview_idempotency_key(
    *,
    canonical_symbol: str,
    market_type: MarketType,
    timeframe: Timeframe,
    event_type: TradingViewEventType,
    candle_close_time: datetime,
) -> str:
    """Compute a deterministic idempotency key from symbol, timeframe, event, and candle time."""
    close_utc = candle_close_time.astimezone(UTC).isoformat()
    raw = f"{canonical_symbol}|{market_type.value}|{timeframe.value}|{event_type.value}|{close_utc}"
    return "tv-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def parse_ai_agent_lens_packet(
    raw_payload: Mapping[str, Any],
    *,
    received_time: datetime,
    now: datetime | None = None,
    freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
) -> TradingViewPacket:
    """Validate and normalize an `ai-agent-lens/v1` webhook payload."""
    received_utc = _require_utc(received_time, "received_time")
    check_time = _require_utc(now, "now") if now is not None else received_utc

    schema_ver = raw_payload.get("schema_version")
    if schema_ver != AI_AGENT_LENS_SCHEMA_VERSION:
        raise TradingViewWebhookError(
            f"unsupported or missing schema_version '{schema_ver}'",
            status_code=422,
            reason="unknown_schema_version",
        )

    for bad_key in _PROBABILISTIC_FORBIDDEN_KEYS:
        if raw_payload.get(bad_key) is not None:
            raise TradingViewWebhookError(
                f"TradingView packet must not supply '{bad_key}'; "
                "keep rule scores non-probabilistic",
                status_code=422,
                reason="probabilistic_score_rejected",
            )

    raw_symbol = raw_payload.get("symbol") or raw_payload.get("ticker")
    if not isinstance(raw_symbol, str) or not raw_symbol.strip():
        raise TradingViewWebhookError(
            "symbol is required in ai-agent-lens/v1 packet",
            status_code=422,
            reason="invalid_symbol",
        )

    raw_mkt = raw_payload.get("market_type")
    if isinstance(raw_mkt, str) and raw_mkt.strip():
        mkt_norm = raw_mkt.strip().lower()
        if mkt_norm == "spot":
            market_type = MarketType.SPOT
        elif mkt_norm in ("perpetual", "perp", "swap"):
            market_type = MarketType.PERPETUAL
        else:
            raise TradingViewWebhookError(
                f"unsupported market_type '{raw_mkt}'",
                status_code=422,
                reason="invalid_market_type",
            )
    else:
        upper_sym = raw_symbol.strip().upper()
        if upper_sym.endswith(".P") or ":USDT" in upper_sym or "PERP" in upper_sym:
            market_type = MarketType.PERPETUAL
        else:
            market_type = MarketType.SPOT

    try:
        mapping = resolve_symbol(raw_symbol, market_type=market_type)
    except SymbolError as exc:
        raise TradingViewWebhookError(
            str(exc),
            status_code=422,
            reason="invalid_symbol",
        ) from None

    extra_flags: list[QualityFlag] = []
    venue = mapping.venue
    raw_venue = raw_payload.get("venue")
    if isinstance(raw_venue, str) and raw_venue.strip():
        venue = raw_venue.strip().lower()
        if venue != mapping.venue:
            extra_flags.append(QualityFlag.SYMBOL_MISMATCH)

    raw_tf = raw_payload.get("timeframe")
    if not isinstance(raw_tf, str) or raw_tf.strip().lower() not in _TV_TIMEFRAME_MAP:
        raise TradingViewWebhookError(
            f"unsupported timeframe '{raw_tf}'",
            status_code=422,
            reason="invalid_timeframe",
        )
    timeframe = _TV_TIMEFRAME_MAP[raw_tf.strip().lower()]

    raw_event = raw_payload.get("event_type", TradingViewEventType.BAR_CLOSE.value)
    if not isinstance(raw_event, str):
        raise TradingViewWebhookError(
            "event_type must be a string",
            status_code=422,
            reason="invalid_event_type",
        )
    try:
        event_type = TradingViewEventType(raw_event.strip().lower())
    except ValueError:
        raise TradingViewWebhookError(
            f"unsupported event_type '{raw_event}'",
            status_code=422,
            reason="invalid_event_type",
        ) from None

    raw_open = raw_payload.get("candle_open_time", raw_payload.get("open_time"))
    raw_close = raw_payload.get(
        "candle_close_time",
        raw_payload.get("close_time", raw_payload.get("candle_time")),
    )
    if raw_open is None and raw_close is None:
        raise TradingViewWebhookError(
            "candle_open_time or candle_close_time is required",
            status_code=422,
            reason="missing_candle_time",
        )
    if raw_open is not None and raw_close is not None:
        open_time = _parse_tv_timestamp(raw_open, "candle_open_time")
        close_time = _parse_tv_timestamp(raw_close, "candle_close_time")
    elif raw_open is not None:
        open_time = _parse_tv_timestamp(raw_open, "candle_open_time")
        close_time = open_time + timeframe.duration
    else:
        close_time = _parse_tv_timestamp(raw_close, "candle_close_time")
        open_time = close_time - timeframe.duration

    if is_candle_stale(close_time, timeframe, check_time, freshness_grace=freshness_grace):
        extra_flags.append(QualityFlag.STALE)

    ohlc_block = raw_payload.get("ohlc")
    src: Mapping[str, Any] = ohlc_block if isinstance(ohlc_block, Mapping) else raw_payload
    open_p = _to_finite_float(src.get("open"), "open")
    high_p = _to_finite_float(src.get("high"), "high")
    low_p = _to_finite_float(src.get("low"), "low")
    close_p = _to_finite_float(src.get("close"), "close")

    raw_vol = src.get("volume", raw_payload.get("volume"))
    volume = _to_finite_float(raw_vol, "volume") if raw_vol is not None else None

    raw_score = raw_payload.get("rule_score", raw_payload.get("lens_score"))
    rule_score = _to_finite_float(raw_score, "rule_score") if raw_score is not None else None

    raw_label = raw_payload.get("rule_label")
    rule_label = (
        str(raw_label).strip() if isinstance(raw_label, str) and raw_label.strip() else None
    )

    commit_raw = raw_payload.get("attribution_commit")
    commit = (
        commit_raw.strip()
        if isinstance(commit_raw, str) and len(commit_raw.strip()) >= 7
        else AI_AGENT_LENS_PINNED_COMMIT
    )

    idempotency_key = compute_tradingview_idempotency_key(
        canonical_symbol=mapping.canonical_symbol,
        market_type=market_type,
        timeframe=timeframe,
        event_type=event_type,
        candle_close_time=close_time,
    )
    sanitized = sanitize_tradingview_payload(raw_payload)
    available_time = max(close_time, received_utc)

    try:
        return TradingViewPacket(
            idempotency_key=idempotency_key,
            schema_version=AI_AGENT_LENS_SCHEMA_VERSION,
            attribution_commit=commit,
            provider="tradingview",
            venue=venue,
            canonical_symbol=mapping.canonical_symbol,
            provider_symbol=mapping.provider_symbol,
            market_type=market_type,
            timeframe=timeframe,
            event_type=event_type,
            candle_open_time=open_time,
            candle_close_time=close_time,
            received_time=received_utc,
            available_time=available_time,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=volume,
            rule_score=rule_score,
            rule_label=rule_label,
            quality_flags=tuple(dict.fromkeys(extra_flags)),
            sanitized_payload=sanitized,
        )
    except ValidationError as exc:
        raise TradingViewWebhookError(
            f"invalid TradingViewPacket: {exc}",
            status_code=422,
            reason="invalid_packet",
        ) from None


def _bps_diff(a: float, b: float) -> float:
    return abs(a - b) / b * 10_000.0


def compare_packet_with_exchange_candles(
    packet: TradingViewPacket,
    exchange_candles: Sequence[MarketCandle],
    *,
    as_of: datetime | None = None,
    use_available_time: bool = False,
) -> TradingViewExchangeComparison:
    """Compare one `TradingViewPacket` against closed exchange candles."""
    flags: list[QualityFlag] = list(packet.quality_flags)
    limitations: list[str] = []

    if QualityFlag.STALE in flags:
        limitations.append("Paket TradingView ai-agent-lens/v1 melewati batas kesegaran (STALE).")

    eligible = [
        c
        for c in exchange_candles
        if c.is_closed
        and c.canonical_symbol == packet.canonical_symbol
        and c.market_type is packet.market_type
        and c.timeframe is packet.timeframe
        and (
            as_of is None
            or (c.available_time if use_available_time else c.close_time) <= as_of.astimezone(UTC)
        )
    ]

    matched = next(
        (c for c in eligible if c.open_time == packet.candle_open_time),
        None,
    )
    if matched is None:
        matched = next(
            (c for c in eligible if c.close_time == packet.candle_close_time),
            None,
        )

    if matched is None:
        if any(c.canonical_symbol != packet.canonical_symbol for c in exchange_candles):
            flags.append(QualityFlag.SYMBOL_MISMATCH)
            limitations.append("Simbol paket TradingView tidak cocok dengan candle bursa.")
        else:
            limitations.append(
                "Candle bursa yang bersesuaian belum ditemukan untuk verifikasi TradingView."
            )
        return TradingViewExchangeComparison(
            idempotency_key=packet.idempotency_key,
            canonical_symbol=packet.canonical_symbol,
            timeframe=packet.timeframe,
            candle_open_time=packet.candle_open_time,
            candle_close_time=packet.candle_close_time,
            tv_close=packet.close,
            exchange_candle_found=False,
            quality_flags=tuple(dict.fromkeys(flags)),
            limitations=tuple(dict.fromkeys(limitations)),
        )

    if packet.venue != matched.venue:
        flags.append(QualityFlag.SYMBOL_MISMATCH)
        limitations.append(
            f"Venue TradingView ({packet.venue}) berbeda dari candle bursa ({matched.venue})."
        )

    close_bps = _bps_diff(packet.close, matched.close)
    high_bps = _bps_diff(packet.high, matched.high)
    low_bps = _bps_diff(packet.low, matched.low)
    max_bps = max(close_bps, high_bps, low_bps)

    if max_bps > PRICE_MISMATCH_BLOCK_BPS:
        flags.append(QualityFlag.SEVERE_PRICE_MISMATCH)
        flags.append(QualityFlag.PRICE_MISMATCH)
        limitations.append(
            "Deviasi harga TradingView vs candle bursa melebihi "
            f"{PRICE_MISMATCH_BLOCK_BPS:.0f} bps (SEVERE_PRICE_MISMATCH)."
        )
    elif max_bps > PRICE_MISMATCH_DEGRADE_BPS:
        flags.append(QualityFlag.PRICE_MISMATCH)
        limitations.append(
            "Terdapat deviasi harga TradingView vs candle bursa di atas "
            f"{PRICE_MISMATCH_DEGRADE_BPS:.0f} bps."
        )

    return TradingViewExchangeComparison(
        idempotency_key=packet.idempotency_key,
        canonical_symbol=packet.canonical_symbol,
        timeframe=packet.timeframe,
        candle_open_time=packet.candle_open_time,
        candle_close_time=packet.candle_close_time,
        tv_close=packet.close,
        exchange_close=matched.close,
        close_deviation_bps=close_bps,
        high_deviation_bps=high_bps,
        low_deviation_bps=low_bps,
        max_deviation_bps=max_bps,
        exchange_candle_found=True,
        quality_flags=tuple(dict.fromkeys(flags)),
        limitations=tuple(dict.fromkeys(limitations)),
    )


def _normalize_ip(raw_ip: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    cleaned = raw_ip.strip()
    if "%" in cleaned:
        cleaned = cleaned.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(cleaned)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


class TradingViewWebhookProcessor:
    """Authenticates, rate-limits, validates, and deduplicates `ai-agent-lens/v1` webhooks."""

    def __init__(
        self,
        *,
        shared_secret: SecretStr,
        allowed_ips: Sequence[str] = TRADINGVIEW_PUBLISHED_WEBHOOK_IPS,
        max_body_bytes: int = DEFAULT_MAX_WEBHOOK_BODY_BYTES,
        rate_limit_per_minute: int = DEFAULT_WEBHOOK_RATE_LIMIT_PER_MINUTE,
        freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
        repository: TradingViewPacketStore | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if not shared_secret.get_secret_value().strip():
            raise ValueError("shared_secret must not be blank")
        if max_body_bytes < 128:
            raise ValueError("max_body_bytes must be >= 128")
        if rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be >= 1")

        self._shared_secret = shared_secret
        self._networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for entry in allowed_ips:
            text = entry.strip()
            if not text:
                continue
            self._networks.append(ipaddress.ip_network(text, strict=False))
        self._max_body_bytes = max_body_bytes
        self._rate_limit = rate_limit_per_minute
        self._freshness_grace = freshness_grace
        self._repository = repository
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._monotonic: Callable[[], float] = monotonic_clock or time.monotonic
        self._ip_windows: dict[str, deque[float]] = {}
        self._seen_keys: OrderedDict[str, None] = OrderedDict()
        self._max_seen_keys = 10_000

    def _remember_key(self, key: str) -> None:
        if key in self._seen_keys:
            self._seen_keys.move_to_end(key)
            return
        self._seen_keys[key] = None
        if len(self._seen_keys) > self._max_seen_keys:
            self._seen_keys.popitem(last=False)

    def _is_ip_allowed(self, source_ip: str) -> bool:
        addr = _normalize_ip(source_ip)
        if addr is None:
            return False
        return any(addr.version == net.version and addr in net for net in self._networks)

    def _check_rate_limit(self, source_ip: str) -> bool:
        now = self._monotonic()
        cutoff = now - 60.0
        if len(self._ip_windows) > 1024:
            stale_ips = [ip for ip, b in self._ip_windows.items() if not b or b[-1] <= cutoff]
            for ip in stale_ips:
                self._ip_windows.pop(ip, None)
        bucket = self._ip_windows.setdefault(source_ip, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self._rate_limit:
            return False
        bucket.append(now)
        return True

    def process_request(
        self,
        raw_body: bytes,
        *,
        source_ip: str,
        query_string: str = "",
    ) -> WebhookResult:
        """Process one raw HTTP/HTTPS webhook body with constant-time auth & deduplication."""
        if not self._is_ip_allowed(source_ip):
            logger.warning("Rejected TradingView webhook from non-allowlisted IP %s", source_ip)
            return WebhookResult(
                status_code=403,
                accepted=False,
                reason="ip_not_allowlisted",
            )

        if len(raw_body) > self._max_body_bytes:
            return WebhookResult(
                status_code=413,
                accepted=False,
                reason="payload_too_large",
            )

        if not self._check_rate_limit(source_ip):
            logger.warning("Rate-limited TradingView webhook from IP %s", source_ip)
            return WebhookResult(
                status_code=429,
                accepted=False,
                reason="rate_limited",
            )

        if query_string:
            qs_params = parse_qs(query_string, keep_blank_values=True)
            bad_keys = sorted(
                {
                    key.strip().lower()
                    for key in qs_params
                    if key.strip().lower() in FORBIDDEN_SECRET_PAYLOAD_KEYS
                }
            )
            if bad_keys:
                logger.warning(
                    "Rejected TradingView webhook with forbidden query key(s): %s",
                    ",".join(bad_keys),
                )
                return WebhookResult(
                    status_code=400,
                    accepted=False,
                    reason="query_secret_not_allowed",
                )

        try:
            decoded = raw_body.decode("utf-8")
            parsed = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return WebhookResult(
                status_code=400,
                accepted=False,
                reason="invalid_json",
            )

        if not isinstance(parsed, dict):
            return WebhookResult(
                status_code=400,
                accepted=False,
                reason="json_object_required",
            )

        provided_raw = parsed.get("secret")
        if provided_raw is None:
            provided_raw = parsed.get("webhook_secret")
        if not isinstance(provided_raw, str) or not provided_raw:
            logger.warning("Rejected TradingView webhook missing body secret from IP %s", source_ip)
            return WebhookResult(
                status_code=401,
                accepted=False,
                reason="invalid_secret",
            )

        expected = self._shared_secret.get_secret_value()
        if not hmac.compare_digest(provided_raw.encode("utf-8"), expected.encode("utf-8")):
            logger.warning(
                "Rejected TradingView webhook with wrong body secret from IP %s",
                source_ip,
            )
            return WebhookResult(
                status_code=401,
                accepted=False,
                reason="invalid_secret",
            )

        now = self._clock()
        try:
            packet = parse_ai_agent_lens_packet(
                parsed,
                received_time=now,
                now=now,
                freshness_grace=self._freshness_grace,
            )
        except TradingViewWebhookError as exc:
            logger.info("Rejected invalid ai-agent-lens packet: %s", exc.reason)
            return WebhookResult(
                status_code=exc.status_code,
                accepted=False,
                reason=exc.reason,
            )

        if packet.idempotency_key in self._seen_keys:
            return WebhookResult(
                status_code=200,
                accepted=True,
                duplicate=True,
                idempotency_key=packet.idempotency_key,
                reason="duplicate_ignored",
            )

        if self._repository is not None:
            inserted = self._repository.save_packet(packet)
            if not inserted:
                self._remember_key(packet.idempotency_key)
                return WebhookResult(
                    status_code=200,
                    accepted=True,
                    duplicate=True,
                    idempotency_key=packet.idempotency_key,
                    reason="duplicate_ignored",
                )

        self._remember_key(packet.idempotency_key)
        return WebhookResult(
            status_code=200,
            accepted=True,
            duplicate=False,
            idempotency_key=packet.idempotency_key,
            packet=packet,
        )


def build_webhook_ssl_context(certfile: Path, keyfile: Path) -> ssl.SSLContext:
    """Create a TLS 1.2+ server context for HTTPS port 443 TradingView webhooks."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    return ctx


class TradingViewWebhookServer:
    """Bounded async HTTPS/HTTP webhook server for `POST /webhook/tradingview`."""

    def __init__(
        self,
        processor: TradingViewWebhookProcessor,
        *,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = DEFAULT_WEBHOOK_PORT,
        ssl_context: ssl.SSLContext | None = None,
        max_body_bytes: int = DEFAULT_MAX_WEBHOOK_BODY_BYTES,
    ) -> None:
        self._processor = processor
        self._host = host
        self._port = port
        self._ssl_context = ssl_context
        self._max_body_bytes = max_body_bytes
        self._server: asyncio.Server | None = None

    @property
    def port(self) -> int:
        if self._server is not None and self._server.sockets:
            sockname = self._server.sockets[0].getsockname()
            return int(sockname[1])
        return self._port

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._host,
            port=self._port,
            ssl=self._ssl_context,
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _send_json(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        payload: dict[str, Any],
    ) -> None:
        reasons = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            413: "Payload Too Large",
            422: "Unprocessable Entity",
            429: "Too Many Requests",
        }
        body = json.dumps(payload).encode("utf-8")
        headers = (
            f"HTTP/1.1 {status_code} {reasons.get(status_code, 'Error')}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        writer.write(headers + body)
        await writer.drain()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw_head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
        except (TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            await writer.wait_closed()
            return

        try:
            lines = raw_head.decode("latin-1").split("\r\n")
            req_parts = lines[0].split(" ")
            if len(req_parts) < 2:
                await self._send_json(writer, 400, {"accepted": False, "reason": "bad_request"})
                return
            method, target = req_parts[0].upper(), req_parts[1]
            path, _, query_string = target.partition("?")
            if path != "/webhook/tradingview":
                await self._send_json(writer, 404, {"accepted": False, "reason": "not_found"})
                return
            if method != "POST":
                await self._send_json(
                    writer,
                    405,
                    {"accepted": False, "reason": "method_not_allowed"},
                )
                return

            content_length = 0
            for hdr_line in lines[1:]:
                if ":" not in hdr_line:
                    continue
                h_name, h_val = hdr_line.split(":", 1)
                if h_name.strip().lower() == "content-length":
                    try:
                        content_length = int(h_val.strip())
                    except ValueError:
                        await self._send_json(
                            writer,
                            400,
                            {"accepted": False, "reason": "invalid_content_length"},
                        )
                        return

            if content_length < 0:
                await self._send_json(writer, 400, {"accepted": False, "reason": "bad_request"})
                return
            if content_length > self._max_body_bytes:
                await self._send_json(
                    writer,
                    413,
                    {"accepted": False, "reason": "payload_too_large"},
                )
                return

            raw_body = (
                await asyncio.wait_for(reader.readexactly(content_length), timeout=5.0)
                if content_length > 0
                else b""
            )
            peer = writer.get_extra_info("peername")
            source_ip = str(peer[0]) if isinstance(peer, tuple) and peer else ""
            result = self._processor.process_request(
                raw_body,
                source_ip=source_ip,
                query_string=query_string,
            )
            await self._send_json(
                writer,
                result.status_code,
                {
                    "accepted": result.accepted,
                    "duplicate": result.duplicate,
                    "idempotency_key": result.idempotency_key,
                    "reason": result.reason,
                },
            )
        except Exception:
            await self._send_json(writer, 400, {"accepted": False, "reason": "malformed_request"})
        finally:
            writer.close()
            await writer.wait_closed()
