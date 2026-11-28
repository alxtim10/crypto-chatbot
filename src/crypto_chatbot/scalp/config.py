"""Typed configuration models for `scalp_v1` strategy and macro event calendar."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_SCALP_UNIVERSE_PATH = Path("config/universe_scalp.txt")
DEFAULT_EVENT_CALENDAR_PATH = Path("config/event_calendar.yaml")

DEFAULT_SCALP_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
)


class ScalpUniverseConfig(BaseModel):
    """Weekly universe eligibility thresholds (SCALP-SPEC-v1 §1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_vol_24h_usd: float = Field(default=100_000_000.0, gt=0.0)
    max_spread_bps: float = Field(default=1.0, gt=0.0)
    min_depth_usd: float = Field(default=250_000.0, gt=0.0)
    min_listing_age_days: int = Field(default=180, ge=1)


class ScalpCostConfig(BaseModel):
    """Per-symbol, per-order-type cost assumptions (SCALP-SPEC-v1 §5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vip_tier: str = "vip0"
    maker_fee_bps: float = Field(default=2.0, ge=0.0)  # Gate VIP0 futures: 0.02%
    taker_fee_bps: float = Field(default=5.0, ge=0.0)  # Gate VIP0 futures: 0.05%
    order_notional_usd: float = Field(default=200.0, gt=0.0)
    fallback_spread_bps: float = Field(default=1.0, ge=0.0)
    fallback_taker_slippage_bps: float = Field(default=1.5, ge=0.0)
    max_cost_pct_of_r: float = Field(default=0.15, gt=0.0, le=1.0)
    legacy_flat_mode: bool = False
    legacy_flat_round_trip_bps: float = Field(default=10.0, ge=0.0)


class ScalpFilterConfig(BaseModel):
    """Blocking filter parameters (SCALP-SPEC-v1 §6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    funding_window_minutes: int = Field(default=10, ge=0)
    macro_pre_minutes: int = Field(default=15, ge=0)
    macro_post_minutes: int = Field(default=30, ge=0)
    liq_cascade_block_bars: int = Field(default=3, ge=1)
    crowded_funding_abs_8h: float = Field(default=0.0005, gt=0.0)  # 0.05%
    news_burst_window_minutes: int = Field(default=30, ge=1)
    daily_stop_r: float = Field(default=-3.0, lt=0.0)
    loss_cooldown_minutes: int = Field(default=30, ge=0)
    max_open_taken_trades: int = Field(default=2, ge=1)
    event_calendar_path: str = str(DEFAULT_EVENT_CALENDAR_PATH)


class ScalpPositionConfig(BaseModel):
    """Owner-configured position size calculator parameters from `.env` (SCALP-SPEC-v1 §8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    equity_usd: float | None = Field(default=None, gt=0.0)
    risk_per_trade_pct: float | None = Field(default=None, gt=0.0, le=5.0)

    @property
    def is_configured(self) -> bool:
        return self.equity_usd is not None and self.risk_per_trade_pct is not None


class ScalpConfig(BaseModel):
    """Full pre-registered parameter set for `scalp_v1` (SCALP-SPEC-v1 §3-§8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: Literal["shadow", "live_small", "live"] = "shadow"
    universe_path: str = str(DEFAULT_SCALP_UNIVERSE_PATH)
    # Mode selection (§3)
    trend_adx_min: float = Field(default=25.0, gt=0.0)
    range_adx_max: float = Field(default=20.0, gt=0.0)
    # Trend pullback (§4.1)
    zone_max_atr_5m: float = Field(default=1.5, gt=0.0)
    ema20_zone_atr_15m: float = Field(default=0.15, gt=0.0)
    rejection_min_body_atr_5m: float = Field(default=0.5, gt=0.0)
    sl_buffer_atr_5m: float = Field(default=0.25, ge=0.0)
    tp1_r: float = Field(default=1.0, gt=0.0)
    tp1_close_frac: float = Field(default=0.60, gt=0.0, lt=1.0)
    tp2_max_r: float = Field(default=2.0, gt=0.0)
    max_hold_bars_5m: int = Field(default=18, ge=1)  # 18 * 5m = 90 min
    # Range fade (§4.2)
    range_min_cost_mult: float = Field(default=6.0, gt=0.0)
    range_min_atr_1h_mult: float = Field(default=1.5, gt=0.0)
    range_tp2_edge_frac: float = Field(default=0.80, gt=0.5, le=1.0)
    # Common rules (§4.3)
    min_r_bps: float = Field(default=20.0, gt=0.0)
    max_r_bps: float = Field(default=80.0, gt=0.0)
    min_net_rr_tp2: float = Field(default=1.3, gt=0.0)
    entry_valid_bars_5m: int = Field(default=3, ge=1)  # 3 * 5m = 15 min
    # Backtest / execution (§9.2)
    human_delay_sec: int = Field(default=90, ge=0)
    # Nested configs
    universe: ScalpUniverseConfig = Field(default_factory=ScalpUniverseConfig)
    costs: ScalpCostConfig = Field(default_factory=ScalpCostConfig)
    filters: ScalpFilterConfig = Field(default_factory=ScalpFilterConfig)
    position: ScalpPositionConfig = Field(default_factory=ScalpPositionConfig)


class MacroEventWindow(BaseModel):
    """Scheduled high-impact macro event entry from `config/event_calendar.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    event_time_utc: datetime
    pre_minutes: int | None = Field(default=None, ge=0)
    post_minutes: int | None = Field(default=None, ge=0)

    @field_validator("event_time_utc")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event_time_utc must be timezone-aware UTC")
        return value.astimezone(UTC)

    def blocks(
        self,
        ts_utc: datetime,
        *,
        default_pre_minutes: int = 15,
        default_post_minutes: int = 30,
    ) -> bool:
        """Return True if `ts_utc` falls inside `[event - pre, event + post]`."""
        ts = ts_utc.astimezone(UTC)
        pre = self.pre_minutes if self.pre_minutes is not None else default_pre_minutes
        post = self.post_minutes if self.post_minutes is not None else default_post_minutes
        start = self.event_time_utc - timedelta(minutes=pre)
        end = self.event_time_utc + timedelta(minutes=post)
        return start <= ts <= end


def load_event_calendar(path: Path | str | None = None) -> tuple[MacroEventWindow, ...]:
    """Load scheduled macro events from `config/event_calendar.yaml` (stdlib YAML subset)."""
    target = Path(path) if path is not None else DEFAULT_EVENT_CALENDAR_PATH
    if not target.is_file() and not target.is_absolute():
        candidate = Path(__file__).resolve().parents[3] / target
        if candidate.is_file():
            target = candidate
    if not target.is_file():
        return ()

    events: list[MacroEventWindow] = []
    current: dict[str, object] | None = None
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped and not stripped.startswith(('"', "'")):
            # Preserve ISO timestamps (no # inside), strip trailing comments
            stripped = stripped.split("#", 1)[0].strip()
            if not stripped:
                continue
        if stripped == "events:":
            continue
        if stripped.startswith("- "):
            if current is not None:
                events.append(MacroEventWindow.model_validate(current))
            current = {}
            rest = stripped[2:].strip()
            if ":" in rest:
                k, _, v = rest.partition(":")
                current[k.strip()] = _unquote(v)
        elif current is not None and ":" in stripped:
            k, _, v = stripped.partition(":")
            val = _unquote(v)
            if k.strip() in {"pre_minutes", "post_minutes"} and isinstance(val, str) and val:
                current[k.strip()] = int(val)
            else:
                current[k.strip()] = val
    if current is not None:
        events.append(MacroEventWindow.model_validate(current))
    return tuple(events)


def _unquote(raw: str) -> str:
    val = raw.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val
