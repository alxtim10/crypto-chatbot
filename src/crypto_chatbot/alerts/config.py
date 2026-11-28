"""Universe loader and stdlib YAML configuration for the hourly alert worker."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crypto_chatbot.market_data.symbols import SymbolError, resolve_default_perp_symbol
from crypto_chatbot.scalp.config import ScalpConfig
from crypto_chatbot.schemas.market import Timeframe

WIB = timezone(timedelta(hours=7))

DEFAULT_UNIVERSE_PATH = Path("config/universe.txt")
DEFAULT_SCREENER_CONFIG_PATH = Path("config/screener.yaml")

FALLBACK_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "SUI/USDT:USDT",
    "DOGE/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "ADA/USDT:USDT",
)


def _find_config_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.is_absolute():
        # Check relative to repo root (3 levels up from src/crypto_chatbot/alerts/config.py)
        repo_candidate = Path(__file__).resolve().parents[3] / path
        if repo_candidate.is_file():
            return repo_candidate
    return None


def load_universe(
    path: Path | str | None = None,
    *,
    fallback: Sequence[str] = FALLBACK_UNIVERSE,
) -> tuple[str, ...]:
    """Load canonical perpetual symbols from `config/universe.txt` (default)."""
    target = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH
    resolved = _find_config_file(target)
    if resolved is None:
        return tuple(fallback)

    symbols: list[str] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        cleaned = raw_line.split("#", 1)[0].strip()
        if not cleaned:
            continue
        try:
            canon = resolve_default_perp_symbol(cleaned).canonical_symbol
        except (SymbolError, ValueError):
            continue
        if canon not in symbols:
            symbols.append(canon)
    return tuple(symbols) if symbols else tuple(fallback)


def _parse_hhmm(raw: str) -> tuple[int, int]:
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid HH:MM time '{raw}'")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"out-of-range HH:MM time '{raw}'")
    return hour, minute


class QuietHoursWib(BaseModel):
    """Daily quiet hours window in WIB (UTC+7); alerts held here go to daily digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    start: str = "23:00"
    end: str = "06:00"

    @field_validator("start", "end")
    @classmethod
    def _validate_hhmm(cls, value: str) -> str:
        _parse_hhmm(value)
        return value.strip()

    def is_quiet(self, dt: datetime) -> bool:
        """Return True if `dt` falls within `[start, end)` in WIB (supports overnight wrap)."""
        if not self.enabled:
            return False
        wib_dt = dt.astimezone(WIB) if dt.tzinfo is not None else dt.replace(tzinfo=WIB)
        now_mins = wib_dt.hour * 60 + wib_dt.minute
        sh, sm = _parse_hhmm(self.start)
        eh, em = _parse_hhmm(self.end)
        start_mins = sh * 60 + sm
        end_mins = eh * 60 + em
        if start_mins == end_mins:
            return False
        if start_mins < end_mins:
            return start_mins <= now_mins < end_mins
        return now_mins >= start_mins or now_mins < end_mins


class MtfTimeframesConfig(BaseModel):
    """Multi-timeframe top-down alert configuration (bias -> setup -> confirm)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bias: tuple[Timeframe, ...] = (Timeframe.H4, Timeframe.H1)
    setup: Timeframe = Timeframe.M15
    confirm: Timeframe = Timeframe.M5

    @field_validator("bias", mode="before")
    @classmethod
    def _parse_bias(cls, value: object) -> tuple[Timeframe, ...]:
        if isinstance(value, str):
            raw = [p.strip().lower() for p in value.split(",") if p.strip()]
        elif isinstance(value, (list, tuple)):
            raw = [
                item.value if isinstance(item, Timeframe) else str(item).strip().lower()
                for item in value
                if str(item).strip()
            ]
        else:
            raise ValueError("timeframes.bias must be a list of timeframes")
        if not raw:
            return (Timeframe.H4, Timeframe.H1)
        return tuple(dict.fromkeys(Timeframe(item) for item in raw))

    @field_validator("setup", "confirm", mode="before")
    @classmethod
    def _parse_single_tf(cls, value: object) -> Timeframe:
        if isinstance(value, Timeframe):
            return value
        return Timeframe(str(value).strip().lower())

    @property
    def all_timeframes(self) -> tuple[Timeframe, ...]:
        return tuple(dict.fromkeys((*self.bias, self.setup, self.confirm)))


class ScreenerConfig(BaseModel):
    """Typed configuration loaded from `config/screener.yaml` (supports scalp_v1, MTF & legacy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["scalp_v1", "swing_mtf"] = "swing_mtf"
    cadence_minutes: int = Field(default=15, ge=1)
    timeframes: tuple[Timeframe, ...] = (Timeframe.M15,)
    mtf_timeframes: MtfTimeframesConfig | None = None
    cooldown_minutes: float | None = Field(default=None, ge=0.0)
    cooldown_hours: dict[str, float] = Field(
        default_factory=lambda: {"15m": 2.0, "1h": 6.0, "4h": 12.0, "1d": 24.0}
    )
    default_cooldown_hours: float = Field(default=2.0, ge=0.0)
    max_alerts_per_run: int = Field(default=4, ge=1)
    setup_expiry_minutes: int = Field(default=90, ge=5)
    quiet_hours_wib: QuietHoursWib = Field(default_factory=QuietHoursWib)
    dead_man_hours: float = Field(default=1.0, gt=0.0)
    dead_man_minutes: float | None = Field(default=None, gt=0.0)
    universe_path: str = "config/universe.txt"
    scalp: ScalpConfig = Field(default_factory=ScalpConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_schema_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        # Alias: universe_file -> universe_path
        if "universe_file" in normalized and "universe_path" not in normalized:
            normalized["universe_path"] = normalized.pop("universe_file")
        elif "universe_file" in normalized:
            normalized.pop("universe_file")

        # Parse quiet_hours_wib when provided as a compact string ("23:00-06:00" or "off")
        qh = normalized.get("quiet_hours_wib")
        if isinstance(qh, str):
            raw_qh = qh.strip()
            if raw_qh.lower() in {"off", "none", "false", "0", ""}:
                normalized["quiet_hours_wib"] = {
                    "enabled": False,
                    "start": "23:00",
                    "end": "06:00",
                }
            elif "-" in raw_qh:
                start_s, _, end_s = raw_qh.partition("-")
                normalized["quiet_hours_wib"] = {
                    "enabled": True,
                    "start": start_s.strip(),
                    "end": end_s.strip(),
                }

        # Parse MTF timeframes mapping: {bias: [4h, 1h], setup: 15m, confirm: 5m}
        raw_tfs = normalized.get("timeframes")
        if isinstance(raw_tfs, dict):
            mtf_cfg = MtfTimeframesConfig.model_validate(raw_tfs)
            normalized["mtf_timeframes"] = mtf_cfg
            normalized["timeframes"] = (mtf_cfg.setup,)

        return normalized

    @field_validator("timeframes", mode="before")
    @classmethod
    def _parse_timeframes(cls, value: object) -> tuple[Timeframe, ...]:
        if isinstance(value, str):
            raw_items = [p.strip().lower() for p in value.split(",") if p.strip()]
        elif isinstance(value, (list, tuple)):
            raw_items = [
                item.value if isinstance(item, Timeframe) else str(item).strip().lower()
                for item in value
                if str(item).strip()
            ]
        else:
            raise ValueError("timeframes must be a list of timeframe strings or an MTF mapping")
        if not raw_items:
            return (Timeframe.H1,)
        return tuple(dict.fromkeys(Timeframe(item) for item in raw_items))

    @property
    def is_scalp_mode(self) -> bool:
        return self.strategy == "scalp_v1"

    @property
    def is_mtf_mode(self) -> bool:
        return self.strategy == "swing_mtf" and self.mtf_timeframes is not None

    @property
    def effective_dead_man_minutes(self) -> float:
        if self.dead_man_minutes is not None:
            return float(self.dead_man_minutes)
        if self.is_scalp_mode:
            return 15.0
        return float(self.dead_man_hours) * 60.0

    @property
    def effective_universe_path(self) -> str:
        if self.is_scalp_mode:
            return self.scalp.universe_path
        return self.universe_path

    def cooldown_for(self, timeframe: Timeframe) -> timedelta:
        if timeframe.value in self.cooldown_hours:
            return timedelta(hours=max(0.0, float(self.cooldown_hours[timeframe.value])))
        if self.cooldown_minutes is not None:
            return timedelta(minutes=max(0.0, float(self.cooldown_minutes)))
        return timedelta(hours=max(0.0, float(self.default_cooldown_hours)))


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none", "~"}:
        return None
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a clean YAML subset (top-level keys, up to 2-level nested dicts/lists, comments)."""
    root: dict[str, Any] = {}
    current_key: str | None = None
    current_subkey: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        # Strip inline comment outside quotes
        if "#" in stripped and not stripped.startswith(('"', "'")):
            parts = stripped.split("#", 1)
            stripped = parts[0].strip()
            if not stripped:
                continue

        if indent == 0:
            if ":" not in stripped:
                continue
            key, _, rest = stripped.partition(":")
            key = key.strip()
            rest = rest.strip()
            current_subkey = None
            if rest:
                root[key] = _parse_scalar(rest)
                current_key = None
            else:
                current_key = key
        elif current_key is not None:
            if indent >= 4 and current_subkey is not None and ":" in stripped:
                sub2_k, _, sub2_v = stripped.partition(":")
                parent_map = root.get(current_key)
                if not isinstance(parent_map, dict):
                    parent_map = {}
                    root[current_key] = parent_map
                nested_map = parent_map.get(current_subkey)
                if not isinstance(nested_map, dict):
                    nested_map = {}
                    parent_map[current_subkey] = nested_map
                nested_map[sub2_k.strip()] = _parse_scalar(sub2_v)
            elif stripped.startswith("- "):
                item_val = _parse_scalar(stripped[2:])
                existing = root.get(current_key)
                if not isinstance(existing, list):
                    existing = []
                    root[current_key] = existing
                existing.append(item_val)
            elif ":" in stripped:
                sub_k, _, sub_v = stripped.partition(":")
                sub_k = sub_k.strip()
                sub_v = sub_v.strip()
                existing_map = root.get(current_key)
                if not isinstance(existing_map, dict):
                    existing_map = {}
                    root[current_key] = existing_map
                if sub_v:
                    existing_map[sub_k] = _parse_scalar(sub_v)
                    current_subkey = None
                else:
                    existing_map[sub_k] = {}
                    current_subkey = sub_k
    return root


def _apply_quiet_hours_env(cfg_dict: dict[str, Any]) -> None:
    env_quiet = os.environ.get("QUIET_HOURS", "").strip()
    if not env_quiet:
        return
    if env_quiet.lower() in {"off", "none", "false", "0"}:
        cfg_dict["quiet_hours_wib"] = {"enabled": False, "start": "23:00", "end": "06:00"}
        return
    if "-" in env_quiet:
        start_s, _, end_s = env_quiet.partition("-")
        cfg_dict["quiet_hours_wib"] = {
            "enabled": True,
            "start": start_s.strip(),
            "end": end_s.strip(),
        }


def _read_env_file_values(env_path: Path = Path(".env")) -> dict[str, str]:
    """Read simple KEY=VALUE pairs from `.env` without modifying process env."""
    if not env_path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            out[k] = v
    return out


def _apply_scalp_position_env(cfg_dict: dict[str, Any]) -> None:
    """Populate `scalp.position` from `SCALP_EQUITY_USD` / `SCALP_RISK_PER_TRADE_PCT`."""
    env_file_vals = _read_env_file_values()
    raw_eq = os.environ.get("SCALP_EQUITY_USD", "").strip() or env_file_vals.get(
        "SCALP_EQUITY_USD", ""
    )
    raw_risk = os.environ.get("SCALP_RISK_PER_TRADE_PCT", "").strip() or env_file_vals.get(
        "SCALP_RISK_PER_TRADE_PCT", ""
    )
    if not raw_eq and not raw_risk:
        return
    scalp_map = cfg_dict.get("scalp")
    if not isinstance(scalp_map, dict):
        scalp_map = {}
        cfg_dict["scalp"] = scalp_map
    pos_map = scalp_map.get("position")
    if not isinstance(pos_map, dict):
        pos_map = {}
        scalp_map["position"] = pos_map
    if raw_eq:
        pos_map["equity_usd"] = float(raw_eq)
    if raw_risk:
        pos_map["risk_per_trade_pct"] = float(raw_risk)


def load_screener_config(path: Path | str | None = None) -> ScreenerConfig:
    """Load `ScreenerConfig` from YAML file (defaults to `config/screener.yaml`) + env overrides."""
    target = Path(path) if path is not None else DEFAULT_SCREENER_CONFIG_PATH
    resolved = _find_config_file(target)
    raw_data: dict[str, Any] = {}
    if resolved is not None:
        raw_data = _parse_simple_yaml(resolved.read_text(encoding="utf-8"))
    _apply_quiet_hours_env(raw_data)
    _apply_scalp_position_env(raw_data)
    return ScreenerConfig.model_validate(raw_data)
