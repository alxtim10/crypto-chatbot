"""Universe loader and stdlib YAML configuration for the hourly alert worker."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from crypto_chatbot.market_data.symbols import SymbolError, resolve_default_perp_symbol
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


class ScreenerConfig(BaseModel):
    """Typed configuration loaded from `config/screener.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframes: tuple[Timeframe, ...] = (Timeframe.H1,)
    cooldown_hours: dict[str, float] = Field(
        default_factory=lambda: {"1h": 6.0, "4h": 12.0, "1d": 24.0}
    )
    default_cooldown_hours: float = Field(default=6.0, ge=0.0)
    max_alerts_per_run: int = Field(default=4, ge=1)
    quiet_hours_wib: QuietHoursWib = Field(default_factory=QuietHoursWib)
    dead_man_hours: float = Field(default=2.0, gt=0.0)
    universe_path: str = "config/universe.txt"

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
            raise ValueError("timeframes must be a list of timeframe strings")
        if not raw_items:
            return (Timeframe.H1,)
        return tuple(dict.fromkeys(Timeframe(item) for item in raw_items))

    def cooldown_for(self, timeframe: Timeframe) -> timedelta:
        hours = self.cooldown_hours.get(timeframe.value, self.default_cooldown_hours)
        return timedelta(hours=max(0.0, float(hours)))


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
    """Parse a clean subset of YAML (top-level keys, 1-level nested dicts/lists, comments)."""
    root: dict[str, Any] = {}
    current_key: str | None = None
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
            if rest:
                root[key] = _parse_scalar(rest)
                current_key = None
            else:
                current_key = key
        elif current_key is not None:
            if stripped.startswith("- "):
                item_val = _parse_scalar(stripped[2:])
                existing = root.get(current_key)
                if not isinstance(existing, list):
                    existing = []
                    root[current_key] = existing
                existing.append(item_val)
            elif ":" in stripped:
                sub_k, _, sub_v = stripped.partition(":")
                existing_map = root.get(current_key)
                if not isinstance(existing_map, dict):
                    existing_map = {}
                    root[current_key] = existing_map
                existing_map[sub_k.strip()] = _parse_scalar(sub_v)
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


def load_screener_config(path: Path | str | None = None) -> ScreenerConfig:
    """Load `ScreenerConfig` from YAML file (defaults to `config/screener.yaml`) + env overrides."""
    target = Path(path) if path is not None else DEFAULT_SCREENER_CONFIG_PATH
    resolved = _find_config_file(target)
    raw_data: dict[str, Any] = {}
    if resolved is not None:
        raw_data = _parse_simple_yaml(resolved.read_text(encoding="utf-8"))
    _apply_quiet_hours_env(raw_data)
    return ScreenerConfig.model_validate(raw_data)
