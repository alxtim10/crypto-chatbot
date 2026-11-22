"""Validated argument schemas for supported Telegram commands."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.market import Timeframe


def _normalize_timeframe(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower()
    return value


class AnalyzeCommandArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    detail: bool = False

    @field_validator("symbol", mode="before")
    @classmethod
    def _validate_symbol(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("symbol must be a string")
        return resolve_mvp_spot_symbol(value).canonical_symbol

    @field_validator("timeframe", mode="before")
    @classmethod
    def _validate_timeframe(cls, value: object) -> object:
        return _normalize_timeframe(value)


class CompareCommandArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: tuple[str, ...] = Field(min_length=2, max_length=3)
    timeframe: Timeframe

    @field_validator("symbols", mode="before")
    @classmethod
    def _validate_symbols(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            resolved: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError("each symbol must be a string")
                resolved.append(resolve_mvp_spot_symbol(item).canonical_symbol)
            if len(resolved) != len(set(resolved)):
                raise ValueError("symbols to compare must be distinct")
            return tuple(resolved)
        return value

    @field_validator("timeframe", mode="before")
    @classmethod
    def _validate_timeframe(cls, value: object) -> object:
        return _normalize_timeframe(value)


class WatchlistAction(StrEnum):
    LIST = "list"
    ADD = "add"
    REMOVE = "remove"


class WatchlistCommandArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: WatchlistAction = WatchlistAction.LIST
    symbol: str | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("symbol", mode="before")
    @classmethod
    def _validate_symbol(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("symbol must be a string")
        return resolve_mvp_spot_symbol(value).canonical_symbol

    @model_validator(mode="after")
    def _validate_action_symbol(self) -> WatchlistCommandArgs:
        if self.action is WatchlistAction.LIST and self.symbol is not None:
            raise ValueError("list action does not accept a symbol")
        if self.action in (WatchlistAction.ADD, WatchlistAction.REMOVE) and self.symbol is None:
            raise ValueError(f"{self.action.value} action requires a symbol")
        return self
