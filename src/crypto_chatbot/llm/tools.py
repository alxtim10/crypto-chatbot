"""Allowlisted deterministic analysis tools exposed to Gemini orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from google.genai import types
from pydantic import BaseModel, ConfigDict, ValidationError

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.commands import AnalyzeCommandArgs, CompareCommandArgs

ALLOWED_TOOL_NAMES: frozenset[str] = frozenset({"analyze_coin", "compare_coins"})
_MVP_TOOL_TIMEFRAMES = ["15m", "1h", "4h", "1d"]


class ToolError(ValueError):
    """Raised when Gemini requests an unknown tool or invalid arguments."""


class ToolExecutionResult(BaseModel):
    """Deterministic tool execution outcome containing only backend snapshots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    snapshots: tuple[AnalysisSnapshot, ...]


def build_gemini_tool_declarations() -> list[types.Tool]:
    """Return narrow `google-genai` tool declarations strictly scoped to Binance spot analysis."""
    analyze_decl = types.FunctionDeclaration(
        name="analyze_coin",
        description=(
            "Run deterministic spot market technical analysis on Binance closed candles "
            "for one supported /USDT cryptocurrency pair."
        ),
        parameters_json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Binance /USDT spot symbol or base ticker (e.g. BTC, ETH, DOGE, SUI/USDT)."
                    ),
                },
                "timeframe": {
                    "type": "string",
                    "enum": _MVP_TOOL_TIMEFRAMES,
                    "description": "Closed-candle timeframe.",
                },
            },
            "required": ["symbol", "timeframe"],
        },
    )
    compare_decl = types.FunctionDeclaration(
        name="compare_coins",
        description=(
            "Run deterministic side-by-side spot technical analysis for 2 or 3 distinct "
            "Binance /USDT spot cryptocurrency pairs on the same timeframe."
        ),
        parameters_json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "symbols": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 3,
                    "items": {
                        "type": "string",
                    },
                    "description": "Distinct Binance /USDT spot symbols or base tickers.",
                },
                "timeframe": {
                    "type": "string",
                    "enum": _MVP_TOOL_TIMEFRAMES,
                    "description": "Closed-candle timeframe.",
                },
            },
            "required": ["symbols", "timeframe"],
        },
    )
    return [types.Tool(function_declarations=[analyze_decl, compare_decl])]


class ToolRegistry:
    """Validates tool names and arguments strictly before invoking backend analysis."""

    def __init__(self, analysis_service: AnalysisService) -> None:
        self._analysis_service = analysis_service

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolExecutionResult:
        """Execute an allowlisted tool with Pydantic-validated arguments."""
        if tool_name not in ALLOWED_TOOL_NAMES:
            raise ToolError(f"tool '{tool_name}' is not in the allowed tool list")

        if not isinstance(arguments, Mapping):
            raise ToolError("tool arguments must be a mapping")

        if tool_name == "analyze_coin":
            try:
                analyze_args = AnalyzeCommandArgs.model_validate(dict(arguments))
            except ValidationError as exc:
                raise ToolError(f"invalid arguments for analyze_coin: {exc}") from None
            snapshot = await self._analysis_service.analyze(
                analyze_args.symbol,
                analyze_args.timeframe,
            )
            return ToolExecutionResult(tool_name=tool_name, snapshots=(snapshot,))

        try:
            compare_args = CompareCommandArgs.model_validate(dict(arguments))
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for compare_coins: {exc}") from None

        snapshots = await asyncio.gather(
            *(
                self._analysis_service.analyze(sym, compare_args.timeframe)
                for sym in compare_args.symbols
            )
        )
        return ToolExecutionResult(tool_name=tool_name, snapshots=tuple(snapshots))
