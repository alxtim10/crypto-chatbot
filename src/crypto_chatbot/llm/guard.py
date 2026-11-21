"""Strict LLM output guard enforcing PROJECT_PLAN §5.9 boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.llm import LlmAnalysisNarration

_BRACED_TOKEN_RE = re.compile(r"\{([^}]*)\}")
_VALID_FACT_BODY_RE = re.compile(r"^F[1-9][0-9]*$")
_TIMEFRAME_TOKEN_RE = re.compile(r"\b(?:1h|4h|1d)\b", re.IGNORECASE)
_PAIR_MENTION_RE = re.compile(r"\b[A-Z0-9]{2,10}/USDT(?::USDT)?\b")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)

OVERCONFIDENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pasti",
        re.compile(
            r"(?<!\btidak\s)(?<!\bbelum\s)(?<!\btak\s)(?<!\bbukan\s)\bpasti\b",
            re.IGNORECASE,
        ),
    ),
    ("dijamin", re.compile(r"\bdijamin\b", re.IGNORECASE)),
    ("jaminan", re.compile(r"\bjaminan\b", re.IGNORECASE)),
    ("garansi", re.compile(r"\bgaransi\b", re.IGNORECASE)),
    ("tanpa risiko", re.compile(r"\btanpa\s+risiko\b", re.IGNORECASE)),
    ("bebas risiko", re.compile(r"\bbebas\s+risiko\b", re.IGNORECASE)),
    ("wajib beli", re.compile(r"\bwajib\s+beli\b", re.IGNORECASE)),
    ("wajib jual", re.compile(r"\bwajib\s+jual\b", re.IGNORECASE)),
    ("beli sekarang", re.compile(r"\bbeli\s+sekarang\b", re.IGNORECASE)),
    ("jual sekarang", re.compile(r"\bjual\s+sekarang\b", re.IGNORECASE)),
    ("buy sekarang", re.compile(r"\bbuy\s+sekarang\b", re.IGNORECASE)),
    ("sell sekarang", re.compile(r"\bsell\s+sekarang\b", re.IGNORECASE)),
    ("all in", re.compile(r"\ball[\s-]+in\b", re.IGNORECASE)),
    ("cuan pasti", re.compile(r"\bcuan\s+pasti\b", re.IGNORECASE)),
    ("profit pasti", re.compile(r"\bprofit\s+pasti\b", re.IGNORECASE)),
    ("guaranteed", re.compile(r"\bguaranteed\b", re.IGNORECASE)),
    ("risk-free", re.compile(r"\brisk[\s-]+free\b", re.IGNORECASE)),
    ("sure win", re.compile(r"\bsure\s+win\b", re.IGNORECASE)),
    ("to the moon", re.compile(r"\bto\s+the\s+moon\b", re.IGNORECASE)),
)


class GuardResult(BaseModel):
    """Validation outcome for structured LLM narration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    reasons: tuple[str, ...] = ()


def _int_param(params: Mapping[str, int | float | str], *keys: str) -> int | None:
    for key in keys:
        val = params.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return None


def _build_allowlist_patterns(
    indicator_parameters: Mapping[str, int | float | str],
    allowed_symbols: tuple[str, ...],
) -> tuple[re.Pattern[str], ...]:
    patterns: list[re.Pattern[str]] = [_TIMEFRAME_TOKEN_RE]

    ema_periods = [
        p
        for p in (
            _int_param(indicator_parameters, "ema_fast"),
            _int_param(indicator_parameters, "ema_mid"),
            _int_param(indicator_parameters, "ema_slow"),
        )
        if p is not None
    ]
    if ema_periods:
        joined = "|".join(str(p) for p in ema_periods)
        patterns.append(re.compile(rf"\bEMA\s*(?:{joined})\b", re.IGNORECASE))

    rsi_p = _int_param(indicator_parameters, "rsi_period", "rsi")
    if rsi_p is not None:
        patterns.append(re.compile(rf"\bRSI\s*{rsi_p}\b", re.IGNORECASE))

    macd_f = _int_param(indicator_parameters, "macd_fast")
    macd_s = _int_param(indicator_parameters, "macd_slow")
    macd_sig = _int_param(indicator_parameters, "macd_signal")
    if macd_f is not None and macd_s is not None:
        if macd_sig is not None:
            patterns.append(
                re.compile(rf"\bMACD\s*{macd_f}/{macd_s}(?:/{macd_sig})?\b", re.IGNORECASE)
            )
        else:
            patterns.append(re.compile(rf"\bMACD\s*{macd_f}/{macd_s}\b", re.IGNORECASE))

    atr_p = _int_param(indicator_parameters, "atr_period", "atr")
    if atr_p is not None:
        patterns.append(re.compile(rf"\bATR%?\s*{atr_p}\b", re.IGNORECASE))

    adx_p = _int_param(indicator_parameters, "adx_period", "adx")
    if adx_p is not None:
        patterns.append(re.compile(rf"(?:\bADX|\bDMI|\+DI|-DI)\s*{adx_p}\b", re.IGNORECASE))

    rvol_p = _int_param(indicator_parameters, "relative_volume_period")
    if rvol_p is not None:
        patterns.append(
            re.compile(rf"\b(?:relative\s+volume|volume\s+relatif)\s*{rvol_p}\b", re.IGNORECASE)
        )

    atr_win = _int_param(indicator_parameters, "atr_percentile_window")
    if atr_win is not None:
        patterns.append(
            re.compile(rf"\b(?:ATR\s+percentile|persentil\s+ATR)\s*{atr_win}\b", re.IGNORECASE)
        )

    for symbol in allowed_symbols:
        if symbol:
            patterns.append(re.compile(re.escape(symbol), re.IGNORECASE))

    return tuple(patterns)


def _strip_valid_placeholders(text: str, known_fact_ids: set[str]) -> tuple[str, bool]:
    """Validate brace syntax and replace known `{F<n>}` placeholders with spaces."""
    bad = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal bad
        token = match.group(1)
        if not _VALID_FACT_BODY_RE.match(token) or token not in known_fact_ids:
            bad = True
            return match.group(0)
        return " "

    stripped = _BRACED_TOKEN_RE.sub(_replace, text)
    if "{" in stripped or "}" in stripped:
        bad = True
    return stripped, bad


def validate_narration(
    narration: LlmAnalysisNarration,
    snapshot: AnalysisSnapshot,
) -> GuardResult:
    """Validate structured LLM narration strictly against backend `AnalysisSnapshot` facts."""
    reasons: list[str] = []
    known_fact_ids = {fact.fact_id for fact in snapshot.facts}

    for fid in narration.fact_ids:
        if fid not in known_fact_ids:
            reasons.append(f"unknown_fact_id:{fid}")

    allowlist = _build_allowlist_patterns(
        snapshot.indicator_parameters,
        (snapshot.canonical_symbol, snapshot.provider_symbol),
    )

    text_fields = [
        narration.summary,
        *narration.supporting_factors,
        *narration.contradicting_factors,
        narration.bullish_scenario,
        narration.bearish_scenario,
        *narration.limitations,
    ]

    for text in text_fields:
        scrubbed, bad_braces = _strip_valid_placeholders(text, known_fact_ids)
        if bad_braces:
            reasons.append("malformed_or_unknown_placeholder")

        for pattern in allowlist:
            scrubbed = pattern.sub(" ", scrubbed)
        if re.search(r"\d", scrubbed):
            reasons.append("raw_digit_detected")

        normalized = " ".join(text.split())
        for term, pattern in OVERCONFIDENCE_PATTERNS:
            if pattern.search(normalized):
                reasons.append(f"overconfidence_lexicon:{term}")

        if _URL_RE.search(text):
            reasons.append("url_not_allowed")

        if any(secret_re.search(text) for secret_re in _SECRET_PATTERNS):
            reasons.append("secret_pattern_detected")

        for pair in _PAIR_MENTION_RE.findall(text.upper()):
            if pair != snapshot.canonical_symbol.upper():
                reasons.append("unsupported_symbol_mentioned")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return GuardResult(is_valid=not unique_reasons, reasons=unique_reasons)
