"""Strict LLM output guard enforcing PROJECT_PLAN §5.9 boundaries."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.schemas.analysis import AnalysisSnapshot, FactStatus, SetupBias, SetupStatus
from crypto_chatbot.schemas.llm import LlmAnalysisNarration

logger = logging.getLogger(__name__)

_BULLISH_BIAS_RE = re.compile(r"\b(?:bullish|long)\b", re.IGNORECASE)
_BEARISH_BIAS_RE = re.compile(r"\b(?:bearish|short)\b", re.IGNORECASE)


def check_directional_consistency(
    text: str,
    bias: SetupBias | None,
) -> tuple[str, ...]:
    """Check whether directional bias words in `text` contradict `bias` (PRD v1.7 §2.3, T-31)."""
    if bias is SetupBias.SHORT and _BULLISH_BIAS_RE.search(text):
        return ("directional_inconsistency:contradicts_short",)
    if bias is SetupBias.LONG and _BEARISH_BIAS_RE.search(text):
        return ("directional_inconsistency:contradicts_long",)
    return ()


_BRACED_TOKEN_RE = re.compile(r"\{([^}]*)\}")
_VALID_FACT_BODY_RE = re.compile(r"^F[1-9][0-9]*$")
_TIMEFRAME_TOKEN_RE = re.compile(r"\b(?:15m|15\s*menit|1h|4h|1d)\b", re.IGNORECASE)
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
            r"\bpasti\s+(?:naik|turun|untung|cuan|profit|terbang|tembus|menang|aman|balik\s+modal|hit\s+tp\d?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "100% claim",
        re.compile(
            r"\b100%\s*(?:profit|cuan|untung|aman|akurat|menang|win|berhasil|pasti)\b",
            re.IGNORECASE,
        ),
    ),
    ("dijamin", re.compile(r"\bdijamin\b", re.IGNORECASE)),
    ("jaminan", re.compile(r"\bjaminan\b", re.IGNORECASE)),
    ("garansi", re.compile(r"\bgaransi\b", re.IGNORECASE)),
    ("tanpa risiko", re.compile(r"\btanpa\s+risiko\b", re.IGNORECASE)),
    ("bebas risiko", re.compile(r"\bbebas\s+risiko\b", re.IGNORECASE)),
    ("tidak ada risiko", re.compile(r"\btidak\s+ada\s+risiko\b", re.IGNORECASE)),
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

_CLAUSE_SPLIT_RE = re.compile(r"[.!?;:,\n]+")
_CLAIM_NEGATOR_RE = re.compile(
    r"\b(?:tidak|tak|belum|bukan|bukanlah|jangan|hindari|dilarang|tanpa|berharap|never|not|no|avoid)\b",
    re.IGNORECASE,
)
_RISK_NEGATOR_RE = re.compile(
    r"\b(?:tidak|tak|bukan|bukanlah|belum|tidaklah|not|never)\b",
    re.IGNORECASE,
)
_RISK_TERMS = frozenset({"tanpa risiko", "bebas risiko", "risk-free"})


def _find_overconfidence_hits(text: str) -> list[str]:
    """Detect positive overconfidence/guarantee claims while allowing negated disclaimers."""
    hits: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        clause_norm = " ".join(clause.split())
        if not clause_norm:
            continue
        for term, pattern in OVERCONFIDENCE_PATTERNS:
            match = pattern.search(clause_norm)
            if match is None:
                continue
            if term == "tidak ada risiko":
                hits.append("overconfidence_lexicon:tanpa risiko")
                continue
            prefix = clause_norm[: match.start()]
            negator = _RISK_NEGATOR_RE if term in _RISK_TERMS else _CLAIM_NEGATOR_RE
            if negator.search(prefix) is not None:
                continue
            hits.append(f"overconfidence_lexicon:{term}")
    return hits


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

    funding_hours = _int_param(indicator_parameters, "funding_norm_interval_hours")
    if funding_hours is not None:
        patterns.append(re.compile(rf"\b{funding_hours}h\b", re.IGNORECASE))

    sentiment_hours = _int_param(indicator_parameters, "sentiment_window_hours")
    if sentiment_hours is not None:
        patterns.append(re.compile(rf"\b{sentiment_hours}h\b", re.IGNORECASE))

    for band_key in ("orderbook_depth_band_tight_bps", "orderbook_depth_band_wide_bps"):
        band_bps = _int_param(indicator_parameters, band_key)
        if band_bps is not None:
            patterns.append(re.compile(rf"\b{band_bps}\s*bps\b", re.IGNORECASE))

    tp_levels = _int_param(indicator_parameters, "trade_setup_tp_levels")
    if tp_levels is not None and tp_levels >= 3:
        patterns.append(re.compile(r"\bTP\s*[123]\b", re.IGNORECASE))

    for symbol in allowed_symbols:
        if symbol:
            patterns.append(re.compile(re.escape(symbol), re.IGNORECASE))

    return tuple(patterns)


def _strip_valid_placeholders(
    text: str,
    known_fact_ids: set[str],
    none_fact_ids: set[str] | None = None,
) -> tuple[str, bool, list[str]]:
    """Validate brace syntax and replace known non-NONE `{F<n>}` placeholders with spaces."""
    bad = False
    none_refs: list[str] = []
    blocked = none_fact_ids or set()

    def _replace(match: re.Match[str]) -> str:
        nonlocal bad
        token = match.group(1)
        if not _VALID_FACT_BODY_RE.match(token) or token not in known_fact_ids:
            bad = True
            return match.group(0)
        if token in blocked:
            none_refs.append(token)
            return match.group(0)
        return " "

    stripped = _BRACED_TOKEN_RE.sub(_replace, text)
    if "{" in stripped or "}" in stripped:
        bad = True
    return stripped, bad, none_refs


def log_guard_violations(
    reasons: Sequence[str],
    *,
    model_version: str,
    attempt: int = 1,
) -> None:
    """Log every guard violation with `violation_type` and `model_version` (PRD v1.7 §2.3)."""
    for reason in reasons:
        logger.warning(
            "llm_guard_violation violation_type=%s model_version=%s attempt=%d",
            reason,
            model_version,
            attempt,
        )


def validate_narration(
    narration: LlmAnalysisNarration,
    snapshot: AnalysisSnapshot,
) -> GuardResult:
    """Validate structured LLM narration strictly against backend `AnalysisSnapshot` facts."""
    reasons: list[str] = []
    known_fact_ids = {fact.fact_id for fact in snapshot.facts}
    none_fact_ids = {fact.fact_id for fact in snapshot.facts if fact.status is FactStatus.NONE}

    for fid in narration.fact_ids:
        if fid not in known_fact_ids:
            reasons.append(f"unknown_fact_id:{fid}")
        elif fid in none_fact_ids:
            reasons.append(f"fact_status_none:{fid}")

    base_ticker = snapshot.canonical_symbol.split("/")[0]
    allowed_symbols: list[str] = [snapshot.canonical_symbol, snapshot.provider_symbol, base_ticker]
    allowed_pairs: set[str] = {snapshot.canonical_symbol.upper()}
    if snapshot.derivatives is not None:
        for deriv_sym in (
            snapshot.derivatives.funding_symbol,
            snapshot.derivatives.open_interest_symbol,
            snapshot.derivatives.orderbook_symbol,
        ):
            if deriv_sym:
                allowed_symbols.append(deriv_sym)
                if "/" in deriv_sym:
                    allowed_pairs.add(deriv_sym.upper())

    allowlist = _build_allowlist_patterns(
        snapshot.indicator_parameters,
        tuple(dict.fromkeys(allowed_symbols)),
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
        scrubbed, bad_braces, none_refs = _strip_valid_placeholders(
            text, known_fact_ids, none_fact_ids
        )
        for fid in none_refs:
            reasons.append(f"fact_status_none:{fid}")
        if bad_braces:
            reasons.append("malformed_or_unknown_placeholder")

        for pattern in allowlist:
            scrubbed = pattern.sub(" ", scrubbed)
        if re.search(r"\d", scrubbed):
            reasons.append("raw_digit_detected")

        normalized = " ".join(text.split())
        reasons.extend(_find_overconfidence_hits(normalized))

        if _URL_RE.search(text):
            reasons.append("url_not_allowed")

        if any(secret_re.search(text) for secret_re in _SECRET_PATTERNS):
            reasons.append("secret_pattern_detected")

        for pair in _PAIR_MENTION_RE.findall(text.upper()):
            if pair not in allowed_pairs:
                reasons.append("unsupported_symbol_mentioned")

    if (
        snapshot.trade_setup is not None
        and snapshot.trade_setup.status is SetupStatus.READY
        and snapshot.trade_setup.bias in (SetupBias.LONG, SetupBias.SHORT)
    ):
        reasons.extend(check_directional_consistency(narration.summary, snapshot.trade_setup.bias))

    unique_reasons = tuple(dict.fromkeys(reasons))
    return GuardResult(is_valid=not unique_reasons, reasons=unique_reasons)


_QA_UNSAFE_ADVICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "direct_execution_instruction",
        re.compile(
            r"\b(?:eksekusi\s+order|open\s+(?:long|short)\s+sekarang|"
            r"masuk\s+(?:long|short)\s+sekarang|taruh\s+\d+%\s+modal|"
            r"alokasikan\s+\d+%)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fabricated_live_price",
        re.compile(
            r"\bharga\s+(?:saat\s+ini|sekarang|live|terkini)\s+"
            r"(?:adalah|berada\s+di|sekitar)\s+\$?\d",
            re.IGNORECASE,
        ),
    ),
)


def validate_conversational_answer(text: str) -> GuardResult:
    """Validate free-form conversational QA answers against safety and compliance rules."""
    cleaned = text.strip()
    if not cleaned:
        return GuardResult(is_valid=False, reasons=("empty_response",))

    reasons: list[str] = []
    normalized = " ".join(cleaned.split())
    reasons.extend(_find_overconfidence_hits(normalized))

    for label, pattern in _QA_UNSAFE_ADVICE_PATTERNS:
        if pattern.search(normalized):
            reasons.append(label)

    if _URL_RE.search(cleaned):
        reasons.append("url_not_allowed")

    if any(secret_re.search(cleaned) for secret_re in _SECRET_PATTERNS):
        reasons.append("secret_pattern_detected")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return GuardResult(is_valid=not unique_reasons, reasons=unique_reasons)
