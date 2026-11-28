"""SC-03: Weekly scalp universe eligibility filter & PIT snapshot writer (SCALP-SPEC-v1 §1)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.scalp.config import DEFAULT_SCALP_UNIVERSE_PATH, ScalpUniverseConfig
from crypto_chatbot.storage.repositories import (
    PostgresScalpUniverseRepository,
    ScalpUniverseSnapshotRecord,
)

STABLECOIN_BASES: frozenset[str] = frozenset(
    {
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "TUSD",
        "FDUSD",
        "USDP",
        "USDD",
        "PYUSD",
        "USDE",
        "EURC",
        "RLUSD",
    }
)

TRADFI_BASES: frozenset[str] = frozenset(
    {
        "XAU",
        "XAG",
        "PAXG",
        "XAUT",
        "SPY",
        "QQQ",
        "NVDA",
        "AAPL",
        "TSLA",
        "EUR",
        "GBP",
        "JPY",
        "DXY",
        "WTI",
        "BRENT",
    }
)


class ScalpUniverseCandidate(BaseModel):
    """Point-in-time market metrics for a candidate Gate USDT perpetual contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str
    base_asset: str
    quote_volume_24h_usd: float = Field(ge=0.0)
    median_spread_7d_bps: float | None = Field(default=None, ge=0.0)
    median_bid_depth_10bps_usd: float | None = Field(default=None, ge=0.0)
    median_ask_depth_10bps_usd: float | None = Field(default=None, ge=0.0)
    listing_age_days: int | None = Field(default=None, ge=0)
    is_tradfi: bool = False
    is_stablecoin: bool = False


class ScalpUniverseEvaluation(BaseModel):
    """Eligibility decision and rejection reasons for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str
    base_asset: str
    eligible: bool
    rejection_reasons: tuple[str, ...] = ()
    quote_volume_24h_usd: float
    median_spread_7d_bps: float | None = None
    median_bid_depth_10bps_usd: float | None = None
    median_ask_depth_10bps_usd: float | None = None
    listing_age_days: int | None = None


class ScalpUniverseRefreshResult(BaseModel):
    """Result of a weekly idempotent scalp universe refresh."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = "scalp_v1"
    snapshot_week_utc: str
    venue: str
    captured_at: datetime
    eligible_symbols: tuple[str, ...]
    evaluations: tuple[ScalpUniverseEvaluation, ...]
    output_path: str | None = None
    db_snapshot_inserted: bool = False


def format_snapshot_week_utc(ts_utc: datetime) -> str:
    """Return ISO week key `YYYY-Www` in UTC for idempotent weekly snapshots."""
    iso_year, iso_week, _ = ts_utc.astimezone(UTC).isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def evaluate_scalp_universe_eligibility(
    candidates: Sequence[ScalpUniverseCandidate],
    *,
    config: ScalpUniverseConfig | None = None,
) -> tuple[ScalpUniverseEvaluation, ...]:
    """Apply SCALP-SPEC-v1 §1 eligibility rules deterministically to all candidates."""
    cfg = config or ScalpUniverseConfig()
    results: list[ScalpUniverseEvaluation] = []

    for cand in candidates:
        base = cand.base_asset.strip().upper()
        reasons: list[str] = []

        if cand.is_stablecoin or base in STABLECOIN_BASES:
            reasons.append("EXCLUDED_STABLECOIN")
        if cand.is_tradfi or base in TRADFI_BASES:
            reasons.append("EXCLUDED_TRADFI")
        if cand.quote_volume_24h_usd < cfg.min_vol_24h_usd:
            vol_m = cand.quote_volume_24h_usd / 1e6
            min_m = cfg.min_vol_24h_usd / 1e6
            reasons.append(f"LOW_24H_VOL({vol_m:.1f}M<{min_m:.0f}M)")
        if cand.median_spread_7d_bps is None:
            reasons.append("MISSING_7D_SPREAD")
        elif cand.median_spread_7d_bps > cfg.max_spread_bps:
            reasons.append(
                f"WIDE_SPREAD({cand.median_spread_7d_bps:.2f}bps>{cfg.max_spread_bps:.2f}bps)"
            )
        if cand.median_bid_depth_10bps_usd is None or cand.median_ask_depth_10bps_usd is None:
            reasons.append("MISSING_L2_DEPTH")
        else:
            min_k = cfg.min_depth_usd / 1e3
            if cand.median_bid_depth_10bps_usd < cfg.min_depth_usd:
                bid_k = cand.median_bid_depth_10bps_usd / 1e3
                reasons.append(f"LOW_BID_DEPTH({bid_k:.0f}k<{min_k:.0f}k)")
            if cand.median_ask_depth_10bps_usd < cfg.min_depth_usd:
                ask_k = cand.median_ask_depth_10bps_usd / 1e3
                reasons.append(f"LOW_ASK_DEPTH({ask_k:.0f}k<{min_k:.0f}k)")
        if cand.listing_age_days is None:
            reasons.append("MISSING_LISTING_AGE")
        elif cand.listing_age_days < cfg.min_listing_age_days:
            reasons.append(f"NEW_LISTING({cand.listing_age_days}d<{cfg.min_listing_age_days}d)")

        results.append(
            ScalpUniverseEvaluation(
                canonical_symbol=cand.canonical_symbol,
                base_asset=base,
                eligible=not reasons,
                rejection_reasons=tuple(reasons),
                quote_volume_24h_usd=cand.quote_volume_24h_usd,
                median_spread_7d_bps=cand.median_spread_7d_bps,
                median_bid_depth_10bps_usd=cand.median_bid_depth_10bps_usd,
                median_ask_depth_10bps_usd=cand.median_ask_depth_10bps_usd,
                listing_age_days=cand.listing_age_days,
            )
        )

    # Sort eligible symbols by 24h quote volume descending for deterministic output
    results.sort(key=lambda e: (-int(e.eligible), -e.quote_volume_24h_usd, e.base_asset))
    return tuple(results)


def write_scalp_universe_file(
    base_assets: Sequence[str],
    *,
    output_path: Path = DEFAULT_SCALP_UNIVERSE_PATH,
    snapshot_week_utc: str,
) -> Path:
    """Write `config/universe_scalp.txt` deterministically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Scalp v1 universe (Gate USDT-settled perpetuals)",
        "# Generated by `crypto-chatbot universe refresh --strategy scalp_v1` "
        f"({snapshot_week_utc})",
        *[asset.strip().upper() for asset in base_assets if asset.strip()],
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def refresh_scalp_universe(
    candidates: Sequence[ScalpUniverseCandidate],
    *,
    venue: str = "gateio",
    captured_at: datetime | None = None,
    config: ScalpUniverseConfig | None = None,
    output_path: Path | None = DEFAULT_SCALP_UNIVERSE_PATH,
    repository: PostgresScalpUniverseRepository | None = None,
) -> ScalpUniverseRefreshResult:
    """Evaluate weekly eligibility, write `config/universe_scalp.txt`, and store PIT DB snapshot."""
    cfg = config or ScalpUniverseConfig()
    now_utc = (captured_at or datetime.now(UTC)).astimezone(UTC)
    week_key = format_snapshot_week_utc(now_utc)
    evaluations = evaluate_scalp_universe_eligibility(candidates, config=cfg)
    eligible = tuple(e for e in evaluations if e.eligible)
    eligible_canonical = tuple(e.canonical_symbol for e in eligible)
    eligible_bases = tuple(e.base_asset for e in eligible)

    written_path: str | None = None
    if output_path is not None:
        write_scalp_universe_file(
            eligible_bases,
            output_path=output_path,
            snapshot_week_utc=week_key,
        )
        written_path = str(output_path)

    inserted = False
    if repository is not None:
        snap = ScalpUniverseSnapshotRecord(
            strategy="scalp_v1",
            snapshot_week_utc=week_key,
            venue=venue,
            captured_at=now_utc,
            eligible_symbols=eligible_canonical,
            evaluations_json=tuple(e.model_dump(mode="json") for e in evaluations),
            rules_json=cfg.model_dump(mode="json"),
        )
        inserted = repository.save_snapshot(snap)

    return ScalpUniverseRefreshResult(
        strategy="scalp_v1",
        snapshot_week_utc=week_key,
        venue=venue,
        captured_at=now_utc,
        eligible_symbols=eligible_canonical,
        evaluations=evaluations,
        output_path=written_path,
        db_snapshot_inserted=inserted,
    )
