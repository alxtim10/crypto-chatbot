"""SC-07: Point-in-time sentiment & flow feature snapshot for `scalp_v1` (SCALP-SPEC-v1 §7).

Every feature is strictly filtered by `available_time <= decision_time` so the resulting
dataset on READY, WAIT, and NO_TRADE decisions is 100% point-in-time clean for §9.5 research.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.schemas.derivatives import ContractStatsObservation, FundingRateObservation
from crypto_chatbot.schemas.news import FearGreedObservation, NormalizedNewsArticle

OiRegimeLabel = Literal[
    "long_buildup",
    "short_buildup",
    "long_flush",
    "short_flush",
    "flat",
    "unavailable",
]


class ScalpFeatureSnapshot(BaseModel):
    """Point-in-time §7 feature vector stored on every scalp decision row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_time_utc: datetime
    # Taker buy/sell ratio (5m, last 3 bars) from contract_stats
    taker_buy_sell_ratio_5m_3bar: float | None = None
    # CVD ratio 15m (normalized net taker imbalance across last 3 x 5m bars)
    cvd_ratio_15m: float | None = None
    # OI change 1h (%) vs price change 1h (%) + buildup/flush label
    oi_change_1h_pct: float | None = None
    price_change_1h_pct: float | None = None
    oi_price_label: OiRegimeLabel = "unavailable"
    # Top-trader L/S ratio
    top_trader_ls_ratio: float | None = None
    # Signed 8h funding rate (decimal)
    funding_rate_8h: float | None = None
    # Liquidations 5m (USD)
    long_liq_5m_usd: float | None = None
    short_liq_5m_usd: float | None = None
    total_liq_5m_usd: float | None = None
    # News count / tone in last 30m (tone never directional in v1)
    news_count_30m: int = 0
    news_tone_30m_label: str | None = None
    # Daily Fear & Greed context feature
    fear_greed_value: int | None = None
    fear_greed_classification: str | None = None


def _classify_oi_price_regime(
    oi_change_pct: float | None,
    price_change_pct: float | None,
    *,
    flat_threshold_pct: float = 0.05,
) -> OiRegimeLabel:
    if oi_change_pct is None or price_change_pct is None:
        return "unavailable"
    if abs(oi_change_pct) < flat_threshold_pct or abs(price_change_pct) < flat_threshold_pct:
        return "flat"
    if oi_change_pct > 0.0 and price_change_pct > 0.0:
        return "long_buildup"
    if oi_change_pct > 0.0 and price_change_pct < 0.0:
        return "short_buildup"
    if oi_change_pct < 0.0 and price_change_pct < 0.0:
        return "long_flush"
    return "short_flush"


def extract_scalp_features_pit(
    *,
    decision_time_utc: datetime,
    canonical_symbol: str,
    contract_stats_5m: Sequence[ContractStatsObservation] = (),
    funding_observations: Sequence[FundingRateObservation] = (),
    news_articles: Sequence[NormalizedNewsArticle] = (),
    fear_greed_observations: Sequence[FearGreedObservation] = (),
) -> ScalpFeatureSnapshot:
    """Build PIT `ScalpFeatureSnapshot` using only rows with `available_time <= decision_time`."""
    cutoff = decision_time_utc.astimezone(UTC)
    base_asset = canonical_symbol.split("/", 1)[0].upper()

    # 1. Contract stats (5m) strictly PIT
    pit_stats = sorted(
        (
            obs
            for obs in contract_stats_5m
            if obs.available_time.astimezone(UTC) <= cutoff
            and obs.observed_time.astimezone(UTC) <= cutoff
        ),
        key=lambda o: o.observed_time.astimezone(UTC),
    )

    taker_ratio_3bar: float | None = None
    cvd_ratio_15m: float | None = None
    oi_change_1h_pct: float | None = None
    price_change_1h_pct: float | None = None
    top_ls: float | None = None
    long_liq: float | None = None
    short_liq: float | None = None
    total_liq: float | None = None

    if pit_stats:
        latest = pit_stats[-1]
        top_ls = latest.top_lsr_size if latest.top_lsr_size is not None else latest.top_lsr_account
        long_liq = latest.long_liq_usd
        short_liq = latest.short_liq_usd
        if long_liq is not None or short_liq is not None:
            total_liq = (long_liq or 0.0) + (short_liq or 0.0)

        last3 = pit_stats[-3:]
        ratios = [o.lsr_taker for o in last3 if o.lsr_taker is not None and o.lsr_taker >= 0.0]
        if ratios:
            taker_ratio_3bar = sum(ratios) / len(ratios)
            # Normalized CVD proxy from taker L/S ratio: (R - 1) / (R + 1) in [-1, 1]
            imbalances = [(r - 1.0) / (r + 1.0) for r in ratios]
            cvd_ratio_15m = sum(imbalances) / len(imbalances)

        # 1h OI change & price change (nearest observation <= latest - 1h)
        target_1h = latest.observed_time.astimezone(UTC) - timedelta(hours=1)
        older = [o for o in pit_stats[:-1] if o.observed_time.astimezone(UTC) <= target_1h]
        if older:
            ref = older[-1]
            if (
                latest.open_interest_usd is not None
                and ref.open_interest_usd is not None
                and ref.open_interest_usd > 0.0
            ):
                oi_change_1h_pct = (
                    (latest.open_interest_usd - ref.open_interest_usd) / ref.open_interest_usd
                ) * 100.0
            if (
                latest.mark_price is not None
                and ref.mark_price is not None
                and ref.mark_price > 0.0
            ):
                price_change_1h_pct = (
                    (latest.mark_price - ref.mark_price) / ref.mark_price
                ) * 100.0

    # 2. Funding rate 8h strictly PIT
    pit_funding = sorted(
        (obs for obs in funding_observations if obs.available_time.astimezone(UTC) <= cutoff),
        key=lambda o: o.available_time.astimezone(UTC),
    )
    funding_rate_8h: float | None = None
    if pit_funding:
        funding_rate_8h = pit_funding[-1].funding_rate
    elif pit_stats and pit_stats[-1].last_funding_rate is not None:
        funding_rate_8h = pit_stats[-1].last_funding_rate

    # 3. News count in last 30m strictly PIT (available_time <= cutoff)
    news_window_start = cutoff - timedelta(minutes=30)
    pit_news = [
        art
        for art in news_articles
        if art.available_time.astimezone(UTC) <= cutoff
        and art.published_time.astimezone(UTC) >= news_window_start
        and (
            base_asset in {ent.base_asset.upper() for ent in art.affected_entities}
            or canonical_symbol.upper()
            in {ent.canonical_symbol.upper() for ent in art.affected_entities}
        )
    ]
    news_count_30m = len(pit_news)
    news_tone_label: str | None = "neutral" if news_count_30m > 0 else None

    # 4. Fear & Greed daily strictly PIT
    pit_fg = sorted(
        (obs for obs in fear_greed_observations if obs.available_time.astimezone(UTC) <= cutoff),
        key=lambda o: o.available_time.astimezone(UTC),
    )
    fg_val = pit_fg[-1].value if pit_fg else None
    fg_cls = pit_fg[-1].classification if pit_fg else None

    return ScalpFeatureSnapshot(
        decision_time_utc=cutoff,
        taker_buy_sell_ratio_5m_3bar=taker_ratio_3bar,
        cvd_ratio_15m=cvd_ratio_15m,
        oi_change_1h_pct=oi_change_1h_pct,
        price_change_1h_pct=price_change_1h_pct,
        oi_price_label=_classify_oi_price_regime(oi_change_1h_pct, price_change_1h_pct),
        top_trader_ls_ratio=top_ls,
        funding_rate_8h=funding_rate_8h,
        long_liq_5m_usd=long_liq,
        short_liq_5m_usd=short_liq,
        total_liq_5m_usd=total_liq,
        news_count_30m=news_count_30m,
        news_tone_30m_label=news_tone_label,
        fear_greed_value=fg_val,
        fear_greed_classification=fg_cls,
    )
