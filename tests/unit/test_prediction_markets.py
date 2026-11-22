"""Unit tests for Track 3 read-only prediction market context (ADR 0012)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from crypto_chatbot.analysis.service import build_analysis_snapshot
from crypto_chatbot.bot.templates import (
    render_compact_snapshot,
    render_deterministic_snapshot,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.prediction_markets.connector import (
    PolymarketConnector,
    normalize_polymarket_market,
)
from crypto_chatbot.schemas.market import QualityFlag, Timeframe
from crypto_chatbot.schemas.prediction_market import (
    PredictionMarketObservation,
    PredictionMarketSummary,
)

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"


def test_normalize_polymarket_market_categories() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    # 1. Crypto price market
    raw_btc = {
        "id": "mkt-btc-1",
        "question": "Will Bitcoin reach $120,000 in Q1 2026?",
        "outcomePrices": '["0.65", "0.35"]',
        "volume24hr": 250000.0,
        "liquidity": 100000.0,
    }
    obs_btc = normalize_polymarket_market(raw_btc, received_time=now)
    assert obs_btc is not None
    assert obs_btc.category == "crypto_price"
    assert obs_btc.canonical_symbol == "BTC/USDT"
    assert pytest.approx(obs_btc.implied_yes_prob, abs=1e-5) == 0.65
    assert not obs_btc.quality_flags

    # 2. Macro rate cut market
    raw_fed = {
        "id": "mkt-fed-1",
        "question": "Will the Fed cut interest rates at the next FOMC meeting?",
        "outcomePrices": ["0.82", "0.18"],
        "volume24hr": 1500000.0,
        "liquidity": 400000.0,
    }
    obs_fed = normalize_polymarket_market(raw_fed, received_time=now)
    assert obs_fed is not None
    assert obs_fed.category == "macro_rate"
    assert pytest.approx(obs_fed.implied_yes_prob, abs=1e-5) == 0.82

    # 3. Low volume flag check
    raw_thin = {
        "id": "mkt-thin-1",
        "question": "Will Solana hit $500 before March?",
        "outcomePrices": ["0.15", "0.85"],
        "volume24hr": 150.0,  # below $1000
        "liquidity": 200.0,
    }
    obs_thin = normalize_polymarket_market(raw_thin, received_time=now)
    assert obs_thin is not None
    assert QualityFlag.ZERO_VOLUME in obs_thin.quality_flags


@pytest.mark.asyncio
async def test_polymarket_connector_aggregation_and_lookahead() -> None:
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    mock_client = AsyncMock()
    mock_client.get_events.return_value = [
        {
            "id": "event-1",
            "title": "Crypto Markets",
            "markets": [
                {
                    "id": "m-btc",
                    "question": "Will Bitcoin reach $100k?",
                    "outcomePrices": ["0.70", "0.30"],
                    "volume24hr": 50000.0,
                    "liquidity": 20000.0,
                },
                {
                    "id": "m-fomc",
                    "question": "Will the Fed cut rates?",
                    "outcomePrices": ["0.45", "0.55"],
                    "volume24hr": 80000.0,
                    "liquidity": 30000.0,
                },
            ],
        }
    ]

    connector = PolymarketConnector(client=mock_client, clock=lambda: ref_time)
    summary = await connector.get_summary(as_of=ref_time, target_symbol="BTC/USDT")
    assert summary.has_liquid_markets is True
    assert summary.crypto_upside_prob == 0.70
    assert summary.macro_fomc_cut_prob == 0.45
    assert len(summary.observations) == 2
    for o in summary.observations:
        assert o.available_time <= ref_time
    await connector.close()


@pytest.mark.asyncio
async def test_polymarket_connector_graceful_timeout() -> None:
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    mock_client = AsyncMock()
    mock_client.get_events.side_effect = TimeoutError("Simulated network timeout")

    connector = PolymarketConnector(client=mock_client, clock=lambda: ref_time)
    summary = await connector.get_summary(as_of=ref_time)
    assert summary.has_liquid_markets is False
    assert len(summary.observations) == 0
    await connector.close()


def test_template_rendering_prediction_markets() -> None:
    data = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    raw = data["raw_ohlcv"]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    received = last_open + Timeframe.H4.duration + timedelta(seconds=1)
    batch = normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )
    base_snap = build_analysis_snapshot(batch)

    obs = (
        PredictionMarketObservation(
            venue="polymarket",
            market_id="mkt-1",
            question="Will Bitcoin hit $110,000 in Q1 2026?",
            category="crypto_price",
            canonical_symbol="BTC/USDT",
            implied_yes_prob=0.72,
            volume_24h_usd=120000.0,
            liquidity_usd=50000.0,
            observed_time=received,
            received_time=received,
            available_time=received,
            url="https://polymarket.com/event/btc-110k",
        ),
    )
    pm = PredictionMarketSummary(
        as_of=received,
        observations=obs,
        crypto_upside_prob=0.72,
        macro_fomc_cut_prob=0.60,
    )

    snapshot = base_snap.model_copy(update={"prediction_markets": pm})

    # Deterministic output
    det_rendered = render_deterministic_snapshot(snapshot)
    assert "Konteks Pasar Prediksi (Harga Kontrak Tersirat" in det_rendered
    assert "POLYMARKET" in det_rendered
    assert "Will Bitcoin hit $110,000" in det_rendered
    assert "72,0%" in det_rendered
    assert "Peringatan: Harga kontrak mencerminkan dinamika taruhan" in det_rendered

    # Compact output
    comp_rendered = render_compact_snapshot(snapshot)
    assert "Pasar Prediksi (Tersirat):" in comp_rendered
    assert "FOMC Cut: 60%" in comp_rendered
    assert "Target YES: 72%" in comp_rendered
