from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot import cli
from crypto_chatbot.news.eval_dataset import (
    EVAL_CSV_COLUMNS,
    evaluate_labeled_headlines_csv,
    export_unlabeled_headlines_csv,
)
from crypto_chatbot.news.pipeline import deduplicate_articles
from crypto_chatbot.schemas.news import RawNewsItem

_UNLABELED_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "sentiment_eval_unlabeled.csv"
)


def test_export_leaves_manual_label_empty_never_auto_labels(tmp_path: Path) -> None:
    t0 = datetime(2026, 11, 27, 10, 0, tzinfo=UTC)
    sample_titles = [
        "Bitcoin surges past resistance after ETF inflows accelerate",
        "Solana protocol suffers smart contract exploit outage",
        "Ethereum developers publish testnet upgrade schedule",
        "Arbitrum governance approves treasury grant allocation",
        "Chainlink integrates cross-chain oracle feed with bank",
    ]
    raw = [
        RawNewsItem(
            publisher="CoinDesk",
            url=f"https://coindesk.com/story-{i}",
            title=sample_titles[i],
            published_time=t0 + timedelta(minutes=i),
            first_seen_time=t0 + timedelta(minutes=i),
        )
        for i in range(5)
    ]
    articles, _ = deduplicate_articles(raw)
    out = tmp_path / "unlabeled.csv"
    exported = export_unlabeled_headlines_csv(articles, output_path=out, count=300)
    assert exported == 5

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == EVAL_CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 5
    # Critical S-05 rule: manual_label must ALWAYS remain blank!
    assert all(row["manual_label"] == "" for row in rows)


def test_evaluate_labeled_headlines_confusion_matrix_and_accuracy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labeled_csv = tmp_path / "labeled.csv"
    with labeled_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVAL_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "news-1",
                    "published_time": "2026-11-27T12:00:00+00:00",
                    "publisher": "CoinDesk",
                    "title": "Bitcoin surges past resistance on rally",
                    "canonical_url": "https://example.com/1",
                    "detected_symbols": "BTC",
                    "manual_label": "bullish",
                },
                {
                    "id": "news-2",
                    "published_time": "2026-11-27T12:01:00+00:00",
                    "publisher": "CoinDesk",
                    "title": "Protocol suffers hack and exploit crash",
                    "canonical_url": "https://example.com/2",
                    "detected_symbols": "ETH",
                    "manual_label": "bearish",
                },
                {
                    "id": "news-3",
                    "published_time": "2026-11-27T12:02:00+00:00",
                    "publisher": "CoinDesk",
                    "title": "Conference agenda released for next week",
                    "canonical_url": "https://example.com/3",
                    "detected_symbols": "",
                    "manual_label": "neutral",
                },
                {
                    "id": "news-4",
                    "published_time": "2026-11-27T12:03:00+00:00",
                    "publisher": "CoinDesk",
                    "title": "Another headline still unlabeled",
                    "canonical_url": "https://example.com/4",
                    "detected_symbols": "",
                    "manual_label": "",
                },
            ]
        )

    report = evaluate_labeled_headlines_csv(labeled_csv)
    assert report.total_rows == 4
    assert report.labeled_rows == 3
    assert report.unlabeled_rows == 1
    assert report.accuracy == 1.0
    assert report.confusion_matrix["bullish"]["bullish"] == 1
    assert report.confusion_matrix["bearish"]["bearish"] == 1
    assert report.confusion_matrix["neutral"]["neutral"] == 1

    rc = cli.main(["evaluate-sentiment-csv", "--input", str(labeled_csv), "--json"])
    assert rc == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["accuracy"] == 1.0


def test_committed_unlabeled_fixture_has_300_recent_headlines_and_no_auto_labels() -> None:
    assert _UNLABELED_FIXTURE.is_file()
    with _UNLABELED_FIXTURE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == EVAL_CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 300
    assert len({row["id"] for row in rows}) == 300
    assert all(row["manual_label"] == "" for row in rows)
    assert all(row["title"].strip() != "" for row in rows)
