"""Unlabeled headline exporter & manual-label lexicon evaluation for S-05."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.news.pipeline import classify_article_sentiment
from crypto_chatbot.schemas.news import NormalizedNewsArticle, SentimentLabel

EVAL_LABELS: tuple[str, ...] = ("bullish", "bearish", "neutral")

_MODEL_TO_EVAL_LABEL: dict[SentimentLabel, str] = {
    SentimentLabel.POSITIVE: "bullish",
    SentimentLabel.NEGATIVE: "bearish",
    SentimentLabel.NEUTRAL: "neutral",
}

_LABEL_ALIASES: dict[str, str] = {
    "bullish": "bullish",
    "positive": "bullish",
    "pos": "bullish",
    "bearish": "bearish",
    "negative": "bearish",
    "neg": "bearish",
    "neutral": "neutral",
    "neu": "neutral",
}

EVAL_CSV_COLUMNS: tuple[str, ...] = (
    "id",
    "published_time",
    "publisher",
    "title",
    "canonical_url",
    "detected_symbols",
    "manual_label",
)


class HeadlineEvalReport(BaseModel):
    """Lexicon scorer evaluation against manually labeled headlines (S-05)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_rows: int = Field(ge=0)
    labeled_rows: int = Field(ge=0)
    unlabeled_rows: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    per_class_recall: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)


def export_unlabeled_headlines_csv(
    articles: Sequence[NormalizedNewsArticle],
    *,
    output_path: Path,
    count: int = 300,
) -> int:
    """Export recent deduplicated headlines with strictly empty `manual_label`."""
    if count < 1:
        raise ValueError("count must be >= 1")
    ordered = sorted(
        articles,
        key=lambda art: (art.published_time, art.canonical_url),
        reverse=True,
    )[:count]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVAL_CSV_COLUMNS))
        writer.writeheader()
        for art in ordered:
            symbols = ",".join(sorted({ent.base_asset for ent in art.affected_entities}))
            writer.writerow(
                {
                    "id": art.article_id,
                    "published_time": art.published_time.isoformat(),
                    "publisher": art.publisher,
                    "title": art.title,
                    "canonical_url": art.canonical_url,
                    "detected_symbols": symbols,
                    # Do NOT auto-label: owner fills `bullish` / `bearish` / `neutral` manually.
                    "manual_label": "",
                }
            )
    return len(ordered)


def evaluate_labeled_headlines_csv(input_path: Path) -> HeadlineEvalReport:
    """Compare deterministic crypto lexicon predictions against manually labeled CSV rows."""
    if not input_path.is_file():
        raise FileNotFoundError(f"headline evaluation CSV not found: {input_path}")

    matrix: dict[str, dict[str, int]] = {
        actual: {pred: 0 for pred in EVAL_LABELS} for actual in EVAL_LABELS
    }
    total_rows = 0
    labeled_rows = 0
    unlabeled_rows = 0
    correct = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or "title" not in reader.fieldnames
            or "manual_label" not in reader.fieldnames
        ):
            raise ValueError("CSV must include 'title' and 'manual_label' columns")
        for row in reader:
            total_rows += 1
            raw_label = (row.get("manual_label") or "").strip().lower()
            if not raw_label:
                unlabeled_rows += 1
                continue
            actual = _LABEL_ALIASES.get(raw_label)
            if actual is None:
                raise ValueError(
                    f"invalid manual_label '{raw_label}' (expected bullish, bearish, or neutral)"
                )
            title = (row.get("title") or "").strip()
            pred_sentiment, _, _ = classify_article_sentiment(title, "")
            predicted = _MODEL_TO_EVAL_LABEL[pred_sentiment]
            matrix[actual][predicted] += 1
            labeled_rows += 1
            if actual == predicted:
                correct += 1

    accuracy = round(correct / labeled_rows, 6) if labeled_rows > 0 else 0.0
    per_class_recall: dict[str, float] = {}
    for cls in EVAL_LABELS:
        cls_total = sum(matrix[cls].values())
        per_class_recall[cls] = round(matrix[cls][cls] / cls_total, 6) if cls_total > 0 else 0.0

    return HeadlineEvalReport(
        total_rows=total_rows,
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        accuracy=accuracy,
        per_class_recall=per_class_recall,
        confusion_matrix=matrix,
    )
