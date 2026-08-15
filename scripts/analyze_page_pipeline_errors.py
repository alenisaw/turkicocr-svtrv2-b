#!/usr/bin/env python
"""Analyze page-level OCR pipeline errors.

This script is intentionally standalone: it does not depend on the turkicocr
package, so it can be used on exported benchmark folders and in CI smoke tests.

Input JSONL rows should contain at least:
  {"page_id": ..., "gt_text": ..., "prediction": ..., "crop_count": ...}

Outputs:
  * per-page CSV
  * JSON containing aggregate summary and per-page rows
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
except Exception:  # pragma: no cover
    _RapidLevenshtein = None

_WORD_RE = re.compile(r"[\wА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё]+", re.UNICODE)


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split()).strip()


def tokenize_words(text: str | None) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(normalize_text(text))]


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if _RapidLevenshtein is not None:
        return int(_RapidLevenshtein.distance(a, b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer(reference: str, prediction: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, prediction: str) -> float:
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(prediction)
    return edit_distance(ref_words, hyp_words) / max(1, len(ref_words))


def _char_ngrams(text: str, n: int) -> Counter[str]:
    text = normalize_text(text.lower())
    if not text:
        return Counter()
    if len(text) < n:
        return Counter([text])
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf_score(reference: str, prediction: str, beta: float = 1.0, max_n: int = 6) -> float:
    scores: list[float] = []
    beta2 = beta * beta
    for n in range(1, max_n + 1):
        ref = _char_ngrams(reference, n)
        hyp = _char_ngrams(prediction, n)
        if not ref and not hyp:
            scores.append(1.0)
            continue
        overlap = sum((ref & hyp).values())
        precision = overlap / max(1, sum(hyp.values()))
        recall = overlap / max(1, sum(ref.values()))
        denom = beta2 * precision + recall
        scores.append((1 + beta2) * precision * recall / denom if denom else 0.0)
    return sum(scores) / len(scores)


def bag_of_words_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    ref = Counter(tokenize_words(reference))
    hyp = Counter(tokenize_words(prediction))
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    overlap = sum((ref & hyp).values())
    precision = overlap / max(1, sum(hyp.values()))
    recall = overlap / max(1, sum(ref.values()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def classify_error(cer_value: float, bow_f1: float, length_ratio: float, prediction: str) -> str:
    pred_n = normalize_text(prediction)
    if not pred_n:
        return "empty_prediction"
    if length_ratio < 0.70 and cer_value > 0.25:
        return "detection_miss_or_too_short"
    if length_ratio > 1.30 and cer_value > 0.25:
        return "duplicate_or_noise_text"
    if bow_f1 >= 0.75 and cer_value >= 0.30:
        return "order_issue"
    if bow_f1 < 0.50 and 0.70 <= length_ratio <= 1.30 and cer_value >= 0.30:
        return "recognizer_or_bad_crop_failure"
    if cer_value < 0.15:
        return "acceptable"
    return "mixed"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def analyze_row(row: dict[str, Any], dataset: str | None = None, sort_mode: str | None = None) -> dict[str, Any]:
    gt = str(row.get("gt_text") or row.get("reference") or row.get("text") or "")
    pred = str(row.get("prediction") or "")
    gt_n = normalize_text(gt)
    pred_n = normalize_text(pred)

    c = cer(gt_n, pred_n)
    w = wer(gt_n, pred_n)
    chrf = chrf_score(gt_n, pred_n)
    bow_p, bow_r, bow_f1 = bag_of_words_scores(gt_n, pred_n)
    length_ratio = len(pred_n) / max(1, len(gt_n))
    error_type = classify_error(c, bow_f1, length_ratio, pred_n)

    return {
        "page_id": row.get("page_id") or row.get("id") or row.get("sample_id") or "unknown",
        "dataset": dataset or row.get("dataset") or "unknown",
        "sort_mode": sort_mode or row.get("sort_mode") or "unknown",
        "gt_len": len(gt_n),
        "pred_len": len(pred_n),
        "length_ratio": length_ratio,
        "crop_count": int(row.get("crop_count") or row.get("zone_count") or 0),
        "cer": c,
        "wer": w,
        "chrf": chrf,
        "word_bag_precision": bow_p,
        "word_bag_recall": bow_r,
        "word_bag_f1": bow_f1,
        "empty_prediction": pred_n == "",
        "too_short": length_ratio < 0.70,
        "too_long": length_ratio > 1.30,
        "error_type": error_type,
    }


def mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def median_value(values: list[float]) -> float:
    finite = sorted(v for v in values if not math.isnan(v))
    if not finite:
        return math.nan
    mid = len(finite) // 2
    return finite[mid] if len(finite) % 2 else (finite[mid - 1] + finite[mid]) / 2


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["error_type"]) for row in rows)
    return {
        "page_count": len(rows),
        "mean_cer": mean([float(row["cer"]) for row in rows]),
        "median_cer": median_value([float(row["cer"]) for row in rows]),
        "mean_wer": mean([float(row["wer"]) for row in rows]),
        "mean_chrf": mean([float(row["chrf"]) for row in rows]),
        "mean_word_bag_f1": mean([float(row["word_bag_f1"]) for row in rows]),
        "mean_length_ratio": mean([float(row["length_ratio"]) for row in rows]),
        "error_type_counts": dict(sorted(counts.items())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze page-level OCR pipeline errors.")
    parser.add_argument("--page-predictions", required=True, help="Input page_predictions.jsonl")
    parser.add_argument("--out-csv", required=True, help="Output diagnostics CSV")
    parser.add_argument("--out-json", required=True, help="Output diagnostics JSON")
    parser.add_argument("--dataset", default=None, help="Optional dataset override")
    parser.add_argument("--sort-mode", default=None, help="Optional sort_mode override")
    args = parser.parse_args()

    input_path = Path(args.page_predictions)
    rows = [analyze_row(row, dataset=args.dataset, sort_mode=args.sort_mode) for row in iter_jsonl(input_path)]
    summary = summarize(rows)

    write_csv(Path(args.out_csv), rows)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "pages": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
