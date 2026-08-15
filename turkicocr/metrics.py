from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from .rare_chars import contains_rare_char, rare_char_stats


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split()).strip()


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def cer(reference: str, prediction: str) -> float:
    reference = reference or ""
    prediction = prediction or ""
    denom = max(1, len(reference))
    return edit_distance(reference, prediction) / denom


def wer(reference: str, prediction: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(prediction).split()
    return edit_distance("\u0001".join(ref_words), "\u0001".join(hyp_words)) / max(
        1, len(ref_words)
    )


def normalized_edit_similarity(reference: str, prediction: str) -> float:
    denom = max(1, max(len(reference or ""), len(prediction or "")))
    return 1.0 - edit_distance(reference or "", prediction or "") / denom


def _char_ngrams(text: str, n: int) -> Counter[str]:
    text = normalize_text(text.lower())
    if len(text) < n:
        return Counter([text]) if text else Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf(reference: str, prediction: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Small chrF-like implementation.

    It is adequate for local tracking. For paper-final reporting, compare this
    against sacrebleu's chrF/chrF++ implementation and freeze the exact method.
    """
    scores = []
    for n in range(1, max_n + 1):
        ref = _char_ngrams(reference, n)
        hyp = _char_ngrams(prediction, n)
        if not ref and not hyp:
            scores.append(1.0)
            continue
        overlap = sum((ref & hyp).values())
        precision = overlap / max(1, sum(hyp.values()))
        recall = overlap / max(1, sum(ref.values()))
        beta2 = beta * beta
        denom = beta2 * precision + recall
        scores.append((1 + beta2) * precision * recall / denom if denom else 0.0)
    return sum(scores) / len(scores)


def text_length_bucket(text: str) -> str:
    n = len(normalize_text(text))
    if n < 50:
        return "short"
    if n <= 200:
        return "medium"
    return "long"


def length_ratio(reference: str, prediction: str) -> float:
    return len(normalize_text(prediction)) / max(1, len(normalize_text(reference)))


def empty_rate(predictions: list[str]) -> float:
    return sum(1 for p in predictions if not normalize_text(p)) / max(1, len(predictions))


def too_short_rate(
    references: list[str], predictions: list[str], min_ratio: float = 0.70
) -> float:
    return sum(
        1 for r, p in zip(references, predictions, strict=False) if length_ratio(r, p) < min_ratio
    ) / max(1, len(references))


def too_long_rate(
    references: list[str], predictions: list[str], max_ratio: float = 1.30
) -> float:
    return sum(
        1 for r, p in zip(references, predictions, strict=False) if length_ratio(r, p) > max_ratio
    ) / max(1, len(references))


def evaluate_pairs(references: list[str], predictions: list[str]) -> dict[str, Any]:
    rows = []
    for ref, pred in zip(references, predictions, strict=False):
        ref_n = normalize_text(ref)
        pred_n = normalize_text(pred)
        rows.append(
            {
                "cer": cer(ref_n, pred_n),
                "wer": wer(ref_n, pred_n),
                "ned_similarity": normalized_edit_similarity(ref_n, pred_n),
                "chrf": chrf(ref_n, pred_n),
                "exact_match": float(ref_n == pred_n),
                "length_ratio": length_ratio(ref_n, pred_n),
            }
        )
    if not rows:
        return {"count": 0}
    summary = {"count": len(rows)}
    for key in rows[0]:
        summary[key] = sum(row[key] for row in rows) / len(rows)
    rare = rare_char_stats(references, predictions)
    summary.update({f"rare_char_{k}": v for k, v in asdict(rare).items()})
    rare_refs = [r for r in references if contains_rare_char(r)]
    rare_preds = [p for r, p in zip(references, predictions, strict=False) if contains_rare_char(r)]
    summary["rare_char_cer"] = (
        evaluate_pairs_no_rare(rare_refs, rare_preds).get("cer", math.nan)
        if rare_refs
        else math.nan
    )
    summary["empty_prediction_rate"] = empty_rate(predictions)
    summary["too_short_rate"] = too_short_rate(references, predictions)
    summary["too_long_rate"] = too_long_rate(references, predictions)
    return summary


def evaluate_pairs_no_rare(references: list[str], predictions: list[str]) -> dict[str, Any]:
    if not references:
        return {"count": 0}
    vals = [
        {
            "cer": cer(normalize_text(r), normalize_text(p)),
            "wer": wer(normalize_text(r), normalize_text(p)),
            "ned_similarity": normalized_edit_similarity(normalize_text(r), normalize_text(p)),
            "chrf": chrf(normalize_text(r), normalize_text(p)),
        }
        for r, p in zip(references, predictions, strict=False)
    ]
    out: dict[str, Any] = {"count": len(vals)}
    for key in vals[0]:
        out[key] = sum(v[key] for v in vals) / len(vals)
    return out


def grouped_metrics(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(group_key, "unknown"))].append(row)
    output = []
    for group, rows in sorted(groups.items()):
        refs = [str(r.get("reference", "")) for r in rows]
        hyps = [str(r.get("prediction", "")) for r in rows]
        metrics = evaluate_pairs(refs, hyps)
        metrics[group_key] = group
        output.append(metrics)
    return output
