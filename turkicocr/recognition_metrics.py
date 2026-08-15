from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .rare_chars import RARE_CHARS
from .recognition_format import normalize_recognition_text, text_length_bucket

try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
except Exception:  # pragma: no cover - optional acceleration
    _RapidLevenshtein = None


def _edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
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
    ref = reference or ""
    hyp = prediction or ""
    return _edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, prediction: str) -> float:
    ref_words = normalize_recognition_text(reference).split()
    hyp_words = normalize_recognition_text(prediction).split()
    return _edit_distance(ref_words, hyp_words) / max(1, len(ref_words))


def _char_ngrams(text: str, n: int) -> Counter[str]:
    text = normalize_recognition_text(text.lower())
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


def exact_match(reference: str, prediction: str, normalize: bool = True) -> bool:
    if normalize:
        return normalize_recognition_text(reference) == normalize_recognition_text(prediction)
    return (reference or "") == (prediction or "")


def length_ratio(reference: str, prediction: str) -> float:
    ref = normalize_recognition_text(reference)
    hyp = normalize_recognition_text(prediction)
    return len(hyp) / max(1, len(ref))


def rare_char_cer(
    reference: str,
    prediction: str,
    rare_chars: Sequence[str] = RARE_CHARS,
) -> float:
    rare = set(rare_chars)
    ref = "".join(ch for ch in (reference or "").lower() if ch in rare)
    hyp = "".join(ch for ch in (prediction or "").lower() if ch in rare)
    if not ref:
        return math.nan
    return _edit_distance(ref, hyp) / max(1, len(ref))


def rare_char_confusion_matrix(
    references: list[str],
    predictions: list[str],
    rare_chars: Sequence[str] = RARE_CHARS,
) -> dict[str, dict[str, int]]:
    rare = set(rare_chars)
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for ref, hyp in zip(references, predictions, strict=False):
        ref_l = (ref or "").lower()
        hyp_l = (hyp or "").lower()
        n = min(len(ref_l), len(hyp_l))
        for i in range(n):
            if ref_l[i] in rare and ref_l[i] != hyp_l[i]:
                matrix[ref_l[i]][hyp_l[i]] += 1
        for ch in ref_l[n:]:
            if ch in rare:
                matrix[ch]["<missing>"] += 1
    return {key: dict(value) for key, value in matrix.items()}


def _mean(values: list[float]) -> float:
    real = [value for value in values if not math.isnan(value)]
    return sum(real) / len(real) if real else math.nan


def _read_charset(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    return {ch for ch in text if ch not in {"\n", "\r"}}


def evaluate_recognition_predictions(
    references: list[str],
    predictions: list[str],
    metadata: list[dict[str, Any]] | None = None,
    charset_path: str | Path | None = None,
) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    metadata = metadata or [{} for _ in references]
    charset = _read_charset(charset_path)
    rows: list[dict[str, float]] = []
    for ref, pred, meta in zip(references, predictions, metadata, strict=False):
        ref_n = normalize_recognition_text(ref)
        pred_n = normalize_recognition_text(pred)
        bucket = str(meta.get("text_length_bucket") or text_length_bucket(ref_n))
        rows.append(
            {
                "cer": cer(ref_n, pred_n),
                "wer": wer(ref_n, pred_n),
                "chrf": chrf_score(ref_n, pred_n),
                "rare_char_cer": rare_char_cer(ref_n, pred_n),
                "exact_match": float(exact_match(ref_n, pred_n)),
                "exact_match_short": float(bucket == "short" and exact_match(ref_n, pred_n)),
                "is_short": float(bucket == "short"),
                "length_ratio": length_ratio(ref_n, pred_n),
                "empty": float(pred_n == ""),
                "too_short": float(length_ratio(ref_n, pred_n) < 0.70),
                "too_long": float(length_ratio(ref_n, pred_n) > 1.30),
                "latency_ms": float(meta.get("latency_ms", math.nan)),
                "invalid_charset": float(
                    bool(charset is not None and any(ch not in charset for ch in pred_n))
                ),
            }
        )
    if not rows:
        return {"val_rec_count": 0.0}
    short_rows = [row for row in rows if row["is_short"]]
    metrics = {
        "val_rec_count": float(len(rows)),
        "val_rec_cer": _mean([row["cer"] for row in rows]),
        "val_rec_wer": _mean([row["wer"] for row in rows]),
        "val_rec_chrf": _mean([row["chrf"] for row in rows]),
        "val_rec_rare_char_cer": _mean([row["rare_char_cer"] for row in rows]),
        "val_rec_exact_match": _mean([row["exact_match"] for row in rows]),
        "val_rec_exact_match_short": _mean([row["exact_match"] for row in short_rows])
        if short_rows
        else math.nan,
        "val_rec_length_ratio": _mean([row["length_ratio"] for row in rows]),
        "val_rec_empty_prediction_rate": _mean([row["empty"] for row in rows]),
        "val_rec_too_short_rate": _mean([row["too_short"] for row in rows]),
        "val_rec_too_long_rate": _mean([row["too_long"] for row in rows]),
        "val_rec_latency_ms_per_crop": _mean([row["latency_ms"] for row in rows]),
    }
    if charset is not None:
        metrics["val_rec_invalid_charset_rate"] = _mean(
            [row["invalid_charset"] for row in rows]
        )
    return metrics


def evaluate_recognition_by_slice(
    records: list[dict[str, Any]], slice_key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        meta = row.get("metadata", {}) or {}
        value = row.get(slice_key, meta.get(slice_key, "unknown"))
        groups[str(value)].append(row)
    output: list[dict[str, Any]] = []
    for value, rows in sorted(groups.items()):
        metrics = evaluate_recognition_predictions(
            [str(row.get("reference", row.get("text", ""))) for row in rows],
            [str(row.get("prediction", "")) for row in rows],
            [dict(row.get("metadata", {}) or {}) for row in rows],
        )
        metrics[slice_key] = value
        output.append(metrics)
    return output


DEFAULT_RECOGNITION_GATE: dict[str, float] = {
    "max_cer": 0.10,
    "max_rare_char_cer": 0.15,
    "min_exact_match_short": 0.80,
    "max_empty_rate": 0.05,
    "max_too_short_rate": 0.05,
    "length_ratio_min": 0.70,
    "length_ratio_max": 1.30,
    "strong_result_cer": 0.05,
    "strong_result_rare_char_cer": 0.08,
    "strong_result_chrf": 0.90,
    "strong_result_exact_match_short": 0.90,
    "smoke_fail_cer": 0.25,
}


def evaluate_recognition_gate(
    metrics: dict[str, float], thresholds: dict[str, float] | None = None
) -> dict[str, Any]:
    cfg = {**DEFAULT_RECOGNITION_GATE, **(thresholds or {})}
    if "val_rec_cer" not in metrics:
        return {
            "status": "FAIL",
            "promote_checkpoint": False,
            "reasons": ["missing val_rec_cer"],
        }

    reasons: list[str] = []
    cer_value = float(metrics["val_rec_cer"])
    rare_value = float(metrics.get("val_rec_rare_char_cer", math.nan))
    short_em = float(metrics.get("val_rec_exact_match_short", math.nan))
    empty_rate = float(metrics.get("val_rec_empty_prediction_rate", 0.0))
    too_short = float(metrics.get("val_rec_too_short_rate", 0.0))
    ratio = float(metrics.get("val_rec_length_ratio", 1.0))

    if cer_value > cfg["smoke_fail_cer"]:
        reasons.append("smoke fail: val_rec_cer exceeds smoke_fail_cer")
        return {"status": "SMOKE_FAIL", "promote_checkpoint": False, "reasons": reasons}
    if cer_value > cfg["max_cer"]:
        reasons.append("val_rec_cer exceeds max_cer")
    if not math.isnan(rare_value) and rare_value > cfg["max_rare_char_cer"]:
        reasons.append("val_rec_rare_char_cer exceeds max_rare_char_cer")
    if not math.isnan(short_em) and short_em < cfg["min_exact_match_short"]:
        reasons.append("val_rec_exact_match_short below min_exact_match_short")
    if empty_rate > cfg["max_empty_rate"]:
        reasons.append("val_rec_empty_prediction_rate exceeds max_empty_rate")
    if too_short > cfg["max_too_short_rate"]:
        reasons.append("val_rec_too_short_rate exceeds max_too_short_rate")
    if ratio < cfg["length_ratio_min"] or ratio > cfg["length_ratio_max"]:
        reasons.append("val_rec_length_ratio outside configured range")
    if reasons:
        return {"status": "FAIL", "promote_checkpoint": False, "reasons": reasons}

    warn_reasons: list[str] = []
    if cer_value > cfg["strong_result_cer"]:
        warn_reasons.append("val_rec_cer above strong_result_cer")
    if not math.isnan(rare_value) and rare_value > cfg["strong_result_rare_char_cer"]:
        warn_reasons.append("val_rec_rare_char_cer above strong_result_rare_char_cer")
    if float(metrics.get("val_rec_chrf", 1.0)) < cfg["strong_result_chrf"]:
        warn_reasons.append("val_rec_chrf below strong_result_chrf")
    if not math.isnan(short_em) and short_em < cfg["strong_result_exact_match_short"]:
        warn_reasons.append("val_rec_exact_match_short below strong_result_exact_match_short")
    status = "WARN" if warn_reasons else "PASS"
    return {
        "status": status,
        "promote_checkpoint": status in {"PASS", "WARN"},
        "reasons": warn_reasons,
    }
