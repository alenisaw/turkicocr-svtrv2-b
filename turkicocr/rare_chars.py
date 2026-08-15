from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

RARE_CHARS: tuple[str, ...] = ("ә", "ғ", "қ", "ң", "ө", "ұ", "ү", "і", "һ")
COMMON_SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "ә": ("а", "e", "е"),
    "ғ": ("г",),
    "қ": ("к",),
    "ң": ("н",),
    "ө": ("о",),
    "ұ": ("у", "ү"),
    "ү": ("у", "ұ"),
    "і": ("и", "i"),
    "һ": ("х", "h"),
}


@dataclass(frozen=True)
class RareCharStats:
    precision: float
    recall: float
    f1: float
    gt_count: int
    pred_count: int
    correct_count: int


def contains_rare_char(text: str) -> bool:
    lowered = text.lower()
    return any(ch in lowered for ch in RARE_CHARS)


def rare_char_stats(references: list[str], predictions: list[str]) -> RareCharStats:
    gt = Counter()
    pred = Counter()
    correct = Counter()
    for ref, hyp in zip(references, predictions, strict=False):
        ref_l = ref.lower()
        hyp_l = hyp.lower()
        for ch in RARE_CHARS:
            gt[ch] += ref_l.count(ch)
            pred[ch] += hyp_l.count(ch)
            correct[ch] += min(ref_l.count(ch), hyp_l.count(ch))
    gt_count = sum(gt.values())
    pred_count = sum(pred.values())
    correct_count = sum(correct.values())
    precision = correct_count / pred_count if pred_count else 0.0
    recall = correct_count / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RareCharStats(precision, recall, f1, gt_count, pred_count, correct_count)


def per_char_recall(references: list[str], predictions: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for ch in RARE_CHARS:
        gt_count = 0
        hit_count = 0
        for ref, hyp in zip(references, predictions, strict=False):
            ref_count = ref.lower().count(ch)
            hyp_count = hyp.lower().count(ch)
            gt_count += ref_count
            hit_count += min(ref_count, hyp_count)
        result[ch] = hit_count / gt_count if gt_count else 0.0
    return result


def rare_confusion_counts(
    references: list[str], predictions: list[str]
) -> dict[str, dict[str, int]]:
    """Approximate rare-char confusion counts using aligned character positions.

    This lightweight method is intentionally conservative. For exact edit-operation
    alignment, replace it with a Levenshtein op-code implementation.
    """
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ref, hyp in zip(references, predictions, strict=False):
        ref_l = ref.lower()
        hyp_l = hyp.lower()
        max_len = min(len(ref_l), len(hyp_l))
        for i in range(max_len):
            r = ref_l[i]
            h = hyp_l[i]
            if r in RARE_CHARS and r != h:
                matrix[r][h] += 1
        if len(ref_l) > max_len:
            for r in ref_l[max_len:]:
                if r in RARE_CHARS:
                    matrix[r]["<missing>"] += 1
    return {k: dict(v) for k, v in matrix.items()}
