from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .metrics import cer, normalize_text
from .rare_chars import rare_confusion_counts
from .utils import ensure_dir, iter_jsonl, write_json, write_jsonl


def _simple_char_substitutions(ref: str, hyp: str) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    n = min(len(ref), len(hyp))
    for i in range(n):
        if ref[i] != hyp[i]:
            counts[(ref[i], hyp[i])] += 1
    for ch in ref[n:]:
        counts[(ch, "<missing>")] += 1
    for ch in hyp[n:]:
        counts[("<inserted>", ch)] += 1
    return counts


def classify_failure(row: dict[str, Any], row_cer: float) -> str:
    meta = row.get("metadata", {}) or {}
    degradation = str(meta.get("degradation_profile", "")).lower()
    layout = str(meta.get("layout_type", "")).lower()
    reference = str(row.get("reference", ""))
    prediction = str(row.get("prediction", ""))
    if row_cer < 0.05:
        return "CORRECT"
    if "stamp" in degradation or "official" in degradation:
        return "STAMP_INTERFERENCE"
    if "low" in degradation or "dpi" in degradation:
        return "LOW_DPI"
    if "phone" in degradation or "photo" in degradation:
        return "LOW_DPI"
    if "table" in layout or "transaction" in layout:
        return "TABLE_BOUNDARY"
    if any(token in reference.lower() for token in ["қ", "ң", "ә", "ү", "ұ"]) and row_cer > 0.15:
        return "MIXED_LANGUAGE"
    if len(prediction) < max(5, len(reference) * 0.5):
        return "FONT_ANOMALY"
    return "UNCLASSIFIED"


def run_error_analysis(predictions_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    rows = list(iter_jsonl(predictions_path))
    substitutions: Counter[tuple[str, str]] = Counter()
    failure_rows: list[dict[str, Any]] = []
    hardcases: list[dict[str, Any]] = []
    refs: list[str] = []
    hyps: list[str] = []

    for row in rows:
        ref = normalize_text(str(row.get("reference", "")))
        hyp = normalize_text(str(row.get("prediction", "")))
        refs.append(ref)
        hyps.append(hyp)
        row_cer = cer(ref, hyp)
        substitutions.update(_simple_char_substitutions(ref, hyp))
        failure = classify_failure(row, row_cer)
        failure_rows.append(
            {
                "id": row.get("id"),
                "cer": row_cer,
                "failure_mode": failure,
                "image_url": row.get("image_url"),
            }
        )
        meta = row.get("metadata", {}) or {}
        degradation = str(meta.get("degradation_profile", ""))
        layout = str(meta.get("layout_type", ""))
        if (
            row_cer > 0.25
            or (degradation in {"phone_photo", "low_dpi_scan", "old_paper"} and row_cer > 0.15)
            or (layout in {"tables_transactional", "forms"} and row_cer > 0.20)
        ):
            hardcases.append(row)

    write_top_substitutions(out / "top_confused_chars.csv", substitutions)
    write_json(out / "rare_char_confusion_matrix.json", rare_confusion_counts(refs, hyps))
    write_failure_csv(out / "page_failure_taxonomy.csv", failure_rows)
    write_failure_distribution(out / "failure_mode_distribution.csv", failure_rows)
    write_suffix_errors(out / "suffix_error_analysis.csv", rows)
    write_jsonl(out / "hardcases.jsonl", hardcases)
    summary = {
        "count": len(rows),
        "hardcase_count": len(hardcases),
        "failure_modes": dict(Counter(r["failure_mode"] for r in failure_rows)),
    }
    write_json(out / "error_analysis_summary.json", summary)
    return summary


def write_top_substitutions(path: str | Path, substitutions: Counter[tuple[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("ground_truth,predicted,count\n")
        for (gt, pred), count in substitutions.most_common(50):
            f.write(f'"{gt}","{pred}",{count}\n')


def write_failure_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("id,cer,failure_mode,image_url\n")
        for row in rows:
            f.write(
                f'"{row.get("id", "")}",{row["cer"]:.6f},"{row["failure_mode"]}","{row.get("image_url", "")}"\n'
            )


def write_failure_distribution(path: str | Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["failure_mode"]].append(float(row["cer"]))
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("failure_mode,count,mean_cer\n")
        for mode, cers in sorted(grouped.items()):
            f.write(f'"{mode}",{len(cers)},{sum(cers) / len(cers):.6f}\n')


def write_suffix_errors(path: str | Path, rows: list[dict[str, Any]]) -> None:
    counts: Counter[str] = Counter()
    for row in rows:
        ref_words = normalize_text(str(row.get("reference", ""))).split()
        hyp_words = normalize_text(str(row.get("prediction", ""))).split()
        for ref, hyp in zip(ref_words, hyp_words, strict=False):
            if ref != hyp and len(ref) >= 3:
                counts[ref[-3:]] += 1
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("suffix,count\n")
        for suffix, count in counts.most_common(100):
            f.write(f'"{suffix}",{count}\n')
