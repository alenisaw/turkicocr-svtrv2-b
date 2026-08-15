from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .recognition_metrics import evaluate_recognition_predictions

BENCHMARK_REGISTRY: dict[str, dict[str, str]] = {
    "kazakhocr_cyrillic": {
        "manifest": "manifests/kazakhocr_cyrillic.jsonl",
        "role": "Kazakh Cyrillic OCR transfer benchmark",
    },
    "kohtd": {
        "manifest": "manifests/kohtd.jsonl",
        "role": "Kazakh handwritten transfer check",
    },
}


def load_benchmark_manifest(benchmark_name: str, asset_root: str | Path) -> list[dict[str, Any]]:
    if benchmark_name not in BENCHMARK_REGISTRY:
        known = ", ".join(sorted(BENCHMARK_REGISTRY))
        raise KeyError(f"Unknown benchmark {benchmark_name!r}. Known benchmarks: {known}")
    path = Path(asset_root) / BENCHMARK_REGISTRY[benchmark_name]["manifest"]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_benchmark(
    benchmark_name: str,
    model_fn: Callable[[dict[str, Any]], str],
    asset_root: str | Path,
) -> dict[str, Any]:
    rows = load_benchmark_manifest(benchmark_name, asset_root)
    references: list[str] = []
    predictions: list[str] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        references.append(str(row.get("reference", row.get("text", ""))))
        predictions.append(model_fn(row))
        metadata.append(dict(row.get("metadata", {}) or {}))
    metrics = evaluate_recognition_predictions(references, predictions, metadata)
    return {"benchmark": benchmark_name, "count": len(rows), "metrics": metrics}


def compare_benchmarks(
    results: dict[str, Any],
    baseline_results: dict[str, Any],
) -> dict[str, Any]:
    current = results.get("metrics", results)
    baseline = baseline_results.get("metrics", baseline_results)
    comparison: dict[str, Any] = {
        "benchmark": results.get("benchmark", baseline_results.get("benchmark", "unknown")),
    }
    for key in ("val_rec_cer", "val_rec_wer", "val_rec_chrf", "val_rec_exact_match"):
        if key in current and key in baseline:
            comparison[f"{key}_delta"] = float(current[key]) - float(baseline[key])
    if "val_rec_cer_delta" in comparison:
        comparison["cer_relative_reduction"] = (
            -comparison["val_rec_cer_delta"] / max(1e-12, float(baseline["val_rec_cer"]))
        )
    return comparison
