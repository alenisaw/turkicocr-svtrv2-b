#!/usr/bin/env python
"""Synchronize canonical line-recognition metrics into a clean release structure."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from turkicocr.utils import get_asset_root

CANONICAL_EPOCH9 = {
    "run_id": "turkicocr_svtrv2_b_epoch9_line_full",
    "model_id": "turkicocr_svtrv2_b_epoch9",
    "checkpoint": "epoch_9.pth",
    "eval_level": "line_recognition",
    "dataset": "alenisaw/turkicocr-cyrillic",
    "split": "heldout_test_full",
    "sample_count": 293814,
    "page_count": None,
    "crop_count": 293814,
    "script": "scripts/sync_canonical_metrics.py",
    "config": "configs/train_svtrv2_b_rec_line.yaml",
    "git_commit": None,
    "cer": 0.04508417933782638,
    "wer": 0.09141855995319104,
    "chrf": 0.9376634609868352,
    "exact_match": 0.8600713376489888,
    "exact_match_short": 0.9107200758237098,
    "rare_char_cer": 0.05994860007020739,
    "length_ratio": None,
    "empty_prediction_rate": 0.00005,
    "too_short_rate": None,
    "too_long_rate": None,
    "invalid_charset_rate": 0.0,
    "latency_ms_per_sample": 14.69823195307572,
    "notes": "Canonical full held-out line-crop evaluation for TurkicOCR-SVTRv2-B epoch 9.",
}

KEY_MAP = {
    "val_rec_count": "sample_count",
    "val_rec_cer": "cer",
    "val_rec_wer": "wer",
    "val_rec_chrf": "chrf",
    "val_rec_exact_match": "exact_match",
    "val_rec_exact_match_short": "exact_match_short",
    "val_rec_rare_char_cer": "rare_char_cer",
    "val_rec_length_ratio": "length_ratio",
    "val_rec_empty_prediction_rate": "empty_prediction_rate",
    "val_rec_too_short_rate": "too_short_rate",
    "val_rec_too_long_rate": "too_long_rate",
    "val_rec_invalid_charset_rate": "invalid_charset_rate",
    "val_rec_latency_ms_per_crop": "latency_ms_per_sample",
}


def load_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized = dict(CANONICAL_EPOCH9)
    for key, value in data.items():
        normalized[KEY_MAP.get(key, key)] = value
    return normalized


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_final_metrics_md(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# Final Metrics

## Canonical Line Recognition

| Metric | Value |
|---|---:|
| Model | `{metrics['model_id']}` |
| Dataset/Split | `{metrics['dataset']} / {metrics['split']}` |
| Samples | {int(metrics['sample_count']):,} line crops |
| CER | {float(metrics['cer']) * 100:.2f}% |
| WER | {float(metrics['wer']) * 100:.2f}% |
| chrF | {float(metrics['chrf']):.4f} |
| Exact Match | {float(metrics['exact_match']) * 100:.2f}% |
| Short Exact Match | {float(metrics['exact_match_short']) * 100:.2f}% |
| Rare-char CER | {float(metrics['rare_char_cer']) * 100:.2f}% |
| Invalid charset rate | {float(metrics.get('invalid_charset_rate') or 0.0) * 100:.4f}% |
| Latency | {float(metrics.get('latency_ms_per_sample') or 0.0):.3f} ms/crop |

## Status

The canonical recognizer metric is line/region-level OCR. Page-level metrics must be reported separately as `page_oracle` or `page_detected`.
"""
    path.write_text(md, encoding="utf-8")


BREAKDOWN_GROUP_KEYS = {
    "by_language": "language",
    "by_layout": "layout_type",
    "by_degradation": "degradation_profile",
    "by_crop_type": "crop_type",
}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_breakdown_csv(path: Path, group_key: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for row in rows:
        entry = {"group": row.get(group_key, "unknown")}
        for key, value in row.items():
            if key == group_key:
                continue
            entry[KEY_MAP.get(key, key)] = value
        normalized.append(entry)
    if not normalized:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(normalized[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def write_predictions_sample(
    source_jsonl: Path, out_path: Path, sample_size: int, seed: int
) -> int:
    lines = source_jsonl.read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(lines)), min(sample_size, len(lines))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for idx in indices:
            fh.write(lines[idx] + "\n")
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write canonical TurkicOCR metrics into clean structure.")
    parser.add_argument("--source-json", default=None, help="Optional source aggregate metrics JSON with val_rec_* keys.")
    parser.add_argument(
        "--out-dir",
        default=str(get_asset_root() / "outputs/turkicocr_benchmark_clean/metrics/recognition_line_full"),
        help="Metrics live on the asset disk, not in the git repo.",
    )
    parser.add_argument("--reports-dir", default=".agent/reports")
    parser.add_argument("--use-known-epoch9", action="store_true", help="Use known epoch-9 canonical values if no source JSON is provided.")
    parser.add_argument("--by-language-json", default=None, help="JSON list of per-language val_rec_* breakdown rows.")
    parser.add_argument("--by-layout-json", default=None, help="JSON list of per-layout_type val_rec_* breakdown rows.")
    parser.add_argument("--by-degradation-json", default=None, help="JSON list of per-degradation_profile val_rec_* breakdown rows.")
    parser.add_argument("--by-crop-type-json", default=None, help="JSON list of per-crop_type val_rec_* breakdown rows.")
    parser.add_argument("--predictions-jsonl", default=None, help="Source full predictions JSONL to sample from.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    if args.source_json:
        metrics = load_metrics(Path(args.source_json))
    elif args.use_known_epoch9:
        metrics = dict(CANONICAL_EPOCH9)
    else:
        raise SystemExit("Provide --source-json or --use-known-epoch9")

    metrics["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["git_commit"] = _git_commit()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_aggregate.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_single_row_csv(out_dir / "metrics_aggregate.csv", metrics)
    write_final_metrics_md(Path(args.reports_dir) / "FINAL_METRICS.md", metrics)

    breakdown_args = {
        "by_language": args.by_language_json,
        "by_layout": args.by_layout_json,
        "by_degradation": args.by_degradation_json,
        "by_crop_type": args.by_crop_type_json,
    }
    for name, source_path in breakdown_args.items():
        if not source_path:
            continue
        rows = json.loads(Path(source_path).read_text(encoding="utf-8"))
        write_breakdown_csv(out_dir / f"{name}.csv", BREAKDOWN_GROUP_KEYS[name], rows)

    if args.predictions_jsonl:
        n = write_predictions_sample(
            Path(args.predictions_jsonl),
            out_dir / "predictions_sample.jsonl",
            args.sample_size,
            args.sample_seed,
        )
        print(f"Wrote {n} sampled predictions to {out_dir / 'predictions_sample.jsonl'}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
