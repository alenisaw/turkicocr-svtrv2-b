#!/usr/bin/env python
"""
run_pipeline_recognition_eval.py
=================================
Evaluates OCR models on page-level external benchmarks using a two-stage pipeline:

  Stage 1 (pre-run): Text-line detection + cropping
    → handled by prepare_pipeline_crops.py
    → produces: line_manifest.jsonl + page_groups.jsonl

  Stage 2 (this script): Per-model evaluation
    For LINE RECOGNIZERS (SVTRv2-B, TrOCR, Tesseract, PP-OCR):
      • Run inference on line crops (calls run_baselines_parallel_sharded.py)
      • Aggregate line predictions per page in reading order
      • Compute page-level CER vs page GT

    For VLMs (OLMOCR, MineRU):
      • Run inference on the original FULL-PAGE manifest (unchanged)
      • Compute page-level CER directly

  Final output: leaderboard CSV + JSON in --out directory.

Usage
-----
PYTHONPATH=. python3 \
    scripts/run_pipeline_recognition_eval.py \
    --all-datasets \
    --config configs/eval_baselines.yaml \
    --pipeline-crops-dir ./assets/outputs/pipeline_crops/henrygagnier__kazakh-ocr \
    --page-manifest ./assets/outputs/external_benchmarks/manifests/henrygagnier__kazakh-ocr.jsonl \
    --dataset henrygagnier__kazakh-ocr \
    --out ./assets/outputs/pipeline_eval/henrygagnier__kazakh-ocr

PYTHONPATH=. python3 \
    scripts/run_pipeline_recognition_eval.py \
    --all-datasets \
    --config configs/eval_baselines.yaml \
    --out ./assets/outputs/pipeline_eval \
    --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from turkicocr.recognition_metrics import evaluate_recognition_predictions
from turkicocr.utils import get_asset_root, write_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ASSET_ROOT = get_asset_root()
_PIPELINE_CROPS_ROOT = str(_ASSET_ROOT / "outputs/pipeline_crops")
_EXTERNAL_MANIFEST_ROOT = str(_ASSET_ROOT / "outputs/external_benchmarks/manifests")
_PIPELINE_EVAL_ROOT = str(_ASSET_ROOT / "outputs/pipeline_eval")

_ALL_DATASETS = [
    "henrygagnier__kazakh-ocr",
    "alenisaw__turkicocr-cyrillic",
]

# Full-page OCR/parser models should receive page images; recognizers receive line crops.
_FULL_PAGE_MODEL_TYPES = {"vlm_document_ocr", "vlm_document_parser"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_baseline_config(config_path: str) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return cfg.get("models", [])


def _is_vlm(model: dict[str, Any]) -> bool:
    return model.get("type") in _FULL_PAGE_MODEL_TYPES


def _aggregate_line_predictions(
    page_groups: list[dict[str, Any]],
    predictions: dict[str, str],
    separator: str = "\n",
) -> dict[str, str]:
    """Concatenate per-line predictions for each page in reading order."""
    page_texts: dict[str, str] = {}
    for page in page_groups:
        page_id = page["page_id"]
        crop_ids = page.get("crop_ids", [])
        line_texts = [
            predictions.get(crop_id, "").strip()
            for crop_id in crop_ids
        ]
        # Filter empty lines before joining
        line_texts = [t for t in line_texts if t]
        page_texts[page_id] = separator.join(line_texts)
    return page_texts


def _load_line_predictions(pred_file: Path) -> dict[str, str]:
    """Read predictions JSONL; returns {sample_id: prediction_text}."""
    preds: dict[str, str] = {}
    if not pred_file.exists():
        return preds
    with pred_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id") or row.get("id", "")
            text = row.get("prediction") or row.get("text") or ""
            preds[sample_id] = str(text).strip()
    return preds


def _load_page_predictions(pred_file: Path) -> dict[str, str]:
    """Load VLM page-level predictions. Returns {page_id/sample_id: text}."""
    preds: dict[str, str] = {}
    if not pred_file.exists():
        return preds
    with pred_file.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or f"row_{idx:06d}")
            text = row.get("prediction") or row.get("text") or ""
            preds[sample_id] = str(text).strip()
    return preds


def _build_page_id_mapping(page_manifest_path: Path) -> dict[int, str]:
    """Return {row_index: page_id} mapping for the original external page manifest."""
    mapping: dict[int, str] = {}
    with page_manifest_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            page_id = f"{meta.get('benchmark', 'unknown').replace('/', '__')}_{meta.get('source_index', idx):06d}"
            mapping[idx] = page_id
    return mapping


def _compute_metrics(
    page_groups: list[dict[str, Any]],
    page_predictions: dict[str, str],
) -> dict[str, Any]:
    refs: list[str] = []
    hyps: list[str] = []
    for page in page_groups:
        page_id = page["page_id"]
        refs.append(page.get("gt_text", ""))
        hyps.append(page_predictions.get(page_id, ""))
    return evaluate_recognition_predictions(refs, hyps)


def _run_baselines_for_manifest(
    manifest_path: Path,
    out_dir: Path,
    config: str,
    gpus: str,
    model_ids: list[str] | None = None,
    force: bool = False,
) -> None:
    """Invoke run_baselines_parallel_sharded.py to produce prediction files."""
    cmd = [
        sys.executable,
        "scripts/run_baselines_parallel_sharded.py",
        "--config", config,
        "--out", str(out_dir),
        "--manifest", str(manifest_path),
        "--gpus", gpus,
    ]

    if force:
        cmd.append("--force")
    for model_id in model_ids or []:
        cmd.extend(["--model-id", model_id])

    print(f"    Running baselines: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"    WARNING: baselines run exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _eval_dataset(
    dataset_name: str,
    pipeline_crops_dir: Path,
    page_manifest_path: Path,
    config: str,
    out_dir: Path,
    gpus: str,
    force: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    line_manifest = pipeline_crops_dir / "line_manifest.jsonl"
    page_groups_path = pipeline_crops_dir / "page_groups.jsonl"

    if not line_manifest.exists():
        return {
            "dataset": dataset_name,
            "status": "error",
            "error": f"line_manifest.jsonl not found in {pipeline_crops_dir}. "
                     "Run prepare_pipeline_crops.py first.",
        }

    page_groups = _read_jsonl(page_groups_path)
    models = _load_baseline_config(config)

    results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 1. LINE RECOGNIZERS  →  run on line crops, then aggregate per page  #
    # ------------------------------------------------------------------ #
    line_models = [m for m in models if not _is_vlm(m)]
    if line_models:
        line_pred_dir = out_dir / "predictions" / "line_models"
        line_model_ids = [m["id"] for m in line_models]

        print(f"  [{dataset_name}] Running {len(line_models)} line recognizers on crops ...")
        _run_baselines_for_manifest(
            manifest_path=line_manifest,
            out_dir=line_pred_dir,
            config=config,
            gpus=gpus,
            model_ids=line_model_ids,
            force=force,
        )

        for model in line_models:
            model_id = model["id"]
            pred_file = line_pred_dir / f"{model_id}_predictions.jsonl"
            line_preds = _load_line_predictions(pred_file)

            # Aggregate line predictions → page text
            page_preds = _aggregate_line_predictions(page_groups, line_preds)

            # Compute page-level CER
            metrics = _compute_metrics(page_groups, page_preds)

            # Save page-level predictions
            page_pred_path = out_dir / "page_predictions" / f"{model_id}_page_predictions.jsonl"
            page_pred_path.parent.mkdir(parents=True, exist_ok=True)
            with page_pred_path.open("w", encoding="utf-8") as fh:
                for page in page_groups:
                    pid = page["page_id"]
                    fh.write(json.dumps({
                        "page_id": pid,
                        "gt_text": page.get("gt_text", ""),
                        "prediction": page_preds.get(pid, ""),
                        "crop_count": len(page.get("crop_ids", [])),
                    }, ensure_ascii=False) + "\n")

            # Save metrics
            metrics_path = out_dir / "metrics" / f"{model_id}_page_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(metrics_path, metrics)

            row = {
                "dataset": dataset_name,
                "model_id": model_id,
                "display_name": model.get("display_name", model_id),
                "eval_mode": "pipeline_line",
                "cer": metrics.get("val_rec_cer"),
                "wer": metrics.get("val_rec_wer"),
                "chrf": metrics.get("val_rec_chrf"),
                "exact_match": metrics.get("val_rec_exact_match"),
                "page_count": len(page_groups),
            }
            results.append(row)
            cer_pct = f"{metrics.get('val_rec_cer', float('nan')) * 100:.2f}%"
            print(f"    {model.get('display_name', model_id):40s}  page-CER={cer_pct}")

    # ------------------------------------------------------------------ #
    # 2. VLMs  →  run on FULL-PAGE manifest, compute page CER directly   #
    # ------------------------------------------------------------------ #
    vlm_models = [m for m in models if _is_vlm(m)]
    if vlm_models and page_manifest_path.exists():
        vlm_pred_dir = out_dir / "predictions" / "vlm_models"
        vlm_model_ids = [m["id"] for m in vlm_models]

        print(f"  [{dataset_name}] Running {len(vlm_models)} VLMs on full pages ...")
        _run_baselines_for_manifest(
            manifest_path=page_manifest_path,
            out_dir=vlm_pred_dir,
            config=config,
            gpus=gpus,
            model_ids=vlm_model_ids,
            force=force,
        )

        # Build a page_id→gt_text lookup from page_groups
        page_gt: dict[str, str] = {p["page_id"]: p.get("gt_text", "") for p in page_groups}

        for model in vlm_models:
            model_id = model["id"]
            pred_file = vlm_pred_dir / f"{model_id}_predictions.jsonl"
            raw_preds = _load_page_predictions(pred_file)

            # VLM predictions are indexed by row order in the page manifest;
            # map them to page_ids using page_groups order
            page_id_order = [p["page_id"] for p in page_groups]
            page_preds: dict[str, str] = {}
            for idx, pid in enumerate(page_id_order):
                row_key = f"row_{idx:06d}" if f"row_{idx:06d}" in raw_preds else pid
                page_preds[pid] = raw_preds.get(row_key, raw_preds.get(pid, ""))

            refs = [page_gt.get(pid, "") for pid in page_id_order]
            hyps = [page_preds.get(pid, "") for pid in page_id_order]
            metrics = evaluate_recognition_predictions(refs, hyps)

            metrics_path = out_dir / "metrics" / f"{model_id}_page_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(metrics_path, metrics)

            row = {
                "dataset": dataset_name,
                "model_id": model_id,
                "display_name": model.get("display_name", model_id),
                "eval_mode": "full_page_vlm",
                "cer": metrics.get("val_rec_cer"),
                "wer": metrics.get("val_rec_wer"),
                "chrf": metrics.get("val_rec_chrf"),
                "exact_match": metrics.get("val_rec_exact_match"),
                "page_count": len(page_groups),
            }
            results.append(row)
            cer_pct = f"{metrics.get('val_rec_cer', float('nan')) * 100:.2f}%"
            print(f"    {model.get('display_name', model_id):40s}  page-CER={cer_pct}  [full-page VLM]")

    # ------------------------------------------------------------------ #
    # 3. Write per-dataset leaderboard                                    #
    # ------------------------------------------------------------------ #
    leaderboard_path = out_dir / "leaderboard.csv"
    if results:
        _write_csv(leaderboard_path, results)
        print(f"  [{dataset_name}] Leaderboard written to {leaderboard_path}")

    return {"dataset": dataset_name, "status": "done", "results": results}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pipeline evaluation: detection+recognition for line models, full-page for VLMs."
    )
    p.add_argument("--dataset-name", default=None,
                   help="Single dataset name (matches sub-dir under pipeline-crops-root).")
    p.add_argument("--pipeline-crops-dir", default=None,
                   help="Path to directory produced by prepare_pipeline_crops.py for a single dataset.")
    p.add_argument("--page-manifest", default=None,
                   help="Path to the original page-level manifest (for VLM evaluation).")
    p.add_argument("--all-datasets", action="store_true",
                   help="Evaluate all 3 selected external datasets.")
    p.add_argument("--pipeline-crops-root", default=_PIPELINE_CROPS_ROOT,
                   help="Root directory with per-dataset pipeline crop outputs.")
    p.add_argument("--manifest-root", default=_EXTERNAL_MANIFEST_ROOT,
                   help="Root directory with original external benchmark manifests.")
    p.add_argument("--config", default="configs/eval_baselines.yaml",
                   help="Path to eval_baselines.yaml.")
    p.add_argument("--out", default=_PIPELINE_EVAL_ROOT,
                   help="Output root directory.")
    p.add_argument("--gpus", default="0,1,2,3",
                   help="GPU IDs for parallel baseline inference.")
    p.add_argument("--force", action="store_true",
                   help="Re-run predictions even if files already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build job list: (dataset_name, pipeline_crops_dir, page_manifest_path, out_dir)
    jobs: list[tuple[str, Path, Path, Path]] = []

    if args.all_datasets:
        crops_root = Path(args.pipeline_crops_root)
        manifest_root = Path(args.manifest_root)
        out_root = Path(args.out)
        for ds_name in _ALL_DATASETS:
            crops_dir = crops_root / ds_name
            if not crops_dir.exists():
                print(f"WARNING: Pipeline crops not found for {ds_name} at {crops_dir}. "
                      "Run prepare_pipeline_crops.py first. Skipping.")
                continue
            # Find the matching page manifest (may have tmp_ prefix)
            page_manifest = None
            for prefix in ("", "tmp_"):
                candidate = manifest_root / f"{prefix}{ds_name}.jsonl"
                if candidate.exists():
                    page_manifest = candidate
                    break
            if page_manifest is None:
                print(f"WARNING: Page manifest not found for {ds_name}. VLMs will be skipped.")
                page_manifest = Path("/dev/null")
            jobs.append((ds_name, crops_dir, page_manifest, out_root / ds_name))
    elif args.dataset_name:
        if not args.pipeline_crops_dir:
            # Use default location
            crops_dir = Path(args.pipeline_crops_root) / args.dataset_name
        else:
            crops_dir = Path(args.pipeline_crops_dir)
        page_manifest = Path(args.page_manifest) if args.page_manifest else Path("/dev/null")
        jobs.append((
            args.dataset_name,
            crops_dir,
            page_manifest,
            Path(args.out) / args.dataset_name,
        ))
    else:
        print("ERROR: specify --dataset-name or --all-datasets")
        sys.exit(1)

    all_results: list[dict[str, Any]] = []
    for ds_name, crops_dir, page_manifest, out_dir in jobs:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        r = _eval_dataset(
            dataset_name=ds_name,
            pipeline_crops_dir=crops_dir,
            page_manifest_path=page_manifest,
            config=args.config,
            out_dir=out_dir,
            gpus=args.gpus,
            force=args.force,
        )
        all_results.extend(r.get("results", []))

    # ------------------------------------------------------------------ #
    # Write combined leaderboard across all datasets                      #
    # ------------------------------------------------------------------ #
    if all_results:
        combined_path = Path(args.out) / "combined_leaderboard.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(combined_path, all_results)
        print(f"\nCombined leaderboard → {combined_path}")

        # Also print a quick summary table
        print("\n=== Combined Results ===")
        print(f"{'Model':<42} {'Dataset':<35} {'Mode':<18} {'CER%':>6}")
        print("-" * 108)
        for r in sorted(all_results, key=lambda x: (x["dataset"], x.get("cer") or 1.0)):
            cer_pct = f"{(r.get('cer') or float('nan')) * 100:.2f}"
            print(f"{r['display_name']:<42} {r['dataset']:<35} {r['eval_mode']:<18} {cer_pct:>6}")

    write_json(Path(args.out) / "pipeline_eval_summary.json", {"results": all_results})
    print(f"\nDone. Summary → {Path(args.out) / 'pipeline_eval_summary.json'}")


if __name__ == "__main__":
    main()
