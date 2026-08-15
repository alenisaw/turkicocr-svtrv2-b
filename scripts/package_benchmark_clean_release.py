#!/usr/bin/env python
"""Package the clean benchmark/pipeline-fix release ZIP.

Assembles README/LICENSE/CITATION, model cards, the canonical metrics tree
(from the asset disk, not the git repo), and the .agent reports into a single
ZIP written to the repo root -- matching the existing convention for release
archives (e.g. turkicocr_aist_benchmark_suite_*.zip), which are build
artifacts and stay untracked (see .gitignore's `*.zip` rule).
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the clean benchmark release ZIP.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--metrics-root",
        default=str(get_asset_root() / "outputs/turkicocr_benchmark_clean/metrics"),
    )
    parser.add_argument("--reports-root", default=".agent/reports")
    parser.add_argument("--name", default="turkicocr_benchmark_clean_20260707")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        if any(part.startswith(("tmp_", "tmp shard")) for part in item.parts):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    metrics_root = Path(args.metrics_root)
    reports_root = Path(args.reports_root)

    staging = Path(f"/tmp/{args.name}_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in ("README.md", "LICENSE", "CITATION.cff", "NOTICE"):
        src = repo_root / name
        if src.exists():
            shutil.copy2(src, staging / name)

    _copy_tree(repo_root / "docs", staging / "docs")

    for cfg in ("eval_page_oracle.yaml", "eval_baselines.yaml"):
        src = repo_root / "configs" / cfg
        if src.exists():
            (staging / "configs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, staging / "configs" / cfg)

    for script in ("analyze_page_pipeline_errors.py", "run_oracle_zone_page_eval.py", "sync_canonical_metrics.py"):
        src = repo_root / "scripts" / script
        if src.exists():
            (staging / "scripts").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, staging / "scripts" / script)

    # Canonical metrics: recognition_line_full, deployment, page_oracle, zone_from_lines,
    # page_detected/layout_aware. Detected-page "simple" (pre-fix reading order) is
    # intentionally excluded from the release -- layout_aware is the current, correct
    # result; simple stays available on the asset disk only.
    for sub in ("recognition_line_full", "deployment", "page_oracle", "zone_from_lines"):
        _copy_tree(metrics_root / sub, staging / "metrics" / sub)
    for mode in ("layout_aware",):
        for dataset_dir in (metrics_root / "page_detected" / mode).glob("*"):
            if not dataset_dir.is_dir():
                continue
            for name in ("page_diagnostics.csv", "page_diagnostics.json", "leaderboard.csv", "leaderboard.json"):
                src = dataset_dir / name
                if src.exists():
                    dst = staging / "metrics" / "page_detected" / mode / dataset_dir.name / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            metrics_dir = dataset_dir / "metrics"
            if metrics_dir.exists():
                _copy_tree(metrics_dir, staging / "metrics" / "page_detected" / mode / dataset_dir.name / "metrics")
            pred_dir = dataset_dir / "page_predictions"
            if pred_dir.exists():
                _copy_tree(pred_dir, staging / "metrics" / "page_detected" / mode / dataset_dir.name / "page_predictions")

    _copy_tree(reports_root, staging / "reports")

    out_path = Path(args.out) if args.out else repo_root / f"{args.name}.zip"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in staging.rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(staging))

    shutil.rmtree(staging)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
