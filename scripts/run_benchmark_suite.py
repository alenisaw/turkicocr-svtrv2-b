#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from turkicocr.utils import ensure_dir, read_yaml, write_json
from turkicocr.utils import get_asset_root

SELECTED_EXTERNAL_BENCHMARKS = {
    "henrygagnier__kazakh-ocr",
    "alenisaw__turkicocr-cyrillic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configured OCR models across benchmark manifests and build a leaderboard."
    )
    parser.add_argument("--config", default="configs/eval_baselines.yaml")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--include-default-test", action="store_true")
    parser.add_argument("--extra-model-json", action="append", default=[])
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--development-echo", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows per benchmark/model.")
    parser.add_argument("--force", action="store_true", help="Regenerate predictions instead of reusing existing files.")
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _benchmark_manifests(args: argparse.Namespace, cfg: dict) -> list[tuple[str, str]]:
    manifests: list[tuple[str, str]] = []
    if args.include_default_test:
        default = cfg.get("evaluation", {}).get("manifest")
        if default:
            manifests.append(("turkicocr_test", str(default)))
    for value in args.manifest:
        path = Path(value)
        name = path.stem.replace(".", "_")
        manifests.append((name, value))
    external_root = Path(args.asset_root) / "outputs" / "external_benchmarks" / "manifests"
    if external_root.exists():
        for path in sorted(external_root.glob("*.jsonl")):
            if path.stem.startswith(("smoke", "debug")):
                continue
            benchmark_name = path.stem
            if benchmark_name.startswith("tmp_"):
                benchmark_name = benchmark_name.removeprefix("tmp_")
            if benchmark_name not in SELECTED_EXTERNAL_BENCHMARKS:
                continue
            manifests.append((benchmark_name, str(path)))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for name, manifest in manifests:
        key = f"{name}:{manifest}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, manifest))
    return unique


def _evaluate_baseline_predictions(
    repo_root: Path,
    baseline_dir: Path,
    metrics_dir: Path,
) -> list[dict]:
    rows = []
    status = _load_json(baseline_dir / "baseline_status.json")
    for model in status.get("models", []):
        pred = Path(str(model.get("predictions", "")))
        model_id = str(model.get("id", "unknown"))
        model_metrics_dir = ensure_dir(metrics_dir / model_id)
        row = {"model": model_id, "status": model.get("status"), "predictions": str(pred)}
        if model.get("error"):
            row["error"] = model.get("error")
        if pred.exists() and pred.stat().st_size > 0 and model.get("status") != "failed":
            _run(
                [
                    sys.executable,
                    "scripts/run_recognition_eval.py",
                    "--predictions",
                    str(pred),
                    "--out",
                    str(model_metrics_dir),
                    "--allow-fail",
                ],
                cwd=repo_root,
            )
            metrics = _load_json(model_metrics_dir / "metrics_aggregate.json")
            row.update(metrics)
            row["metrics"] = str(model_metrics_dir / "metrics_aggregate.json")
        rows.append(row)
    return rows


def _write_leaderboard_csv(path: Path, rows: list[dict]) -> None:
    keys = [
        "benchmark",
        "model",
        "status",
        "val_rec_count",
        "val_rec_cer",
        "val_rec_wer",
        "val_rec_chrf",
        "val_rec_exact_match",
        "val_rec_rare_char_cer",
        "val_rec_latency_ms_per_crop",
        "error",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    f'"{str(row.get(key, "")).replace(chr(34), chr(34) * 2)}"' for key in keys
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    cfg = read_yaml(args.config)
    out = ensure_dir(args.out or asset_root / "outputs" / "benchmark_suite")
    if args.extra_model_json:
        extras = [json.loads(item) for item in args.extra_model_json]
        cfg = {**cfg, "models": [*cfg.get("models", []), *extras]}
        runtime_config = out / "benchmark_eval_config.yaml"
        runtime_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        config_path = str(runtime_config)
    else:
        config_path = args.config
    manifests = _benchmark_manifests(args, cfg)
    leaderboard: list[dict] = []
    benchmark_rows = []

    for benchmark_name, manifest in manifests:
        benchmark_dir = ensure_dir(out / benchmark_name)
        baseline_dir = ensure_dir(benchmark_dir / "predictions")
        metrics_dir = ensure_dir(benchmark_dir / "metrics")
        if args.force:
            shutil.rmtree(baseline_dir, ignore_errors=True)
            shutil.rmtree(metrics_dir, ignore_errors=True)
            baseline_dir = ensure_dir(benchmark_dir / "predictions")
            metrics_dir = ensure_dir(benchmark_dir / "metrics")
        _run(
            [
                sys.executable,
                "scripts/run_baselines_parallel_sharded.py",
                "--config",
                config_path,
                "--out",
                str(baseline_dir),
                "--manifest",
                manifest,
                "--gpus",
                args.gpus,
                *(["--max-samples", str(args.max_samples)] if args.max_samples is not None else []),
                *(["--force"] if args.force else []),
                *(["--development-echo"] if args.development_echo else []),
            ],
            cwd=repo_root,
        )
        rows = _evaluate_baseline_predictions(repo_root, baseline_dir, metrics_dir)
        for row in rows:
            row["benchmark"] = benchmark_name
            row["manifest"] = manifest
        leaderboard.extend(rows)
        benchmark_rows.append(
            {
                "benchmark": benchmark_name,
                "manifest": manifest,
                "predictions_dir": str(baseline_dir),
                "metrics_dir": str(metrics_dir),
                "models": len(rows),
            }
        )

    sortable = [row for row in leaderboard if isinstance(row.get("val_rec_cer"), int | float)]
    best_by_benchmark = {}
    for benchmark_name, _ in manifests:
        candidates = [row for row in sortable if row.get("benchmark") == benchmark_name]
        if candidates:
            best_by_benchmark[benchmark_name] = min(candidates, key=lambda row: row["val_rec_cer"])

    write_json(
        out / "benchmark_suite_summary.json",
        {
            "benchmarks": benchmark_rows,
            "leaderboard": leaderboard,
            "best_by_benchmark": best_by_benchmark,
        },
    )
    _write_leaderboard_csv(out / "leaderboard.csv", leaderboard)
    print(f"Benchmark suite written to {out}")


if __name__ == "__main__":
    main()
