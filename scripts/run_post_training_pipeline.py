#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from turkicocr.utils import ensure_dir, write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TurkicOCR post-training evaluation and reports.")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-name", default="turkicocr_svtrv2_b_rec_line_v1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/train_svtrv2_b_rec_line.yaml")
    parser.add_argument("--devices", default="gpu")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--development-echo-zone", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path, stage: str, status: dict) -> None:
    print("+ " + " ".join(cmd), flush=True)
    status.setdefault("commands", []).append({"stage": stage, "cmd": cmd})
    subprocess.run(cmd, cwd=cwd, check=True)


def _checkpoint(args: argparse.Namespace, asset_root: Path) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    run_dir = asset_root / "checkpoints" / args.run_name
    for name in ("best.pth", "latest.pth"):
        path = run_dir / name
        if path.exists():
            return path
    epochs = sorted(run_dir.glob("epoch_*.pth"))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(f"No checkpoint found for {args.run_name}")


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root)
    repo_root = Path(args.repo_root).resolve()
    checkpoint = _checkpoint(args, asset_root)
    out_root = ensure_dir(asset_root / "outputs" / "recognition")
    reports = ensure_dir(out_root / "reports" / args.run_name)
    predictions = ensure_dir(out_root / "predictions" / args.run_name / "post_training")
    metrics = ensure_dir(out_root / "metrics" / args.run_name / "post_training")
    manifests = asset_root / "outputs" / "recognition" / "manifests"
    charset = out_root / "charset_turkic_cyrillic.txt"
    status = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint),
        "reports": str(reports),
    }

    line_full_predictions = predictions / "line_full.jsonl"
    _run(
        [
            sys.executable,
            "scripts/run_recognition_inference.py",
            "--manifest",
            str(manifests / "diagnostic_rec_full_line_only.jsonl"),
            "--checkpoint",
            str(checkpoint),
            "--turkicocr-config",
            args.config,
            "--asset-root",
            str(asset_root),
            "--device",
            "cpu" if args.devices == "cpu" else "gpu",
            "--out",
            str(line_full_predictions),
        ],
        cwd=repo_root,
        stage="line_full_inference",
        status=status,
    )
    _run(
        [
            sys.executable,
            "scripts/run_recognition_eval.py",
            "--predictions",
            str(line_full_predictions),
            "--out",
            str(metrics / "line_full"),
            "--charset",
            str(charset),
            "--allow-fail",
        ],
        cwd=repo_root,
        stage="line_full_eval",
        status=status,
    )

    test_line_predictions = predictions / "line_test.jsonl"
    _run(
        [
            sys.executable,
            "scripts/run_recognition_inference.py",
            "--manifest",
            str(manifests / "test_rec_large_line_only.jsonl"),
            "--checkpoint",
            str(checkpoint),
            "--turkicocr-config",
            args.config,
            "--asset-root",
            str(asset_root),
            "--device",
            "cpu" if args.devices == "cpu" else "gpu",
            "--out",
            str(test_line_predictions),
        ],
        cwd=repo_root,
        stage="line_test_inference",
        status=status,
    )
    _run(
        [
            sys.executable,
            "scripts/build_recognition_reports.py",
            "--metrics-root",
            str(asset_root / "outputs" / "recognition" / "metrics" / args.run_name / "checkpoint_monitor"),
            "--predictions",
            str(line_full_predictions),
            "--out",
            str(reports),
        ],
        cwd=repo_root,
        stage="reports",
        status=status,
    )

    hardcase_manifest = manifests / "train_hardcase_rec.jsonl"
    _run(
        [
            sys.executable,
            "scripts/export_hardcases.py",
            "--predictions",
            str(line_full_predictions),
            "--out-manifest",
            str(hardcase_manifest),
            "--out-openocr",
            str(manifests / "train_hardcase_openocr.txt"),
        ],
        cwd=repo_root,
        stage="hardcases",
        status=status,
    )

    zone_out = metrics / "zone_from_lines"
    zone_cmd = [
        sys.executable,
        "scripts/run_zone_from_lines_eval.py",
        "--zones",
        str(manifests / "test_rec_large.jsonl"),
        "--lines",
        str(manifests / "test_rec_large_line_only.jsonl"),
        "--out",
        str(zone_out),
    ]
    if args.development_echo_zone:
        zone_cmd.append("--development-echo")
    else:
        zone_cmd.extend(["--line-predictions", str(test_line_predictions)])
    _run(zone_cmd, cwd=repo_root, stage="zone_from_lines", status=status)

    if not args.skip_baselines:
        _run(
            [
                sys.executable,
                "scripts/run_benchmark_suite.py",
                "--config",
                "configs/eval_baselines.yaml",
                "--asset-root",
                str(asset_root),
                "--repo-root",
                str(repo_root),
                "--out",
                str(asset_root / "outputs" / "benchmark_suite" / args.run_name),
                "--manifest",
                str(manifests / "test_rec_large_line_only.jsonl"),
                "--device",
                "gpu:0",
            ],
            cwd=repo_root,
            stage="benchmark_suite",
            status=status,
        )

    write_json(reports / "post_training_status.json", status)
    print(f"Post-training pipeline complete: {reports}")


if __name__ == "__main__":
    main()
