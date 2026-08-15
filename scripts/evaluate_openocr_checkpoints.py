#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenOCR checkpoints and promote by val_rec_cer.")
    parser.add_argument("--config", default="configs/train_svtrv2_b_rec.yaml")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--charset", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--include-latest", action="store_true")
    parser.add_argument("--inference-device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_", 1)[1]), stem
        except ValueError:
            pass
    if stem == "latest":
        return 10**12, stem
    return 10**11, stem


def discover_checkpoints(checkpoint_dir: Path, include_latest: bool) -> list[Path]:
    paths = list(checkpoint_dir.glob("epoch_*.pth"))
    if include_latest and (checkpoint_dir / "latest.pth").exists():
        paths.append(checkpoint_dir / "latest.pth")
    return sorted(paths, key=_checkpoint_sort_key)


def run_checkpoint_eval(args: argparse.Namespace, checkpoint: Path) -> dict:
    predictions = Path(args.predictions_dir) / f"{checkpoint.stem}.jsonl"
    metrics_dir = Path(args.metrics_dir) / checkpoint.stem
    infer_cmd = [
        sys.executable,
        "scripts/run_recognition_inference.py",
        "--manifest",
        args.manifest,
        "--checkpoint",
        str(checkpoint),
        "--turkicocr-config",
        args.config,
        "--asset-root",
        args.asset_root,
        "--device",
        args.inference_device,
        "--device-id",
        str(args.device_id),
        "--out",
        str(predictions),
    ]
    eval_cmd = [
        sys.executable,
        "scripts/run_recognition_eval.py",
        "--predictions",
        str(predictions),
        "--out",
        str(metrics_dir),
        "--charset",
        args.charset,
    ]
    subprocess.run(infer_cmd, check=True)
    proc = subprocess.run(eval_cmd, check=False)
    if proc.returncode not in {0, 2}:
        raise subprocess.CalledProcessError(proc.returncode, eval_cmd)
    metrics = json.loads((metrics_dir / "metrics_aggregate.json").read_text(encoding="utf-8"))
    gate = json.loads((metrics_dir / "recognition_gate_result.json").read_text(encoding="utf-8"))
    return {
        "checkpoint": str(checkpoint),
        "predictions": str(predictions),
        "metrics_dir": str(metrics_dir),
        "metrics": metrics,
        "gate": gate,
    }


def _is_better(candidate: dict, best: dict | None) -> bool:
    cer = float(candidate["metrics"].get("val_rec_cer", math.inf))
    if math.isnan(cer):
        return False
    if best is None:
        return True
    best_cer = float(best["metrics"].get("val_rec_cer", math.inf))
    return cer < best_cer


def write_summary(args: argparse.Namespace, results: list[dict], best: dict | None) -> None:
    out = Path(args.metrics_dir) / "checkpoint_eval_summary.json"
    write_json(
        out,
        {
            "checkpoint_dir": args.checkpoint_dir,
            "manifest": args.manifest,
            "evaluated_count": len(results),
            "best": best,
            "results": results,
        },
    )


def maybe_promote(args: argparse.Namespace, best: dict | None) -> None:
    if not best:
        return
    gate = best.get("gate", {})
    if not gate.get("promote_checkpoint", False):
        return
    source = Path(best["checkpoint"])
    checkpoint_dir = Path(args.checkpoint_dir)
    shutil.copy2(source, checkpoint_dir / "best.pth")
    write_json(checkpoint_dir / "best_val_rec_cer.json", best)


def evaluate_available(args: argparse.Namespace, seen: set[str], results: list[dict]) -> set[str]:
    best = min(
        (row for row in results if "val_rec_cer" in row.get("metrics", {})),
        key=lambda row: float(row["metrics"].get("val_rec_cer", math.inf)),
        default=None,
    )
    for checkpoint in discover_checkpoints(Path(args.checkpoint_dir), args.include_latest):
        key = str(checkpoint)
        if key in seen:
            continue
        result = run_checkpoint_eval(args, checkpoint)
        results.append(result)
        seen.add(key)
        if result["gate"].get("promote_checkpoint", False) and _is_better(result, best):
            best = result
            maybe_promote(args, best)
        write_summary(args, results, best)
    return seen


def main() -> None:
    args = parse_args()
    Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    results: list[dict] = []
    while True:
        seen = evaluate_available(args, seen, results)
        if not args.watch:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
