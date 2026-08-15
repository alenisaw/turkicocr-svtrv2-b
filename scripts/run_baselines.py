#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from turkicocr.baselines import load_baseline_config, run_baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run configured OCR baselines.")
    p.add_argument("--config", default="configs/eval_baselines.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=None, help="Override evaluation manifest/glob.")
    p.add_argument("--device", default=None, help="Backend device hint, e.g. gpu:0 or cpu.")
    p.add_argument("--model-id", action="append", default=[], help="Run only the selected model id. Repeatable.")
    p.add_argument("--isolated", action="store_true", help="Run each configured model in a separate subprocess.")
    p.add_argument("--max-samples", type=int, default=None, help="Stop after this many manifest rows.")
    p.add_argument("--force", action="store_true", help="Regenerate predictions even if files exist.")
    p.add_argument(
        "--development-echo",
        action="store_true",
        help="Write echo predictions to validate pipeline only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.isolated and not args.model_id:
        status_path = Path(args.out) / "baseline_status.json"
        if args.force and status_path.exists():
            status_path.unlink()
        for model in load_baseline_config(args.config):
            # Skip if parent directory already contains final prediction file
            final_pred_path = Path(args.out).parent / f"{model.id}_predictions.jsonl"
            if final_pred_path.exists() and final_pred_path.stat().st_size > 0 and not args.force:
                print(f"Skipping {model.id} in sharding because final predictions exist at {final_pred_path}")
                continue
            cmd = [
                sys.executable,
                __file__,
                "--config",
                args.config,
                "--out",
                args.out,
                "--model-id",
                model.id,
            ]
            if args.manifest:
                cmd.extend(["--manifest", args.manifest])
            if args.device:
                child_device = args.device
                env = os.environ.copy()
                if args.device.startswith("gpu:"):
                    gpu_id = args.device.split(":", 1)[1]
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    env["FLAGS_selected_gpus"] = "0"
                    child_device = "gpu:0"
                cmd.extend(["--device", child_device])
            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.force:
                cmd.append("--force")
            if args.development_echo:
                cmd.append("--development-echo")
            subprocess.run(cmd, check=True, env=env if args.device else None)
        return
    result = run_baselines(
        args.config,
        args.out,
        development_echo=args.development_echo,
        manifest_override=args.manifest,
        device=args.device,
        force=args.force,
        model_ids=set(args.model_id) if args.model_id else None,
        max_samples=args.max_samples,
    )
    print(result)


if __name__ == "__main__":
    main()
