#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from turkicocr.recognition_format import load_recognition_manifest, to_openocr_format
from turkicocr.utils import read_yaml, write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurkicOCR SVTRv2-B training wrapper.")
    parser.add_argument("--config", default="configs/train_svtrv2_b_rec.yaml")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    parser.add_argument("--train")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and write resolved command without launching long training.")
    parser.add_argument("--launch", action="store_true", help="Actually launch OpenOCR training.")
    return parser.parse_args()


def write_sampled_openocr_list(source: str, limit: int, asset_root: str, run_name: str) -> str:
    out = Path(asset_root) / "outputs" / "audit" / f"{run_name}_train_openocr_first_{limit}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(source).open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            written += 1
            if written >= limit:
                break
    if written == 0:
        raise RuntimeError(f"No OpenOCR samples were written from {source}")
    return str(out)


def resolve_train_override(source: str, asset_root: str, run_name: str) -> tuple[str | None, str]:
    path = Path(source)
    if path.suffix == ".txt":
        return None, str(path)
    records = load_recognition_manifest(path)
    out = Path(asset_root) / "outputs" / "audit" / f"{run_name}_{path.stem}_openocr.txt"
    to_openocr_format(records, out)
    return str(path), str(out)


def write_resolved_turkicocr_config(cfg: dict, args: argparse.Namespace) -> Path | str:
    run_name = args.run_name or cfg["run"]["name"]
    resolved = copy.deepcopy(cfg)
    resolved["run"]["name"] = run_name
    resolved["run"]["output_dir"] = str(Path(args.asset_root) / "checkpoints" / run_name)
    if args.resume:
        resolved.setdefault("model", {})["resume_checkpoint"] = args.resume
    if args.train:
        manifest, openocr_txt = resolve_train_override(args.train, args.asset_root, run_name)
        if manifest is not None:
            resolved.setdefault("data", {})["train_manifest_jsonl"] = manifest
        resolved.setdefault("data", {})["train_openocr_txt"] = openocr_txt
    if os.environ.get("NUM_GPUS"):
        resolved.setdefault("training", {})["num_gpus"] = int(os.environ["NUM_GPUS"])
    if args.max_train_samples and Path(cfg["data"]["train_openocr_txt"]).exists():
        resolved["data"]["train_openocr_txt"] = write_sampled_openocr_list(
            cfg["data"]["train_openocr_txt"],
            args.max_train_samples,
            args.asset_root,
            run_name,
        )
    if resolved == cfg:
        return args.config

    out = Path(args.asset_root) / "outputs" / "audit" / f"{run_name}_turkicocr_train.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def count_lines(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    resolved_config = write_resolved_turkicocr_config(cfg, args)
    cfg = read_yaml(resolved_config)
    selector = cfg.get("checkpoint_selection", {})
    if selector.get("monitor_metric") not in {"val_rec_cer", "line_val_rec_cer"}:
        raise RuntimeError("checkpoint_selection.monitor_metric must be val_rec_cer or line_val_rec_cer")
    if selector.get("allow_train_loss_fallback", False):
        raise RuntimeError("allow_train_loss_fallback must be false")
    required = [
        cfg["model"]["pretrained"],
        cfg["data"].get("train_manifest_jsonl"),
        cfg["data"].get("val_manifest_jsonl"),
        cfg["data"].get("train_openocr_txt"),
        cfg["data"].get("val_openocr_txt"),
        cfg["recognition_validation"].get("diagnostic_manifest"),
    ]
    missing = [path for path in required if path and not Path(path).exists()]
    train_samples = count_lines(cfg["data"]["train_openocr_txt"]) if Path(cfg["data"]["train_openocr_txt"]).exists() else None
    train_cfg = cfg.get("training", {})
    num_gpus = int(train_cfg.get("num_gpus", os.environ.get("NUM_GPUS", 1)))
    batch_per_gpu = int(train_cfg.get("batch_size_per_gpu", 1))
    effective_batch = num_gpus * batch_per_gpu
    steps_per_epoch = train_samples // effective_batch if train_samples is not None else None
    total_steps = steps_per_epoch * int(train_cfg.get("max_epochs", 1)) if steps_per_epoch is not None else None
    report = {
        "run_name": args.run_name or cfg["run"]["name"],
        "config": str(resolved_config),
        "missing_required_paths": missing,
        "max_train_samples": args.max_train_samples,
        "train_samples": train_samples,
        "num_gpus": num_gpus,
        "batch_size_per_gpu": batch_per_gpu,
        "effective_global_batch": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "eval_schedule": "external checkpoint monitor after each epoch_N.pth; polling interval 300 seconds in full pipeline",
        "dry_run": args.dry_run or not args.launch,
    }
    out = Path(args.asset_root) / "outputs" / "audit" / "train_svtrv2_b_rec_preflight.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)
    cmd = [
        sys.executable,
        "scripts/run_openocr_train.py",
        "--turkicocr-config",
        str(resolved_config),
        "--asset-root",
        args.asset_root,
    ]
    if args.dry_run or not args.launch:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
