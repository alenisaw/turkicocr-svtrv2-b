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

from turkicocr.utils import read_yaml, write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate TurkicOCR config into an OpenOCR training command.")
    parser.add_argument("--turkicocr-config", default="configs/train_svtrv2_b_rec.yaml")
    parser.add_argument("--openocr-root", default=str(get_asset_root() / "openocr/repo"))
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _set_transform_value(transforms: list[dict], op_name: str, key: str, value) -> None:
    for item in transforms:
        if op_name in item:
            item[op_name][key] = value


def ensure_classifier_head_stripped_checkpoint(source: str, asset_root: str) -> str:
    source_path = Path(source)
    out = Path(asset_root) / "models" / "OpenOCR" / "SVTRv2-B" / f"{source_path.stem}_no_classifier_head.pth"
    if out.exists():
        return str(out)

    import torch

    checkpoint = torch.load(source_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    filtered = {key: value for key, value in state_dict.items() if not key.startswith("decoder.fc.")}
    if "state_dict" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["state_dict"] = filtered
    else:
        checkpoint = filtered
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    return str(out)


def write_resolved_openocr_config(config_path: str, openocr_root: str, asset_root: str) -> Path:
    cfg = read_yaml(config_path)
    openocr_cfg = cfg.get("openocr", {}).get("base_config", "configs/rec/svtrv2/svtrv2_ch.yml")
    base_path = Path(openocr_root) / openocr_cfg
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    data = cfg["data"]
    train = cfg.get("training", {})
    validation = cfg.get("recognition_validation", {})
    max_text_length = int(train.get("max_text_length", 256))
    image_width = int(train.get("image_width", 640))
    image_height = int(train.get("image_height", 48))

    resolved = copy.deepcopy(base)
    resolved["Global"]["output_dir"] = cfg["run"]["output_dir"]
    pretrained_model = f"{cfg['model']['pretrained']}/openocr_svtrv2_ch.pth"
    if cfg.get("model", {}).get("replace_classifier_head", True):
        pretrained_model = ensure_classifier_head_stripped_checkpoint(pretrained_model, asset_root)
    resolved["Global"]["pretrained_model"] = pretrained_model
    if cfg.get("model", {}).get("resume_checkpoint"):
        resolved["Global"]["checkpoints"] = cfg["model"]["resume_checkpoint"]
    resolved["Global"]["character_dict_path"] = cfg["model"]["charset"]
    resolved["Global"]["max_text_length"] = max_text_length
    resolved["Global"]["use_space_char"] = True
    resolved["Global"]["use_amp"] = bool(train.get("openocr_use_amp", False))
    resolved["Global"]["cal_metric_during_train"] = bool(train.get("openocr_cal_metric_during_train", False))
    resolved["Global"]["epoch_num"] = int(train.get("max_epochs", 10))
    resolved["Global"]["project_name"] = cfg["run"]["name"]
    resolved["Global"]["save_epoch_step"] = [0, int(train.get("save_every_epoch", 1))]
    resolved["Global"]["eval_epoch_step"] = [0, int(train.get("eval_every_epoch", 1))]
    resolved["PostProcess"]["character_dict_path"] = cfg["model"]["charset"]
    resolved["PostProcess"]["use_space_char"] = True
    resolved["Optimizer"]["lr"] = float(train.get("learning_rate", 0.0003))
    resolved["Optimizer"]["weight_decay"] = float(train.get("weight_decay", 0.00005))
    if "LRScheduler" in resolved:
        resolved["LRScheduler"]["warmup_epoch"] = int(train.get("warmup_epochs", 1))

    resolved["Train"]["dataset"]["data_dir"] = "/"
    resolved["Train"]["dataset"]["label_file_list"] = [data["train_openocr_txt"]]
    _set_transform_value(
        resolved["Train"]["dataset"]["transforms"],
        "CTCLabelEncode",
        "character_dict_path",
        cfg["model"]["charset"],
    )
    _set_transform_value(
        resolved["Train"]["dataset"]["transforms"],
        "CTCLabelEncode",
        "max_text_length",
        max_text_length,
    )
    _set_transform_value(
        resolved["Train"]["dataset"]["transforms"],
        "RecTVResize",
        "image_shape",
        [image_height, image_width],
    )
    resolved["Train"]["loader"]["batch_size_per_card"] = int(train.get("batch_size_per_gpu", 64))
    resolved["Train"]["loader"]["num_workers"] = int(train.get("num_workers", 8))
    if "pin_memory" in train:
        resolved["Train"]["loader"]["pin_memory"] = bool(train.get("pin_memory", False))

    resolved["Eval"]["dataset"]["data_dir"] = "/"
    resolved["Eval"]["dataset"]["label_file_list"] = [data["val_openocr_txt"]]
    _set_transform_value(
        resolved["Eval"]["dataset"]["transforms"],
        "CTCLabelEncode",
        "character_dict_path",
        cfg["model"]["charset"],
    )
    _set_transform_value(
        resolved["Eval"]["dataset"]["transforms"],
        "CTCLabelEncode",
        "max_text_length",
        max_text_length,
    )
    _set_transform_value(
        resolved["Eval"]["dataset"]["transforms"],
        "RecDynamicResize",
        "image_shape",
        [image_height, image_width],
    )
    resolved["Eval"]["loader"]["batch_size_per_card"] = int(validation.get("batch_size", 64))
    resolved["Eval"]["loader"]["num_workers"] = int(validation.get("num_workers", 8))
    if "pin_memory" in train:
        resolved["Eval"]["loader"]["pin_memory"] = bool(train.get("pin_memory", False))

    out = Path(asset_root) / "openocr" / "configs" / f"{cfg['run']['name']}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def build_command(config_path: str, openocr_root: str, asset_root: str) -> list[str]:
    cfg = read_yaml(config_path)
    train = cfg.get("training", {})
    resolved_config = write_resolved_openocr_config(config_path, openocr_root, asset_root)
    visible_devices = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    default_gpus = len(visible_devices) if visible_devices else 1
    num_gpus = int(train.get("num_gpus", default_gpus))
    if num_gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            str(Path(openocr_root) / "tools" / "train_rec.py"),
            "--c",
            str(resolved_config),
        ]
        if not bool(train.get("openocr_eval_during_train", False)):
            cmd.append("--no-eval")
        return cmd
    return [
        sys.executable,
        str(Path(openocr_root) / "tools" / "train_rec.py"),
        "--c",
        str(resolved_config),
    ]


def resolved_config_from_command(cmd: list[str]) -> str:
    if "--c" in cmd:
        return cmd[cmd.index("--c") + 1]
    return ""


def main() -> None:
    args = parse_args()
    cmd = build_command(args.turkicocr_config, args.openocr_root, args.asset_root)
    report = {
        "command": cmd,
        "resolved_openocr_config": resolved_config_from_command(cmd),
        "dry_run": args.dry_run,
    }
    write_json(Path(args.asset_root) / "outputs" / "audit" / "openocr_train_command.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.dry_run:
        subprocess.run(cmd, cwd=args.openocr_root, check=True)


if __name__ == "__main__":
    main()
