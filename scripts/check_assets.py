#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from turkicocr.utils import read_yaml, write_json

METADATA_NAMES = (
    "README.md",
    "README_ZH.md",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.model",
)

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pdparams",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit local model/dataset asset preparation.")
    p.add_argument("--config", default="configs/assets.yaml")
    p.add_argument("--out", default="outputs/assets/asset_audit.json")
    p.add_argument(
        "--asset-root",
        default=None,
        help=(
            "Audit configured external/ model and dataset dirs under this storage root. "
            "Use the same value as prepare_assets.py --asset-root."
        ),
    )
    return p.parse_args()


def _disk_report(path: str | Path = ".") -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": round(usage.total / 1024**3, 2),
        "used_gb": round(usage.used / 1024**3, 2),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def _asset_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    dataset = dict(cfg.get("dataset", {}))
    if dataset:
        dataset["kind"] = "dataset"
        rows.append(dataset)
    for model in cfg.get("models", []):
        row = dict(model)
        row["kind"] = "system" if "local_dir" not in row else "model"
        if row.get("role") == "ours":
            row["kind"] = "future_checkpoint"
        rows.append(row)
    for dataset in cfg.get("external_benchmarks", []):
        row = dict(dataset)
        row["kind"] = "external_benchmark"
        rows.append(row)
    return rows


def _has_metadata(path: Path) -> bool:
    if not path.exists():
        return False
    return any((path / name).exists() for name in METADATA_NAMES)


def _has_weights(path: Path) -> bool:
    if not path.exists():
        return False
    return any(file.suffix in WEIGHT_SUFFIXES for file in path.rglob("*") if file.is_file())


def _local_dir_for_asset(local_dir: str | Path | None, asset_root: str | Path | None) -> str | None:
    if local_dir is None:
        return None
    path = Path(local_dir)
    if asset_root is None:
        return str(path)
    parts = path.parts
    if parts and parts[0] == "external":
        return str(Path(asset_root).joinpath(*parts[1:]))
    return str(path)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    cache = cfg.get("cache", {})
    disk = _disk_report(".")
    min_free = float(cache.get("min_free_gb_for_full_download", 250))
    rows = []
    for asset in _asset_rows(cfg):
        local_dir = asset.get("local_dir")
        resolved_local_dir = _local_dir_for_asset(local_dir, args.asset_root)
        kind = asset.get("kind")
        row: dict[str, Any] = {
            "id": asset.get("id"),
            "kind": kind,
            "role": asset.get("role"),
            "display_name": asset.get("display_name"),
            "local_dir": resolved_local_dir,
        }
        if kind == "system":
            row["metadata_ready"] = True
            row["weights_ready"] = None
            row["status"] = "system_dependency"
        elif kind == "future_checkpoint":
            row["metadata_ready"] = False
            row["weights_ready"] = False
            row["status"] = "pending_training"
        else:
            path = Path(str(resolved_local_dir))
            row["metadata_ready"] = _has_metadata(path)
            row["weights_ready"] = _has_weights(path)
            row["status"] = "metadata_ready" if row["metadata_ready"] else "missing_metadata"
            if row["weights_ready"]:
                row["status"] = "weights_ready"
        rows.append(row)
    report = {
        "config": args.config,
        "asset_root": args.asset_root,
        "disk": disk,
        "min_free_gb_for_full_download": min_free,
        "full_download_allowed": disk["free_gb"] >= min_free,
        "assets": rows,
        "summary": {
            "count": len(rows),
            "metadata_ready": sum(1 for row in rows if row.get("metadata_ready")),
            "weights_ready": sum(1 for row in rows if row.get("weights_ready") is True),
            "system_dependencies": sum(1 for row in rows if row.get("kind") == "system"),
            "pending_training": sum(1 for row in rows if row.get("kind") == "future_checkpoint"),
        },
    }
    write_json(args.out, report)
    print(
        f"Asset audit: {report['summary']['metadata_ready']}/{report['summary']['count']} "
        f"metadata-ready, {report['summary']['weights_ready']} with weights. "
        f"Full download allowed: {report['full_download_allowed']}"
    )


if __name__ == "__main__":
    main()
