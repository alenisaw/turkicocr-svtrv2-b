#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any

from turkicocr.utils import read_yaml, write_json

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pdparams", ".pdopt", ".pt", ".pth", ".onnx", ".gguf")
DATA_SUFFIXES = (".parquet", ".arrow", ".jsonl", ".zip", ".tar", ".tar.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate HF model/dataset snapshot sizes without downloading files."
    )
    p.add_argument("--config", default="configs/assets.yaml")
    p.add_argument("--out", default="outputs/assets/asset_size_estimate.json")
    p.add_argument("--asset", action="append", default=[], help="Shell-style id/role/kind filter.")
    p.add_argument("--asset-root", default=None, help="Storage root used for free-space reporting.")
    p.add_argument("--include-future-checkpoint", action="store_true")
    p.add_argument("--timeout", type=float, default=30.0)
    return p.parse_args()


def _matches_asset(asset: dict[str, Any], patterns: list[str]) -> bool:
    if not patterns:
        return True
    candidates = [
        str(asset.get("id", "")),
        str(asset.get("role", "")),
        str(asset.get("kind", "")),
        str(asset.get("display_name", "")),
    ]
    return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns for candidate in candidates)


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


def _nearest_existing_parent(path: str | Path) -> Path:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return Path(".")
        candidate = parent
    return candidate


def _disk_report(path: str | Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / 1024**3, 2),
        "used_gb": round(usage.used / 1024**3, 2),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def _asset_rows(cfg: dict[str, Any], asset_root: str | None, include_future: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = dict(cfg.get("dataset", {}))
    if dataset:
        dataset["kind"] = "dataset"
        dataset["repo_type"] = dataset.get("repo_type", "dataset")
        dataset["local_dir"] = _local_dir_for_asset(dataset.get("local_dir"), asset_root)
        rows.append(dataset)
    for model in cfg.get("models", []):
        row = dict(model)
        if "local_dir" not in row:
            continue
        if row.get("role") == "ours" and not include_future:
            row["kind"] = "future_checkpoint"
            row["status"] = "pending_training"
            row["local_dir"] = _local_dir_for_asset(row.get("local_dir"), asset_root)
            rows.append(row)
            continue
        row["kind"] = "model"
        row["repo_type"] = row.get("repo_type", "model")
        row["local_dir"] = _local_dir_for_asset(row.get("local_dir"), asset_root)
        rows.append(row)
    for dataset in cfg.get("external_benchmarks", []):
        row = dict(dataset)
        row["kind"] = "external_benchmark"
        row["repo_type"] = row.get("repo_type", "dataset")
        row["local_dir"] = _local_dir_for_asset(row.get("local_dir"), asset_root)
        rows.append(row)
    return rows


def _suffix_bytes(files: list[dict[str, Any]], suffixes: tuple[str, ...]) -> int:
    return sum(
        int(file.get("size") or 0)
        for file in files
        if any(str(file.get("path", "")).endswith(suffix) for suffix in suffixes)
    )


def _estimate_repo(api: Any, asset: dict[str, Any], timeout: float) -> dict[str, Any]:
    repo_id = str(asset["id"])
    repo_type = str(asset.get("repo_type", "model"))
    if repo_type == "dataset":
        info = api.dataset_info(repo_id, files_metadata=True, timeout=timeout)
    else:
        info = api.model_info(repo_id, files_metadata=True, timeout=timeout)
    files = [
        {"path": sibling.rfilename, "size": getattr(sibling, "size", None)}
        for sibling in getattr(info, "siblings", [])
    ]
    known_files = [file for file in files if file["size"] is not None]
    total_bytes = sum(int(file["size"] or 0) for file in known_files)
    largest = sorted(known_files, key=lambda file: int(file["size"] or 0), reverse=True)[:8]
    return {
        "status": "estimated",
        "revision": getattr(info, "sha", None),
        "file_count": len(files),
        "known_size_file_count": len(known_files),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1024**3, 3),
        "weights_gb": round(_suffix_bytes(files, WEIGHT_SUFFIXES) / 1024**3, 3),
        "data_gb": round(_suffix_bytes(files, DATA_SUFFIXES) / 1024**3, 3),
        "largest_files": largest,
    }


def main() -> None:
    args = parse_args()
    from huggingface_hub import HfApi

    cfg = read_yaml(args.config)
    rows = [
        row
        for row in _asset_rows(cfg, args.asset_root, args.include_future_checkpoint)
        if _matches_asset(row, args.asset)
    ]
    disk_path = _nearest_existing_parent(args.asset_root or cfg.get("cache", {}).get("hf_home", "."))
    api = HfApi()
    assets = []
    for asset in rows:
        row = {
            "id": asset.get("id"),
            "kind": asset.get("kind"),
            "role": asset.get("role"),
            "repo_type": asset.get("repo_type"),
            "local_dir": asset.get("local_dir"),
        }
        if asset.get("kind") == "future_checkpoint":
            row.update({"status": "pending_training", "total_gb": None})
        else:
            try:
                row.update(_estimate_repo(api, asset, args.timeout))
            except Exception as exc:
                row.update({"status": "failed", "error": str(exc), "total_gb": None})
        assets.append(row)
    estimated = [row for row in assets if row.get("status") == "estimated"]
    total_gb = round(sum(float(row.get("total_gb") or 0) for row in estimated), 3)
    report = {
        "config": args.config,
        "asset_filters": args.asset,
        "asset_root": args.asset_root,
        "disk": _disk_report(disk_path),
        "assets": assets,
        "summary": {
            "assets_total": len(assets),
            "estimated": len(estimated),
            "failed": sum(1 for row in assets if row.get("status") == "failed"),
            "pending_training": sum(1 for row in assets if row.get("status") == "pending_training"),
            "total_estimated_gb": total_gb,
            "fits_current_disk": total_gb <= _disk_report(disk_path)["free_gb"],
        },
    }
    write_json(args.out, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Asset size estimate written to {args.out}")


if __name__ == "__main__":
    main()
