#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
from pathlib import Path
from typing import Any

from turkicocr.utils import ensure_dir, read_yaml, write_json
from turkicocr.utils import get_asset_root

METADATA_PATTERNS = [
    "README*",
    "LICENSE*",
    "NOTICE*",
    "config*.json",
    "*.yaml",
    "*.yml",
    "tokenizer*",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
    "*.model",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare local HF caches for datasets and baseline models."
    )
    p.add_argument("--config", default="configs/assets.yaml")
    p.add_argument(
        "--metadata-only", action="store_true", help="Download cards/config/tokenizers only."
    )
    p.add_argument("--include-datasets", action="store_true", help="Include dataset snapshots.")
    p.add_argument(
        "--include-benchmarks", action="store_true", help="Include external benchmark snapshots."
    )
    p.add_argument(
        "--include-weights",
        action="store_true",
        help="Download full model weights when disk allows.",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve plan without downloading.")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument(
        "--asset",
        action="append",
        default=[],
        help=(
            "Download/audit only matching assets. Repeatable. Matches id, role, kind, "
            "or display_name with shell-style globs, e.g. --asset 'PaddlePaddle/*'."
        ),
    )
    p.add_argument(
        "--asset-root",
        default=None,
        help=(
            "Place configured external/ model and dataset dirs under this storage root. "
            "Example: --asset-root ./assets."
        ),
    )
    p.add_argument(
        "--include-future-checkpoint",
        action="store_true",
        help="Try to download a future TurkicOCR-SVTRv2-B checkpoint from Hub before training.",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Minimum free GB required before full weight downloads.",
    )
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


def _snapshot_download(
    repo_id: str,
    repo_type: str,
    local_dir: str | Path,
    cache_dir: str | Path,
    max_workers: int,
    metadata_only: bool,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": None if repo_type == "model" else repo_type,
        "local_dir": str(local_dir),
        "cache_dir": str(cache_dir),
        "max_workers": max_workers,
    }
    if metadata_only:
        kwargs["allow_patterns"] = METADATA_PATTERNS
    path = snapshot_download(**kwargs)
    return {"status": "downloaded", "path": path}


def _disk_report(path: str | Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": round(usage.total / 1024**3, 2),
        "used_gb": round(usage.used / 1024**3, 2),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def _nearest_existing_parent(path: str | Path) -> Path:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return Path(".")
        candidate = parent
    return candidate


def _asset_plan(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    if args.include_datasets:
        dataset = dict(cfg["dataset"])
        dataset["kind"] = "dataset"
        assets.append(dataset)
    for model in cfg.get("models", []):
        if "local_dir" not in model:
            assets.append({**model, "kind": "system"})
            continue
        if model.get("role") == "ours" and not args.include_future_checkpoint:
            assets.append({**model, "kind": "future_checkpoint", "status": "pending_training"})
            continue
        assets.append({**model, "kind": "model"})
    if args.include_benchmarks:
        for dataset in cfg.get("external_benchmarks", []):
            assets.append({**dataset, "kind": "external_benchmark"})
    prepared = []
    for asset in assets:
        row = dict(asset)
        row["local_dir"] = _local_dir_for_asset(row.get("local_dir"), args.asset_root)
        prepared.append(row)
    return [asset for asset in prepared if _matches_asset(asset, args.asset)]


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    cache_cfg = cfg.get("cache", {})
    hf_home = (
        Path(args.asset_root, "huggingface")
        if args.asset_root
        else Path(cache_cfg.get("hf_home", "external/huggingface"))
    )
    disk_path = _nearest_existing_parent(hf_home)
    if not args.dry_run:
        ensure_dir(hf_home)
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    max_workers = args.max_workers or int(cache_cfg.get("max_parallel_downloads", 8))
    metadata_only = args.metadata_only or not args.include_weights
    min_free_gb = args.min_free_gb
    if min_free_gb is None:
        min_free_gb = float(cache_cfg.get("min_free_gb_for_full_download", 250))
    disk = _disk_report(disk_path)
    if args.include_weights and not args.dry_run and disk["free_gb"] < min_free_gb:
        raise SystemExit(
            f"Refusing full weight download: {disk['free_gb']} GB free, "
            f"need at least {min_free_gb} GB. Use metadata-only or add storage."
        )
    assets = _asset_plan(cfg, args)
    status: dict[str, Any] = {
        "config": args.config,
        "metadata_only": metadata_only,
        "dry_run": args.dry_run,
        "asset_filters": args.asset,
        "asset_root": args.asset_root,
        "disk": disk,
        "min_free_gb_for_full_download": min_free_gb,
        "assets": [],
    }

    for asset in assets:
        row = {
            "id": asset.get("id"),
            "kind": asset.get("kind"),
            "role": asset.get("role"),
            "display_name": asset.get("display_name"),
            "local_dir": asset.get("local_dir"),
        }
        if asset.get("kind") in {"system", "future_checkpoint"}:
            row["status"] = asset.get("status", "managed_by_system_package")
        elif args.dry_run:
            row["status"] = "planned"
        else:
            try:
                repo_type = asset.get(
                    "repo_type", "dataset" if asset.get("kind") != "model" else "model"
                )
                result = _snapshot_download(
                    repo_id=str(asset["id"]),
                    repo_type=repo_type,
                    local_dir=asset["local_dir"],
                    cache_dir=hf_home,
                    max_workers=max_workers,
                    metadata_only=metadata_only,
                )
                row.update(result)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
        status["assets"].append(row)

    status_path = Path(cache_cfg.get("status_file", "outputs/assets/asset_status.json"))
    write_json(status_path, status)
    print(f"Asset status written to {status_path}")


if __name__ == "__main__":
    main()
