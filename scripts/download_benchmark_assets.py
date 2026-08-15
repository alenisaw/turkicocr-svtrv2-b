#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from turkicocr.utils import ensure_dir, read_yaml, write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or resume all model and dataset assets required by benchmark configs."
    )
    parser.add_argument("--config", default="configs/assets.yaml")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--models", action="store_true", help="Download model assets.")
    parser.add_argument("--datasets", action="store_true", help="Download external benchmark datasets.")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _selected(args: argparse.Namespace, kind: str) -> bool:
    if not args.models and not args.datasets:
        return True
    return bool((kind == "model" and args.models) or (kind == "dataset" and args.datasets))


def _download(repo_id: str, repo_type: str, local_dir: Path, cache_dir: Path, workers: int) -> dict[str, Any]:
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir),
            max_workers=workers,
            resume_download=True,
        )
        files = [p for p in local_dir.rglob("*") if p.is_file()]
        return {
            "id": repo_id,
            "repo_type": repo_type,
            "local_dir": str(local_dir),
            "snapshot_path": path,
            "status": "ok",
            "files": len(files),
            "size_bytes": sum(p.stat().st_size for p in files),
        }
    except Exception as exc:
        return {
            "id": repo_id,
            "repo_type": repo_type,
            "local_dir": str(local_dir),
            "status": "failed",
            "error": repr(exc),
        }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    asset_root = Path(args.asset_root).resolve()
    cache_dir = Path(args.cache_dir or asset_root / "huggingface").resolve()
    ensure_dir(cache_dir)
    rows: list[dict[str, Any]] = []

    if _selected(args, "model"):
        for item in cfg.get("models", []):
            repo_id = str(item.get("id", ""))
            local_dir = item.get("local_dir")
            if not repo_id or "/" not in repo_id or not local_dir:
                continue
            rows.append(_download(repo_id, "model", Path(str(local_dir)), cache_dir, args.workers))

    if _selected(args, "dataset"):
        for item in cfg.get("external_benchmarks", []):
            repo_id = str(item.get("id", ""))
            local_dir = item.get("local_dir")
            if not repo_id or "/" not in repo_id or not local_dir:
                continue
            rows.append(_download(repo_id, str(item.get("repo_type", "dataset")), Path(str(local_dir)), cache_dir, args.workers))

    out = ensure_dir(asset_root / "outputs" / "benchmark_assets") / "download_status.json"
    write_json(out, {"assets": rows})
    print(f"Asset download status written to {out}")


if __name__ == "__main__":
    main()
