#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import shutil
import zipfile
from pathlib import Path
from turkicocr.utils import get_asset_root

EXCLUDE_PATTERNS = (
    "*.pth",
    "*.onnx",
    "*.onnx.data",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "__pycache__/*",
)

SELECTED_EXTERNAL_BENCHMARKS = {
    "henrygagnier__kazakh-ocr",
    "alenisaw__turkicocr-cyrillic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight release ZIP in the repo root.")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--name", default="turkicocr_svtrv2_b_epoch9_release_20260705")
    parser.add_argument("--recognition-release-name", default="turkicocr_svtrv2_b_epoch9_release_20260705")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _excluded(path: Path) -> bool:
    text = str(path)
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(text, pattern) for pattern in EXCLUDE_PATTERNS)


def _copy_tree_files(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for item in src.rglob("*"):
        if not item.is_file() or _excluded(item):
            continue
        if item.stem.startswith(("smoke", "tmp", "debug")):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _copy_selected_tree_files(src: Path, dst: Path, selected: set[str], copy_root_summaries: bool = True) -> None:
    if not src.exists():
        return
    for name in selected:
        _copy_tree_files(src / name, dst / name)
    if copy_root_summaries:
        for filename in ("leaderboard.csv", "combined_leaderboard.csv", "benchmark_suite_summary.json", "pipeline_eval_summary.json"):
            _copy_if_exists(src / filename, dst / filename)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file() and not _excluded(src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_external_manifests(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for item in src.glob("*.jsonl"):
        if not item.is_file() or item.stem.startswith(("smoke", "debug")):
            continue
        name = item.stem.removeprefix("tmp_")
        if name not in SELECTED_EXTERNAL_BENCHMARKS:
            continue
        target = dst / f"{name}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    zip_path = Path(args.out).resolve() if args.out else repo_root / f"{args.name}.zip"
    stage_root = repo_root / ".release_staging" / args.name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    for name in (
        "README.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        "requirements.txt",
        "requirements-inference.txt",
        "pyproject.toml",
    ):
        _copy_if_exists(repo_root / name, stage_root / name)

    for folder in ("configs", "docs", "metrics"):
        _copy_tree_files(repo_root / folder, stage_root / folder)

    recognition_release = asset_root / "outputs" / "recognition" / "release_package" / args.recognition_release_name
    for folder in ("metrics", "reports", "figures", "qualitative"):
        _copy_tree_files(recognition_release / folder, stage_root / folder)
    _copy_if_exists(recognition_release / "manifest_line_counts.txt", stage_root / "manifest_line_counts.txt")
    _copy_if_exists(recognition_release / "reproducibility_hashes.sha256", stage_root / "reproducibility_hashes.sha256")

    external_manifests = asset_root / "outputs" / "external_benchmarks" / "manifests"
    _copy_external_manifests(external_manifests, stage_root / "external_benchmarks" / "manifests")

    asset_status = asset_root / "outputs" / "benchmark_assets" / "download_status.json"
    _copy_if_exists(asset_status, stage_root / "benchmark_assets" / "download_status.json")

    pipeline_eval = asset_root / "outputs" / "pipeline_eval"
    _copy_selected_tree_files(pipeline_eval, stage_root / "pipeline_eval", SELECTED_EXTERNAL_BENCHMARKS)

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(stage_root.rglob("*")):
            if item.is_file() and not _excluded(item):
                archive.write(item, item.relative_to(stage_root.parent))

    shutil.rmtree(stage_root.parent)
    print(f"Created lightweight release ZIP at {zip_path}")


if __name__ == "__main__":
    main()
