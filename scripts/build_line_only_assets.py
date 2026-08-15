#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from turkicocr.recognition_format import (
    load_recognition_manifest,
    save_recognition_manifest,
    to_openocr_format,
)
from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build line-only manifests and OpenOCR label files.")
    parser.add_argument("--manifest-dir", default=str(get_asset_root() / "outputs/recognition/manifests"))
    parser.add_argument("--suffix", default="line_only")
    parser.add_argument("--crop-type", action="append", default=["line"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic-size", type=int, default=256)
    parser.add_argument("--full-size", type=int, default=5000)
    return parser.parse_args()


def _filter(rows: list[dict[str, Any]], crop_types: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("crop_type")) in crop_types]


def _sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= size:
        return list(rows)
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    return selected[:size]


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "crop_type": dict(Counter(str(row.get("crop_type", "unknown")) for row in rows)),
        "language": dict(Counter(str(row.get("language", "unknown")) for row in rows)),
        "text_length_bucket": dict(Counter(str(row.get("text_length_bucket", "unknown")) for row in rows)),
    }


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    crop_types = set(args.crop_type)
    report: dict[str, Any] = {"crop_types": sorted(crop_types), "splits": {}}

    split_inputs = {
        "train": manifest_dir / "train_rec_large.jsonl",
        "val": manifest_dir / "val_rec_large.jsonl",
        "test": manifest_dir / "test_rec_large.jsonl",
    }
    filtered_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, source in split_inputs.items():
        rows = load_recognition_manifest(source)
        filtered = _filter(rows, crop_types)
        filtered_by_split[split] = filtered
        manifest_out = manifest_dir / f"{split}_rec_large_{args.suffix}.jsonl"
        save_recognition_manifest(filtered, manifest_out)
        if split in {"train", "val", "test"}:
            openocr_name = "val" if split == "val" else split
            to_openocr_format(filtered, manifest_dir / f"{openocr_name}_openocr_{args.suffix}.txt")
        report["splits"][split] = {
            "source": str(source),
            "manifest": str(manifest_out),
            "coverage": _coverage(filtered),
        }

    val_rows = filtered_by_split["val"]
    full = _sample(val_rows, args.full_size, args.seed)
    diagnostic = _sample(full, args.diagnostic_size, args.seed)
    diagnostic_paths = {
        "diagnostic": manifest_dir / f"diagnostic_rec_256_{args.suffix}.jsonl",
        "full": manifest_dir / f"diagnostic_rec_full_{args.suffix}.jsonl",
    }
    save_recognition_manifest(diagnostic, diagnostic_paths["diagnostic"])
    save_recognition_manifest(full, diagnostic_paths["full"])
    report["diagnostics"] = {
        "diagnostic": {"path": str(diagnostic_paths["diagnostic"]), "coverage": _coverage(diagnostic)},
        "full": {"path": str(diagnostic_paths["full"]), "coverage": _coverage(full)},
    }
    out = manifest_dir / f"{args.suffix}_asset_report.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
