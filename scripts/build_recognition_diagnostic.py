#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from turkicocr.rare_chars import contains_rare_char
from turkicocr.recognition_format import load_recognition_manifest, save_recognition_manifest
from turkicocr.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed recognition diagnostic manifests.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-size", type=int, default=256)
    parser.add_argument("--diagnostic-size", type=int, default=256)
    parser.add_argument("--full-size", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("language", "unknown")),
        str(row.get("layout_type", "unknown")),
        str(row.get("degradation_profile", "unknown")),
        str(row.get("crop_type", "unknown")),
        str(row.get("text_length_bucket", "unknown")),
    )


def _sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= size:
        return list(rows)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_key(row)].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
    selected: list[dict[str, Any]] = []
    pointers = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < size:
        added = False
        for key in keys:
            idx = pointers[key]
            if idx < len(buckets[key]):
                selected.append(buckets[key][idx])
                pointers[key] += 1
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
    rng.shuffle(selected)
    return selected


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "language": Counter(),
        "layout_type": Counter(),
        "degradation_profile": Counter(),
        "crop_type": Counter(),
        "text_length_bucket": Counter(),
        "rare_char_presence": Counter(),
    }
    for row in rows:
        for key in ("language", "layout_type", "degradation_profile", "crop_type", "text_length_bucket"):
            counters[key][str(row.get(key, "unknown"))] += 1
        counters["rare_char_presence"][str(contains_rare_char(str(row.get("text", ""))))] += 1
    return {"count": len(rows), **{key: dict(value) for key, value in counters.items()}}


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Recognition Diagnostic Manifest Report", ""]
    for name, section in report.items():
        lines.extend([f"## {name}", "", f"- count: {section['count']}", ""])
        for attr, values in section.items():
            if attr == "count":
                continue
            lines.append(f"### {attr}")
            for key, count in sorted(values.items()):
                lines.append(f"- `{key}`: {count}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "smoke": out_dir / "diagnostic_rec_smoke.jsonl",
        "diagnostic": out_dir / "diagnostic_rec_256.jsonl",
        "full": out_dir / "diagnostic_rec_full.jsonl",
    }
    if not args.force and all(path.exists() for path in paths.values()):
        print("Recognition diagnostic manifests already exist. Use --force to regenerate.")
        return
    rows = load_recognition_manifest(args.validation)
    full = _sample(rows, args.full_size, args.seed)
    diag = _sample(full, args.diagnostic_size, args.seed)
    smoke = _sample(diag, args.smoke_size, args.seed)
    for name, selected in (("full", full), ("diagnostic", diag), ("smoke", smoke)):
        save_recognition_manifest(selected, paths[name])
    report = {"smoke": _coverage(smoke), "diagnostic": _coverage(diag), "full": _coverage(full)}
    write_json(out_dir / "diagnostic_manifest_report.json", report)
    _write_md(out_dir / "diagnostic_manifest_report.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
