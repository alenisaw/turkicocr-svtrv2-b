#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from turkicocr.recognition_format import to_openocr_format
from turkicocr.recognition_metrics import cer, rare_char_cer
from turkicocr.utils import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hardcase recognition rows from predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-openocr")
    parser.add_argument("--min-cer", type=float, default=0.10)
    parser.add_argument("--min-rare-cer", type=float, default=0.15)
    return parser.parse_args()


def _hardcase(row: dict[str, Any], min_cer: float, min_rare_cer: float) -> tuple[bool, dict[str, Any]]:
    ref = str(row.get("reference", ""))
    pred = str(row.get("prediction", ""))
    row_cer = cer(ref, pred)
    row_rare = rare_char_cer(ref, pred)
    metadata = dict(row.get("metadata", {}) or {})
    too_short = len(pred) < len(ref) * 0.7
    empty = pred.strip() == ""
    reasons = []
    if row_cer >= min_cer:
        reasons.append("high_cer")
    if row_rare == row_rare and row_rare >= min_rare_cer:
        reasons.append("rare_char_failure")
    if empty:
        reasons.append("empty_prediction")
    elif too_short:
        reasons.append("too_short_prediction")
    layout = str(metadata.get("layout_type", "")).lower()
    if "table" in layout or "form" in layout:
        reasons.append("table_or_form")
    return bool(reasons), {
        "sample_id": str(row.get("sample_id", Path(str(row.get("image_path", ""))).stem)),
        "image_path": str(row.get("image_path", "")),
        "text": ref,
        "language": str(metadata.get("language", "unknown")),
        "layout_type": str(metadata.get("layout_type", "unknown")),
        "degradation_profile": str(metadata.get("degradation_profile", "unknown")),
        "crop_type": str(metadata.get("crop_type", "line")),
        "source_page_id": str(metadata.get("source_page_id", row.get("source_page_id", "unknown"))),
        "text_length_bucket": str(metadata.get("text_length_bucket", "unknown")),
        "hardcase_reasons": reasons,
        "baseline_prediction": pred,
        "baseline_cer": row_cer,
    }


def main() -> None:
    args = parse_args()
    hardcases = []
    total = 0
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            keep, record = _hardcase(row, args.min_cer, args.min_rare_cer)
            if keep:
                hardcases.append(record)
    write_jsonl(args.out_manifest, hardcases)
    if args.out_openocr:
        to_openocr_format(hardcases, args.out_openocr)
    write_json(
        Path(args.out_manifest).with_suffix(".summary.json"),
        {"total": total, "hardcase_count": len(hardcases), "manifest": args.out_manifest},
    )
    print(json.dumps({"total": total, "hardcase_count": len(hardcases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
