#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from turkicocr.recognition_metrics import (
    evaluate_recognition_by_slice,
    evaluate_recognition_gate,
    evaluate_recognition_predictions,
)
from turkicocr.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recognition prediction JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--charset")
    parser.add_argument("--gate-config", default="configs/recognition_gate.yaml")
    parser.add_argument("--allow-fail", action="store_true", help="Write metrics but return 0 even when the gate fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    refs = [str(row.get("reference", row.get("text", ""))) for row in rows]
    preds = [str(row.get("prediction", "")) for row in rows]
    metadata = [dict(row.get("metadata", {}) or {}) for row in rows]
    metrics = evaluate_recognition_predictions(refs, preds, metadata, args.charset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics_aggregate.json", metrics)
    for key, filename in (
        ("language", "metrics_by_language.json"),
        ("degradation_profile", "by_degradation.json"),
        ("layout_type", "by_layout.json"),
        ("crop_type", "by_crop_type.json"),
    ):
        write_json(out / filename, evaluate_recognition_by_slice(rows, key))
    thresholds = {}
    if Path(args.gate_config).exists():
        cfg = yaml.safe_load(Path(args.gate_config).read_text(encoding="utf-8")) or {}
        thresholds = cfg.get("recognition_gate", cfg)
    gate = evaluate_recognition_gate(metrics, thresholds)
    write_json(out / "recognition_gate_result.json", gate)
    print(json.dumps({"metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2))
    if gate["status"] in {"FAIL", "SMOKE_FAIL"} and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
