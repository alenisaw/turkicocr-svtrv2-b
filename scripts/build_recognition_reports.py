#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from turkicocr.reports import write_checkpoint_metrics_report, write_qualitative_report
from turkicocr.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quantitative and qualitative OCR reports.")
    parser.add_argument("--metrics-root", help="Checkpoint monitor metrics root.")
    parser.add_argument("--predictions", help="Prediction JSONL for qualitative examples.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-group", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    if args.metrics_root:
        summary["checkpoint_metrics"] = write_checkpoint_metrics_report(
            args.metrics_root,
            f"{args.out}/quantitative",
        )
    if args.predictions:
        summary["qualitative"] = write_qualitative_report(
            args.predictions,
            f"{args.out}/qualitative",
            per_group=args.per_group,
        )
    write_json(f"{args.out}/report_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
