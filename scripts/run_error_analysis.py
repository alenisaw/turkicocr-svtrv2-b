#!/usr/bin/env python
from __future__ import annotations

import argparse

from turkicocr.error_analysis import run_error_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OCR error analysis.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_error_analysis(args.predictions, args.out)
    print(f"Error analysis written to {args.out}: {summary}")


if __name__ == "__main__":
    main()
