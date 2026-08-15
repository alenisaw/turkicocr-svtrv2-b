#!/usr/bin/env python
from __future__ import annotations

import argparse

from turkicocr.plotting import make_metric_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate plots from metrics and error-analysis outputs."
    )
    p.add_argument("--metrics", required=True)
    p.add_argument("--error-analysis", default=None)
    p.add_argument("--external", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generated = make_metric_plots(args.metrics, args.out)
    print(f"Generated {len(generated)} plot(s) under {args.out}")


if __name__ == "__main__":
    main()
