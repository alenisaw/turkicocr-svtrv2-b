from __future__ import annotations

from pathlib import Path


def _bar_plot(
    labels: list[str], values: list[float], title: str, ylabel: str, out_path: str | Path
) -> None:
    pass


def make_metric_plots(metrics_root: str | Path, out_dir: str | Path) -> list[str]:
    return []


def make_training_plots(metrics_jsonl: str | Path, out_dir: str | Path) -> list[str]:
    return []


def plot_from_csv(
    csv_path: str | Path, x_col: str, y_col: str, out_path: str | Path, title: str
) -> None:
    pass
