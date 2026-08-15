from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_layout_aware_sort_two_columns():
    root = Path(__file__).resolve().parents[1]
    prep = load_module(root / "scripts" / "prepare_pipeline_crops.py", "prepare_pipeline_crops_for_test")
    boxes = [
        [520, 140, 900, 160],  # right 2
        [50, 100, 430, 120],   # left 1
        [520, 100, 900, 120],  # right 1
        [50, 140, 430, 160],   # left 2
    ]
    ordered = prep._sort_reading_order_layout_aware(boxes, page_width=1000, page_height=800)
    assert ordered == [
        [50, 100, 430, 120],
        [50, 140, 430, 160],
        [520, 100, 900, 120],
        [520, 140, 900, 160],
    ]


def test_layout_aware_sort_overlapping_paragraph_lines_stay_top_to_bottom():
    # Regression test: henrygagnier__kazakh-ocr_000003 (real detector output).
    # Three sequential full-width paragraph lines whose Y-ranges overlap
    # slightly (a normal detector artifact) previously got chain-merged into
    # one "row" by _cluster_rows_by_y_overlap and sorted by X instead of Y,
    # fully reversing the reading order (bottom line first, top line last).
    root = Path(__file__).resolve().parents[1]
    prep = load_module(root / "scripts" / "prepare_pipeline_crops.py", "prepare_pipeline_crops_for_test")
    top = [57, 60, 1253, 124]
    middle = [54, 100, 1219, 164]
    bottom = [54, 145, 463, 182]
    ordered = prep._sort_reading_order_layout_aware(
        [bottom, top, middle], page_width=1300, page_height=800
    )
    assert ordered == [top, middle, bottom]


def test_page_diagnostics_classifies_order_issue(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "page_predictions.jsonl"
    rows = [
        {
            "page_id": "p1",
            "dataset": "synthetic",
            "sort_mode": "simple",
            "gt_text": "alpha beta gamma delta",
            "prediction": "gamma delta alpha beta",
            "crop_count": 4,
        },
        {
            "page_id": "p2",
            "dataset": "synthetic",
            "sort_mode": "simple",
            "gt_text": "alpha beta gamma delta",
            "prediction": "alpha beta",
            "crop_count": 2,
        },
    ]
    input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out_csv = tmp_path / "diag.csv"
    out_json = tmp_path / "diag.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_page_pipeline_errors.py"),
            "--page-predictions",
            str(input_path),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
        check=True,
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["pages"][0]["error_type"] == "order_issue"
    assert data["pages"][1]["error_type"] == "detection_miss_or_too_short"


def test_oracle_zone_eval_development_echo(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    pages = tmp_path / "pages.jsonl"
    zones = tmp_path / "zones.jsonl"
    cfg = tmp_path / "eval_page_oracle.yaml"
    out_dir = tmp_path / "page_oracle"
    pages.write_text(
        json.dumps({"page_id": "p1", "gt_text": "hello world"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    zones.write_text(
        json.dumps({"page_id": "p1", "text": "hello", "image_path": "unused.png", "order": 0}, ensure_ascii=False) + "\n"
        + json.dumps({"page_id": "p1", "text": "world", "image_path": "unused.png", "order": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cfg.write_text(
        f"""
evaluation:
  name: smoke_oracle
  recognizer_backend: openocr_svtrv2
  checkpoint: dummy.pth
  pages_manifest: {pages}
  zones_manifest: {zones}
  output_dir: {out_dir}
  predictions_dir: {out_dir}
  dataset: smoke
""".strip(),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_oracle_zone_page_eval.py"),
            "--config",
            str(cfg),
            "--development-echo",
        ],
        cwd=root,
        check=True,
    )
    metrics = json.loads((out_dir / "metrics_aggregate.json").read_text(encoding="utf-8"))
    assert metrics["cer"] == 0
    assert metrics["page_count"] == 1
