#!/usr/bin/env python
"""Oracle-zone page evaluation for TurkicOCR.

Purpose:
  Measure page-level reconstruction quality without detector errors.

The script supports two zone-manifest modes:
  1. pre-cropped zones/lines with `image_path` and `text`;
  2. page image + `bbox` zones, where crops are created on the fly.

Use `--development-echo` for smoke tests; it returns zone reference text without
loading the recognizer.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

try:
    from scripts.analyze_page_pipeline_errors import analyze_row, cer, chrf_score, wer
except Exception:  # pragma: no cover - allows execution from repo root or copied file.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_page_pipeline_errors import analyze_row, cer, chrf_score, wer


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split()).strip()


def page_id_from_row(row: dict[str, Any], idx: int) -> str:
    if row.get("page_id"):
        return str(row["page_id"])
    if row.get("source_page_id"):
        return str(row["source_page_id"])
    meta = row.get("metadata", {}) or {}
    if meta.get("page_id"):
        return str(meta["page_id"])
    src = meta.get("source_index", row.get("source_index", idx))
    bench = str(meta.get("benchmark", row.get("benchmark", "turkicocr-cyrillic"))).replace("/", "__")
    return f"{bench}_{int(src):06d}" if isinstance(src, int) or str(src).isdigit() else f"{bench}_{src}"


def page_gt_text(row: dict[str, Any]) -> str:
    if row.get("gt_text"):
        return normalize_text(str(row["gt_text"]))
    if row.get("text"):
        return normalize_text(str(row["text"]))
    parts = [str(t.get("text", "")) for t in row.get("text_info", []) if t.get("tag") != "mask"]
    return normalize_text(" ".join(parts))


def zone_page_id(row: dict[str, Any], idx: int) -> str:
    for key in ("page_id", "source_page_id"):
        if row.get(key):
            return str(row[key])
    meta = row.get("metadata", {}) or {}
    for key in ("page_id", "source_page_id"):
        if meta.get(key):
            return str(meta[key])
    return page_id_from_row(row, idx)


def zone_order_key(row: dict[str, Any], idx: int) -> tuple[float, float, int]:
    if row.get("order") is not None:
        return (float(row["order"]), 0.0, idx)
    if row.get("reading_order") is not None:
        return (float(row["reading_order"]), 0.0, idx)
    bbox = row.get("bbox") or (row.get("metadata", {}) or {}).get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return ((y1 + y2) / 2.0, (x1 + x2) / 2.0, idx)
    return (float(idx), 0.0, idx)


def zone_text(row: dict[str, Any]) -> str:
    return normalize_text(str(row.get("text") or row.get("gt_text") or row.get("reference") or ""))


def zone_image_path(row: dict[str, Any]) -> str | None:
    if row.get("image_path"):
        return str(row["image_path"])
    if row.get("image_url"):
        return str(row["image_url"])
    image_info = row.get("image_info") or []
    if image_info and isinstance(image_info[0], dict):
        return str(image_info[0].get("image_url") or image_info[0].get("image_path") or "") or None
    return None


def crop_zone_from_page(row: dict[str, Any], pages_by_id: dict[str, dict[str, Any]], out_dir: Path, idx: int) -> str:
    pid = zone_page_id(row, idx)
    page = pages_by_id.get(pid)
    if not page:
        raise ValueError(f"Cannot find page for zone {idx}: {pid}")
    page_image = zone_image_path(page)
    if not page_image:
        raise ValueError(f"Page {pid} has no image path")
    bbox = row.get("bbox") or (row.get("metadata", {}) or {}).get("bbox")
    if not (isinstance(bbox, list) and len(bbox) >= 4):
        raise ValueError(f"Zone {idx} has no bbox and no image_path")
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(page_image) as img:
        w, h = img.size
        crop = img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        crop_path = out_dir / f"{pid}_oracle_{idx:06d}.png"
        crop.save(crop_path)
        return str(crop_path)


def create_backend(backend_name: str, checkpoint: str, device: str | None):
    from turkicocr.inference import create_backend as _create_backend
    return _create_backend(backend_name=backend_name, checkpoint=checkpoint, device=device)


def predict_zone(backend: Any, image_path: str, reference: str, development_echo: bool) -> str:
    if development_echo:
        return reference
    return str(backend.predict(image_url=image_path, prompt=""))


def aggregate_metrics(page_rows: list[dict[str, Any]], model_id: str, dataset: str, run_id: str) -> dict[str, Any]:
    refs = [str(r["gt_text"]) for r in page_rows]
    hyps = [str(r["prediction"]) for r in page_rows]
    return {
        "run_id": run_id,
        "model_id": model_id,
        "eval_level": "page_oracle",
        "dataset": dataset,
        "split": "oracle_pages",
        "sample_count": len(page_rows),
        "page_count": len(page_rows),
        "crop_count": sum(int(r.get("zone_count") or 0) for r in page_rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cer": sum(cer(r, h) for r, h in zip(refs, hyps, strict=False)) / max(1, len(refs)),
        "wer": sum(wer(r, h) for r, h in zip(refs, hyps, strict=False)) / max(1, len(refs)),
        "chrf": sum(chrf_score(r, h) for r, h in zip(refs, hyps, strict=False)) / max(1, len(refs)),
        "rare_char_cer": None,
        "exact_match": sum(1 for r, h in zip(refs, hyps, strict=False) if normalize_text(r) == normalize_text(h)) / max(1, len(refs)),
        "length_ratio": sum(len(normalize_text(h)) / max(1, len(normalize_text(r))) for r, h in zip(refs, hyps, strict=False)) / max(1, len(refs)),
        "empty_prediction_rate": sum(1 for h in hyps if not normalize_text(h)) / max(1, len(hyps)),
        "failed_zone_rate": sum(int(r.get("failed_zone_count") or 0) for r in page_rows) / max(1, sum(int(r.get("zone_count") or 0) for r in page_rows)),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle-zone page reconstruction evaluation.")
    parser.add_argument("--config", default="configs/eval_page_oracle.yaml")
    parser.add_argument("--pages-manifest", default=None)
    parser.add_argument("--zones-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--model-id", default="turkicocr_svtrv2_b_epoch9")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--development-echo", action="store_true")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--predictions-jsonl",
        default=None,
        help=(
            "Reuse an existing per-crop predictions file (sample_id -> prediction) instead of "
            "re-running the recognizer. Use this when the zones manifest is the same crop set "
            "already evaluated for the canonical line-recognition metric."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if Path(args.config).exists() else {}
    eval_cfg = cfg.get("evaluation", cfg)

    pages_manifest = Path(args.pages_manifest or eval_cfg.get("pages_manifest"))
    zones_manifest = Path(args.zones_manifest or eval_cfg.get("zones_manifest"))
    output_dir = Path(args.output_dir or eval_cfg.get("output_dir") or "metrics/page_oracle")
    predictions_dir = Path(args.predictions_dir or eval_cfg.get("predictions_dir") or output_dir)
    checkpoint = args.checkpoint or eval_cfg.get("checkpoint") or ""
    backend_name = args.backend or eval_cfg.get("recognizer_backend") or "openocr_svtrv2"
    run_id = str(eval_cfg.get("name") or "turkicocr_oracle_zone_page_eval")

    pages = list(iter_jsonl(pages_manifest))
    pages_by_id = {page_id_from_row(row, idx): row for idx, row in enumerate(pages)}
    zones_by_page: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, zone in enumerate(iter_jsonl(zones_manifest)):
        zones_by_page[zone_page_id(zone, idx)].append((idx, zone))

    reuse_predictions: dict[str, str] | None = None
    if args.predictions_jsonl:
        reuse_predictions = {}
        for row in iter_jsonl(Path(args.predictions_jsonl)):
            sid = row.get("sample_id") or row.get("id")
            if sid:
                reuse_predictions[str(sid)] = str(row.get("prediction") or "")

    backend = (
        None
        if args.development_echo or reuse_predictions is not None
        else create_backend(backend_name, checkpoint, args.device)
    )
    page_rows: list[dict[str, Any]] = []
    crop_dir = predictions_dir / "oracle_crops"

    processed_pages = 0
    for page_idx, page in enumerate(pages):
        if args.max_pages is not None and processed_pages >= args.max_pages:
            break
        pid = page_id_from_row(page, page_idx)
        page_zones = sorted(zones_by_page.get(pid, []), key=lambda item: zone_order_key(item[1], item[0]))
        if not page_zones:
            continue

        refs: list[str] = []
        preds: list[str] = []
        failed = 0
        for zone_idx, zone in page_zones:
            ref = zone_text(zone)
            refs.append(ref)
            try:
                if reuse_predictions is not None:
                    sid = str(zone.get("sample_id") or "")
                    if sid not in reuse_predictions:
                        raise KeyError(f"no reused prediction for sample_id={sid!r}")
                    preds.append(reuse_predictions[sid])
                    continue
                img_path = zone_image_path(zone) or crop_zone_from_page(zone, pages_by_id, crop_dir, zone_idx)
                preds.append(predict_zone(backend, img_path, ref, args.development_echo))
            except Exception as exc:
                failed += 1
                preds.append("")
                print(f"WARNING: zone failed page={pid} zone_idx={zone_idx}: {exc}", flush=True)

        gt_page = page_gt_text(page) or normalize_text("\n".join(refs))
        pred_page = normalize_text("\n".join(preds))
        row = {
            "page_id": pid,
            "dataset": eval_cfg.get("dataset", "turkicocr-cyrillic"),
            "eval_level": "page_oracle",
            "gt_text": gt_page,
            "prediction": pred_page,
            "zone_count": len(page_zones),
            "failed_zone_count": failed,
            "length_ratio": len(pred_page) / max(1, len(normalize_text(gt_page))),
            "cer": cer(gt_page, pred_page),
            "wer": wer(gt_page, pred_page),
            "chrf": chrf_score(gt_page, pred_page),
        }
        page_rows.append(row)
        processed_pages += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "page_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in page_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = aggregate_metrics(
        page_rows,
        model_id=args.model_id,
        dataset=str(eval_cfg.get("dataset", "turkicocr-cyrillic")),
        run_id=run_id,
    )
    metrics["script"] = "scripts/run_oracle_zone_page_eval.py"
    metrics["config"] = str(args.config)
    metrics["checkpoint"] = checkpoint
    metrics["backend"] = backend_name
    metrics["development_echo"] = args.development_echo

    write_json(output_dir / "metrics_aggregate.json", metrics)
    write_csv(output_dir / "metrics_aggregate.csv", [metrics])
    diagnostics = [analyze_row(row, dataset=row.get("dataset"), sort_mode="oracle") for row in page_rows]
    write_csv(output_dir / "page_diagnostics.csv", diagnostics)

    report_path = Path(".agent/reports/ORACLE_PAGE_EVAL.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Oracle Page Evaluation\n\n"
        f"Pages: {metrics['page_count']}\n\n"
        f"CER: {float(metrics['cer']) * 100:.2f}%\n\n"
        f"WER: {float(metrics['wer']) * 100:.2f}%\n\n"
        f"chrF: {float(metrics['chrf']):.4f}\n\n"
        f"Failed-zone rate: {float(metrics['failed_zone_rate']) * 100:.2f}%\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
