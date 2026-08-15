from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .recognition_metrics import evaluate_recognition_predictions


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_structural_reference(text: str) -> bool:
    return bool(re.fullmatch(r"\[[^\]]+\]", text.strip()))


def _bbox(zone: dict[str, Any]) -> tuple[float, float, float, float]:
    box = zone.get("bbox") or zone.get("box")
    if box is not None and len(box) == 4:
        return tuple(float(v) for v in box)  # type: ignore[return-value]
    x = float(zone.get("x", zone.get("left", 0.0)))
    y = float(zone.get("y", zone.get("top", 0.0)))
    w = float(zone.get("width", zone.get("w", 0.0)))
    h = float(zone.get("height", zone.get("h", 0.0)))
    return x, y, x + w, y + h


def sort_zones_geometric(zones: list[dict[str, Any]], y_tolerance: float = 12.0) -> list[dict[str, Any]]:
    decorated = []
    for zone in zones:
        x1, y1, x2, y2 = _bbox(zone)
        height = max(1.0, y2 - y1)
        line_key = round(y1 / max(y_tolerance, height * 0.5))
        decorated.append((line_key, y1, x1, zone))
    return [zone for _, _, _, zone in sorted(decorated, key=lambda item: (item[0], item[1], item[2]))]


def bbox_overlap_ratio(inner: dict[str, Any], outer: dict[str, Any]) -> float:
    ix1, iy1, ix2, iy2 = _bbox(inner)
    ox1, oy1, ox2, oy2 = _bbox(outer)
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / inner_area


def assign_lines_to_zone(
    zone: dict[str, Any],
    lines: list[dict[str, Any]],
    min_overlap: float = 0.50,
) -> list[dict[str, Any]]:
    source_page_id = str(zone.get("source_page_id", zone.get("page_id", "")))
    candidates = []
    for line in lines:
        if str(line.get("source_page_id", line.get("page_id", ""))) != source_page_id:
            continue
        overlap = bbox_overlap_ratio(line, zone)
        if overlap >= min_overlap:
            row = dict(line)
            row["zone_overlap_ratio"] = overlap
            candidates.append(row)
    return sort_zones_geometric(candidates)


def reconstruct_zone_from_lines(
    zone: dict[str, Any],
    lines: list[dict[str, Any]],
    line_predictions: dict[str, str],
    min_overlap: float = 0.50,
    separator: str = "\n",
) -> dict[str, Any]:
    assigned = assign_lines_to_zone(zone, lines, min_overlap=min_overlap)
    parts: list[str] = []
    line_ids: list[str] = []
    for line in assigned:
        sample_id = str(line.get("sample_id", line.get("id", "")))
        text = " ".join(str(line_predictions.get(sample_id, "")).split()).strip()
        if text:
            parts.append(text)
        line_ids.append(sample_id)
    return {
        "zone_id": str(zone.get("sample_id", zone.get("zone_id", zone.get("id", "")))),
        "source_page_id": str(zone.get("source_page_id", zone.get("page_id", ""))),
        "reference": str(zone.get("text", zone.get("reference", ""))),
        "prediction": separator.join(parts),
        "line_ids": line_ids,
        "line_count": len(assigned),
    }


def _assign_cached_lines_to_zone(
    zone: dict[str, Any],
    lines: list[dict[str, Any]],
    min_overlap: float = 0.50,
) -> list[dict[str, Any]]:
    ox1, oy1, ox2, oy2 = _bbox(zone)
    candidates = []
    for line in lines:
        ix1, iy1, ix2, iy2 = line.get("_bbox_tuple", _bbox(line))
        if iy2 <= oy1 or iy1 >= oy2 or ix2 <= ox1 or ix1 >= ox2:
            continue
        inter_x1 = max(ix1, ox1)
        inter_y1 = max(iy1, oy1)
        inter_x2 = min(ix2, ox2)
        inter_y2 = min(iy2, oy2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
        overlap = inter_area / inner_area
        if overlap >= min_overlap:
            row = dict(line)
            row.pop("_bbox_tuple", None)
            row["zone_overlap_ratio"] = overlap
            candidates.append(row)
    return sort_zones_geometric(candidates)


def _reconstruct_zone_from_cached_lines(
    zone: dict[str, Any],
    lines: list[dict[str, Any]],
    line_predictions: dict[str, str],
    min_overlap: float = 0.50,
    separator: str = "\n",
) -> dict[str, Any]:
    assigned = _assign_cached_lines_to_zone(zone, lines, min_overlap=min_overlap)
    parts: list[str] = []
    line_ids: list[str] = []
    for line in assigned:
        sample_id = str(line.get("sample_id", line.get("id", "")))
        text = " ".join(str(line_predictions.get(sample_id, "")).split()).strip()
        if text:
            parts.append(text)
        line_ids.append(sample_id)
    return {
        "zone_id": str(zone.get("sample_id", zone.get("zone_id", zone.get("id", "")))),
        "source_page_id": str(zone.get("source_page_id", zone.get("page_id", ""))),
        "reference": str(zone.get("text", zone.get("reference", ""))),
        "prediction": separator.join(parts),
        "line_ids": line_ids,
        "line_count": len(assigned),
    }


def reconstruct_page_text(
    zones: list[dict[str, Any]],
    predictions: dict[str, str] | list[str],
    separator: str = "\n",
) -> str:
    ordered = sort_zones_geometric(zones)
    lines: list[str] = []
    for idx, zone in enumerate(ordered):
        if isinstance(predictions, dict):
            key = str(zone.get("zone_id", zone.get("sample_id", zone.get("id", idx))))
            text = predictions.get(key, "")
        else:
            text = predictions[idx] if idx < len(predictions) else ""
        text = " ".join(str(text).split()).strip()
        if text:
            lines.append(text)
    return separator.join(lines)


def oracle_zone_page_eval(
    pages_manifest: str | Path,
    zones_manifest: str | Path,
    recognizer_fn: Callable[[dict[str, Any]], str],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = _ensure_dir(output_dir)
    pages = _read_jsonl(pages_manifest)
    zones = _read_jsonl(zones_manifest)
    zones_by_page: dict[str, list[dict[str, Any]]] = {}
    for zone in zones:
        zones_by_page.setdefault(str(zone.get("source_page_id", zone.get("page_id", ""))), []).append(zone)

    prediction_rows: list[dict[str, Any]] = []
    refs: list[str] = []
    hyps: list[str] = []
    for page in pages:
        page_id = str(page.get("page_id", page.get("source_page_id", page.get("sample_id", ""))))
        page_zones = zones_by_page.get(page_id, [])
        predictions = {
            str(zone.get("zone_id", zone.get("sample_id", idx))): recognizer_fn(zone)
            for idx, zone in enumerate(sort_zones_geometric(page_zones))
        }
        text = reconstruct_page_text(page_zones, predictions)
        reference = str(page.get("text", page.get("reference", "")))
        refs.append(reference)
        hyps.append(text)
        prediction_rows.append({"page_id": page_id, "reference": reference, "prediction": text})

    metrics = evaluate_recognition_predictions(refs, hyps)
    _write_jsonl(out / "page_predictions.jsonl", prediction_rows)
    _write_json(out / "page_metrics.json", metrics)
    return metrics


def oracle_zone_from_lines_eval(
    zones_manifest: str | Path,
    lines_manifest: str | Path,
    line_predictions: dict[str, str],
    output_dir: str | Path,
    min_overlap: float = 0.50,
) -> dict[str, Any]:
    out = _ensure_dir(output_dir)
    raw_zones = [
        row for row in _read_jsonl(zones_manifest)
        if str(row.get("crop_type", "zone")) in {"zone", "table_cell", "field"}
    ]
    zones = [
        row for row in raw_zones
        if not _is_structural_reference(str(row.get("text", row.get("reference", ""))))
    ]
    lines = [
        row for row in _read_jsonl(lines_manifest)
        if str(row.get("crop_type", "")) == "line"
    ]
    lines_by_page: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        page_id = str(line.get("source_page_id", line.get("page_id", "")))
        line["_bbox_tuple"] = _bbox(line)
        lines_by_page.setdefault(page_id, []).append(line)
    rows: list[dict[str, Any]] = []
    refs: list[str] = []
    hyps: list[str] = []
    metadata: list[dict[str, Any]] = []
    for zone in zones:
        page_id = str(zone.get("source_page_id", zone.get("page_id", "")))
        reconstructed = _reconstruct_zone_from_cached_lines(
            zone,
            lines_by_page.get(page_id, []),
            line_predictions,
            min_overlap=min_overlap,
        )
        meta = {
            "language": zone.get("language", "unknown"),
            "layout_type": zone.get("layout_type", "unknown"),
            "degradation_profile": zone.get("degradation_profile", "unknown"),
            "crop_type": "zone_from_lines",
            "text_length_bucket": zone.get("text_length_bucket", "unknown"),
            "line_count": reconstructed["line_count"],
        }
        row = {
            "sample_id": reconstructed["zone_id"],
            "source_page_id": reconstructed["source_page_id"],
            "reference": reconstructed["reference"],
            "prediction": reconstructed["prediction"],
            "line_ids": reconstructed["line_ids"],
            "metadata": meta,
        }
        rows.append(row)
        refs.append(row["reference"])
        hyps.append(row["prediction"])
        metadata.append(meta)

    metrics = evaluate_recognition_predictions(refs, hyps, metadata)
    metrics["val_rec_structural_zone_skipped_count"] = float(len(raw_zones) - len(zones))
    _write_jsonl(out / "zone_from_lines_predictions.jsonl", rows)
    _write_json(out / "zone_from_lines_metrics.json", metrics)
    return metrics


def detected_zone_page_eval(
    pages_manifest: str | Path,
    detector_fn: Callable[[dict[str, Any]], list[dict[str, Any]]],
    recognizer_fn: Callable[[dict[str, Any]], str],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = _ensure_dir(output_dir)
    pages = _read_jsonl(pages_manifest)
    prediction_rows: list[dict[str, Any]] = []
    refs: list[str] = []
    hyps: list[str] = []
    for page in pages:
        zones = detector_fn(page)
        predictions = {
            str(zone.get("zone_id", zone.get("sample_id", idx))): recognizer_fn(zone)
            for idx, zone in enumerate(sort_zones_geometric(zones))
        }
        text = reconstruct_page_text(zones, predictions)
        reference = str(page.get("text", page.get("reference", "")))
        refs.append(reference)
        hyps.append(text)
        prediction_rows.append(
            {
                "page_id": page.get("page_id", page.get("sample_id")),
                "reference": reference,
                "prediction": text,
                "detected_zone_count": len(zones),
            }
        )
    metrics = evaluate_recognition_predictions(refs, hyps)
    _write_jsonl(out / "detected_page_predictions.jsonl", prediction_rows)
    _write_json(out / "detected_page_metrics.json", metrics)
    return metrics
