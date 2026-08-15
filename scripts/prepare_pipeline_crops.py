#!/usr/bin/env python
"""
prepare_pipeline_crops.py
=========================
Universal script that converts a page-level external benchmark manifest into
line-level crops by running text-line detection (OpenOCR DBNet ONNX).

Outputs per dataset:
  1. <out_dir>/line_manifest.jsonl   – one entry per detected line crop.
  2. <out_dir>/page_groups.jsonl     – one entry per source page with page GT,
                                       ordered crop ids, metadata and sort_mode.

This patched version adds:
  * --sort-mode simple|layout_aware
  * layout-aware reading order for multi-column/table/form pages
  * sort_mode propagation into line/page records and summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # Allows syntax/self-check tests without the runtime installed.
    ort = None  # type: ignore[assignment]
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from turkicocr.utils import get_asset_root

_ASSET_ROOT = get_asset_root()
_DEFAULT_DET_MODEL = str(_ASSET_ROOT / "models/OpenOCR/SVTRv2-B/openocr_det_model.onnx")
_EXTERNAL_MANIFEST_ROOT = str(_ASSET_ROOT / "outputs/external_benchmarks/manifests")
_PIPELINE_CROPS_ROOT = str(_ASSET_ROOT / "outputs/pipeline_crops")
_VALID_SORT_MODES = {"simple", "layout_aware"}

_ALL_DATASETS = [
    "henrygagnier__kazakh-ocr",
    "alenisaw__turkicocr-cyrillic",
]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _resize_for_det(image: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float, float]:
    """Resize image so longest side <= max_side and dims are multiples of 32."""
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    nh = max(32, int(h * scale / 32) * 32)
    nw = max(32, int(w * scale / 32) * 32)
    scale_h = nh / h
    scale_w = nw / w
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, scale_h, scale_w


def _preprocess_det(image_bgr: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float, float]:
    resized, sh, sw = _resize_for_det(image_bgr, max_side)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _DET_MEAN) / _DET_STD
    return normalized.transpose(2, 0, 1)[None], sh, sw  # [1,3,H,W]


def _postprocess_det(
    prob_map: np.ndarray,
    orig_h: int,
    orig_w: int,
    scale_h: float,
    scale_w: float,
    prob_thresh: float = 0.25,
    box_thresh: float = 0.2,
    min_h: int = 6,
    min_w: int = 8,
    padding: int = 4,
) -> list[list[int]]:
    """Convert DBNet probability map into axis-aligned boxes [x1, y1, x2, y2]."""
    prob = prob_map[0, 0]
    binary = (prob > prob_thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[list[int]] = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        x1 = max(0, int(bx / scale_w) - padding)
        y1 = max(0, int(by / scale_h) - padding)
        x2 = min(orig_w, int((bx + bw) / scale_w) + padding)
        y2 = min(orig_h, int((by + bh) / scale_h) + padding)
        if (x2 - x1) < min_w or (y2 - y1) < min_h:
            continue

        bx2_det = min(prob.shape[1], bx + bw)
        by2_det = min(prob.shape[0], by + bh)
        region = prob[by:by2_det, bx:bx2_det]
        if region.size == 0 or region.mean() < box_thresh:
            continue
        boxes.append([x1, y1, x2, y2])

    return boxes


# ---------------------------------------------------------------------------
# Reading-order sorting
# ---------------------------------------------------------------------------
def _box_center(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_width(box: list[int]) -> int:
    return max(1, box[2] - box[0])


def _box_height(box: list[int]) -> int:
    return max(1, box[3] - box[1])


def _box_y_overlap_ratio(a: list[int], b: list[int]) -> float:
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    overlap = max(0, bottom - top)
    return overlap / max(1, min(_box_height(a), _box_height(b)))


def _sort_reading_order_simple(boxes: list[list[int]], line_tol_ratio: float = 0.6) -> list[list[int]]:
    """Old behavior: rough top-to-bottom, left-to-right sorting."""
    if not boxes:
        return boxes
    decorated = []
    for box in boxes:
        x1, y1, x2, y2 = box
        h = max(1, y2 - y1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        row_key = round(cy / (h * line_tol_ratio))
        decorated.append((row_key, cy, cx, box))
    decorated.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in decorated]


def _box_x_overlap_ratio(a: list[int], b: list[int]) -> float:
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    overlap = max(0, right - left)
    return overlap / max(1, min(_box_width(a), _box_width(b)))


def _cluster_rows_by_y_overlap(
    boxes: list[list[int]],
    min_y_overlap: float = 0.5,
    max_x_overlap: float = 0.3,
) -> list[list[list[int]]]:
    """Group boxes into true reading-order "rows": items that sit side by
    side on (approximately) the same text line.

    A box joins a row only if it is compatible with EVERY box already in
    that row (no growing "representative bbox", so there is no transitive
    chaining), where "compatible" means high Y-overlap (vertically aligned)
    AND low X-overlap (horizontally disjoint -- i.e. actually side by side,
    not stacked). This keeps sequential paragraph lines (which share nearly
    the same X range, and so are barred by the X-overlap check even when
    their Y ranges happen to overlap) from ever being merged and sorted by
    the wrong axis, while still merging genuine table cells / side-by-side
    blocks (low X-overlap, high Y-overlap by construction).
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: (_box_center(b)[1], _box_center(b)[0]))
    rows: list[list[list[int]]] = []

    for box in sorted_boxes:
        placed = False
        for row in rows:
            compatible = True
            for member in row:
                y_overlap = _box_y_overlap_ratio(box, member)
                x_overlap = _box_x_overlap_ratio(box, member)
                if y_overlap < min_y_overlap or x_overlap > max_x_overlap:
                    compatible = False
                    break
            if compatible:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    for row in rows:
        row.sort(key=lambda b: (_box_center(b)[0], b[0]))
    rows.sort(key=lambda row: median(_box_center(b)[1] for b in row))
    return rows


def _flatten_rows(rows: list[list[list[int]]]) -> list[list[int]]:
    return [box for row in rows for box in row]


def _estimate_column_groups(
    boxes: list[list[int]],
    page_width: int,
    min_boxes: int = 4,
) -> list[list[list[int]]]:
    """Split boxes into likely columns. Returns one group if no clear column gap exists."""
    if len(boxes) < min_boxes or page_width <= 0:
        return [boxes]

    by_x = sorted(boxes, key=lambda b: _box_center(b)[0])
    centers = [_box_center(b)[0] for b in by_x]
    widths = [_box_width(b) for b in by_x]
    med_w = median(widths) if widths else 1

    gaps: list[tuple[float, int]] = []
    for idx in range(len(centers) - 1):
        gaps.append((centers[idx + 1] - centers[idx], idx))
    if not gaps:
        return [boxes]

    best_gap, split_idx = max(gaps, key=lambda item: item[0])
    left = by_x[: split_idx + 1]
    right = by_x[split_idx + 1 :]

    min_side = max(2, int(0.20 * len(boxes)))
    clear_gap = best_gap >= max(0.15 * page_width, 0.75 * med_w)
    balanced_enough = len(left) >= min_side and len(right) >= min_side

    if clear_gap and balanced_enough:
        return [left, right]
    return [boxes]


def _looks_like_table_or_form(boxes: list[list[int]], page_width: int, layout_type: str | None = None) -> bool:
    layout = (layout_type or "").lower()
    if any(token in layout for token in ("table", "form", "receipt", "worksheet", "exam", "schedule")):
        return True
    if len(boxes) < 10 or page_width <= 0:
        return False
    narrow_ratio = sum(1 for box in boxes if _box_width(box) < 0.45 * page_width) / len(boxes)
    rows = _cluster_rows_by_y_overlap(boxes)
    multi_cell_rows = sum(1 for row in rows if len(row) >= 3)
    return narrow_ratio >= 0.65 and multi_cell_rows >= max(2, int(0.15 * len(rows)))


def _order_atomic_region(
    region: list[list[int]],
    page_width: int,
    layout_type: str | None,
) -> list[list[int]]:
    if len(region) <= 1:
        return region

    if _looks_like_table_or_form(region, page_width=page_width, layout_type=layout_type):
        return _flatten_rows(_cluster_rows_by_y_overlap(region))

    column_groups = _estimate_column_groups(region, page_width=page_width)
    if len(column_groups) > 1:
        ordered: list[list[int]] = []
        for column in sorted(column_groups, key=lambda group: median(_box_center(b)[0] for b in group)):
            ordered.extend(_order_atomic_region(column, page_width=page_width, layout_type=layout_type))
        return ordered

    return _flatten_rows(_cluster_rows_by_y_overlap(region))


def _sort_reading_order_layout_aware(
    boxes: list[list[int]],
    page_width: int,
    page_height: int,
    layout_type: str | None = None,
) -> list[list[int]]:
    """Layout-aware reading order for multi-column, table and form-like pages.

    Policy:
    - Table/form-like layouts: row clustering, then left-to-right within each row.
    - Multi-column layouts: left column top-to-bottom, then right column top-to-bottom.
    - Otherwise: row clustering top-to-bottom, left-to-right (which, for a
      plain paragraph, is one line per row, sorted by Y).
    """
    del page_height  # Reserved for future heuristics.
    if len(boxes) <= 1:
        return boxes

    valid_boxes = [b for b in boxes if _box_width(b) > 0 and _box_height(b) > 0]
    if len(valid_boxes) <= 1:
        return valid_boxes

    return _order_atomic_region(valid_boxes, page_width=page_width, layout_type=layout_type)


def _sort_reading_order(
    boxes: list[list[int]],
    sort_mode: str = "simple",
    page_width: int | None = None,
    page_height: int | None = None,
    layout_type: str | None = None,
) -> list[list[int]]:
    if sort_mode not in _VALID_SORT_MODES:
        raise ValueError(f"Unknown sort_mode={sort_mode!r}; expected one of {sorted(_VALID_SORT_MODES)}")
    if sort_mode == "simple":
        return _sort_reading_order_simple(boxes)
    return _sort_reading_order_layout_aware(
        boxes,
        page_width=page_width or 0,
        page_height=page_height or 0,
        layout_type=layout_type,
    )


# ---------------------------------------------------------------------------
# Manifest reading helpers
# ---------------------------------------------------------------------------
def _read_page_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _page_image_path(row: dict[str, Any]) -> str:
    return row["image_info"][0]["image_url"]


def _page_gt_text(row: dict[str, Any]) -> str:
    parts = [t["text"] for t in row.get("text_info", []) if t.get("tag") != "mask"]
    if parts:
        return " ".join(parts).strip()
    return str(row.get("text", row.get("gt_text", ""))).strip()


def _page_id(row: dict[str, Any], idx: int) -> str:
    meta = row.get("metadata", {})
    src = meta.get("source_index", row.get("source_index", idx))
    bench = meta.get("benchmark", row.get("benchmark", "unknown")).replace("/", "__")
    return f"{bench}_{int(src):06d}" if isinstance(src, int) or str(src).isdigit() else f"{bench}_{src}"


# ---------------------------------------------------------------------------
# Core detection pipeline
# ---------------------------------------------------------------------------
def _detect_and_crop(
    sess: Any,
    image_bgr: np.ndarray,
    max_side: int = 1280,
    prob_thresh: float = 0.25,
    box_thresh: float = 0.4,
    sort_mode: str = "simple",
    layout_type: str | None = None,
) -> list[dict[str, Any]]:
    """Run detection on a single page image. Returns list of {bbox, crop_bgr}."""
    orig_h, orig_w = image_bgr.shape[:2]
    inp, sh, sw = _preprocess_det(image_bgr, max_side=max_side)
    input_name = sess.get_inputs()[0].name
    (prob_map,) = sess.run(None, {input_name: inp})

    boxes = _postprocess_det(
        prob_map,
        orig_h,
        orig_w,
        sh,
        sw,
        prob_thresh=prob_thresh,
        box_thresh=box_thresh,
    )
    boxes = _sort_reading_order(
        boxes,
        sort_mode=sort_mode,
        page_width=orig_w,
        page_height=orig_h,
        layout_type=layout_type,
    )

    results = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        results.append({"bbox": box, "crop_bgr": crop})
    return results


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def _process_dataset(
    manifest_path: Path,
    dataset_name: str,
    out_dir: Path,
    det_session: Any,
    max_side: int,
    prob_thresh: float,
    box_thresh: float,
    resume: bool,
    sort_mode: str,
) -> dict[str, Any]:
    crops_img_dir = out_dir / "images"
    crops_img_dir.mkdir(parents=True, exist_ok=True)

    line_manifest_path = out_dir / "line_manifest.jsonl"
    page_groups_path = out_dir / "page_groups.jsonl"

    if resume and line_manifest_path.exists() and page_groups_path.exists():
        line_count = sum(1 for _ in line_manifest_path.open())
        page_count = sum(1 for _ in page_groups_path.open())
        if line_count > 0 and page_count > 0:
            print(f"  [RESUME] {dataset_name}: {line_count} crops / {page_count} pages already prepared, skipping.")
            return {
                "dataset": dataset_name,
                "pages": page_count,
                "crops": line_count,
                "sort_mode": sort_mode,
                "status": "cached",
            }

    rows = _read_page_manifest(manifest_path)
    print(f"  Processing {dataset_name}: {len(rows)} pages ...")

    total_crops = 0
    failed_pages = 0

    with line_manifest_path.open("w", encoding="utf-8") as lf, page_groups_path.open("w", encoding="utf-8") as pf:
        for idx, row in enumerate(rows):
            page_id = _page_id(row, idx)
            gt_text = _page_gt_text(row)
            img_path = _page_image_path(row)
            meta = row.get("metadata", {})
            layout_type = str(meta.get("layout_type", row.get("layout_type", "unknown")))

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"    WARNING: cannot read image {img_path}, skipping page {page_id}")
                failed_pages += 1
                continue

            try:
                detections = _detect_and_crop(
                    det_session,
                    image_bgr,
                    max_side=max_side,
                    prob_thresh=prob_thresh,
                    box_thresh=box_thresh,
                    sort_mode=sort_mode,
                    layout_type=layout_type,
                )
            except Exception as exc:
                print(f"    WARNING: detection failed for page {page_id}: {exc}")
                failed_pages += 1
                continue

            crop_ids: list[str] = []
            for crop_idx, det in enumerate(detections):
                crop_id = f"{page_id}_line_{crop_idx:04d}"
                crop_path = crops_img_dir / f"{crop_id}.png"
                crop_ids.append(crop_id)

                if not (resume and crop_path.exists()):
                    crop_rgb = cv2.cvtColor(det["crop_bgr"], cv2.COLOR_BGR2RGB)
                    Image.fromarray(crop_rgb).save(crop_path)

                line_record = {
                    "sample_id": crop_id,
                    "image_path": str(crop_path),
                    "text": "",
                    "language": meta.get("language", "unknown"),
                    "layout_type": layout_type,
                    "degradation_profile": meta.get("degradation_profile", "unknown"),
                    "crop_type": "line",
                    "source_page_id": page_id,
                    "bbox": det["bbox"],
                    "sort_mode": sort_mode,
                    "contains_rare_chars": False,
                }
                lf.write(json.dumps(line_record, ensure_ascii=False) + "\n")

            page_record = {
                "page_id": page_id,
                "source_image": img_path,
                "gt_text": gt_text,
                "crop_ids": crop_ids,
                "dataset": dataset_name,
                "language": meta.get("language", "unknown"),
                "layout_type": layout_type,
                "degradation_profile": meta.get("degradation_profile", "unknown"),
                "sort_mode": sort_mode,
            }
            pf.write(json.dumps(page_record, ensure_ascii=False) + "\n")

            total_crops += len(crop_ids)
            if (idx + 1) % 100 == 0:
                print(f"    {idx + 1}/{len(rows)} pages done, {total_crops} crops so far ...")

    print(
        f"  Done {dataset_name}: {len(rows) - failed_pages} pages, {total_crops} crops "
        f"({failed_pages} pages failed, sort_mode={sort_mode})."
    )
    return {
        "dataset": dataset_name,
        "pages": len(rows) - failed_pages,
        "crops": total_crops,
        "failed_pages": failed_pages,
        "sort_mode": sort_mode,
        "status": "done",
        "line_manifest": str(line_manifest_path),
        "page_groups": str(page_groups_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run text-line detection on page-level OCR benchmarks and export crops.")
    p.add_argument("--manifest", default=None, help="Path to a single page-level manifest JSONL. Overrides --all-datasets.")
    p.add_argument("--dataset-name", default=None, help="Short dataset name used for output sub-directory.")
    p.add_argument("--all-datasets", action="store_true", help="Process selected external datasets from manifest root.")
    p.add_argument("--manifest-root", default=_EXTERNAL_MANIFEST_ROOT, help="Root directory for external benchmark manifests.")
    p.add_argument("--out-root", default=_PIPELINE_CROPS_ROOT, help="Root output directory; each dataset gets its own sub-directory.")
    p.add_argument("--det-model", default=_DEFAULT_DET_MODEL, help="Path to the OpenOCR DBNet detection ONNX model.")
    p.add_argument("--max-side", type=int, default=1280, help="Max image side for detection (default 1280).")
    p.add_argument("--prob-thresh", type=float, default=0.25, help="DBNet probability map threshold (default 0.25).")
    p.add_argument("--box-thresh", type=float, default=0.2, help="Mean probability threshold for accepted boxes (default 0.2).")
    p.add_argument("--resume", action="store_true", help="Skip datasets whose output already exists.")
    p.add_argument(
        "--sort-mode",
        choices=sorted(_VALID_SORT_MODES),
        default="simple",
        help="Reading-order sorting mode. Use layout_aware for multi-column/table/form pages.",
    )
    p.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="ONNX Runtime providers (default: CPUExecutionProvider). Use CUDAExecutionProvider for GPU.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    jobs: list[tuple[Path, str, Path]] = []
    if args.manifest:
        if not args.dataset_name:
            raise ValueError("--dataset-name is required when --manifest is specified.")
        jobs.append((Path(args.manifest), args.dataset_name, Path(args.out_root) / args.dataset_name))
    elif args.all_datasets:
        manifest_root = Path(args.manifest_root)
        out_root = Path(args.out_root)
        for ds_name in _ALL_DATASETS:
            for prefix in ("", "tmp_"):
                candidate = manifest_root / f"{prefix}{ds_name}.jsonl"
                if candidate.exists():
                    jobs.append((candidate, ds_name, out_root / ds_name))
                    break
            else:
                print(f"WARNING: manifest not found for {ds_name}, skipping.")
    else:
        raise ValueError("Specify --manifest or --all-datasets.")

    if not jobs:
        print("No datasets to process. Exiting.")
        return
    if ort is None:
        raise RuntimeError("onnxruntime is required to run detection. Install onnxruntime or onnxruntime-gpu.")

    providers = [p.strip() for p in args.providers.split(",")]
    print(f"Loading detection model from {args.det_model} ...")
    det_session = ort.InferenceSession(args.det_model, providers=providers)
    print("Detection model loaded.\n")

    summary: list[dict[str, Any]] = []
    for manifest_path, ds_name, out_dir in jobs:
        print(f"[{ds_name}]")
        result = _process_dataset(
            manifest_path=manifest_path,
            dataset_name=ds_name,
            out_dir=out_dir,
            det_session=det_session,
            max_side=args.max_side,
            prob_thresh=args.prob_thresh,
            box_thresh=args.box_thresh,
            resume=args.resume,
            sort_mode=args.sort_mode,
        )
        summary.append(result)

    print("\n=== Summary ===")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "pipeline_crops_summary.json"
    summary_path.write_text(json.dumps({"jobs": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
