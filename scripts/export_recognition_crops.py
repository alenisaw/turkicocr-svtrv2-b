#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PIL import Image

from turkicocr.rare_chars import RARE_CHARS, contains_rare_char
from turkicocr.recognition_format import text_length_bucket, validate_recognition_record
from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TurkicOCR line/zone crops for OpenOCR recognition training.")
    parser.add_argument("--dataset", default="alenisaw/turkicocr-cyrillic")
    parser.add_argument("--config", default="large")
    parser.add_argument("--dataset-root", default=str(get_asset_root() / "datasets/alenisaw/turkicocr-cyrillic"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--charset-out", required=True)
    parser.add_argument("--include", default="line,zone,table_cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pages-per-split", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_dataset(dataset: str, config: str, dataset_root: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets>=3.0.0 to export recognition crops.") from exc
    local = Path(dataset_root)
    if local.exists():
        return load_dataset(str(local), name=config)
    return load_dataset(dataset, name=config)


def _page_image(dataset_root: Path, row: dict[str, Any]) -> Image.Image:
    tar_path = Path(str(row["tar_path"]))
    if not tar_path.is_absolute():
        tar_path = dataset_root / tar_path
    member = str(row["image_filename"])
    with tarfile.open(tar_path) as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} not found in {tar_path}")
        return Image.open(extracted).convert("RGB")


def _safe_crop(image: Image.Image, bbox: list[float] | tuple[float, ...], padding: int = 2) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    left = max(0, int(x1) - padding)
    top = max(0, int(y1) - padding)
    right = min(w, int(x2) + padding)
    bottom = min(h, int(y2) + padding)
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop bbox {bbox} for image size {(w, h)}")
    return image.crop((left, top, right, bottom))


def _infer_language(row: dict[str, Any], zone: dict[str, Any] | None, text: str) -> str:
    if zone and zone.get("language"):
        return str(zone["language"])
    lower = text.lower()
    if any(ch in lower for ch in ("ә", "ғ", "қ", "ұ", "і", "һ")):
        return "kazakh"
    if any(ch in lower for ch in ("ң", "ө", "ү")):
        return "kyrgyz_or_kazakh"
    return "russian_or_other"


def _zone_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not row.get("zones_json"):
        return []
    zones = json.loads(row["zones_json"])
    return zones if isinstance(zones, list) else []


def _iter_crop_specs(row: dict[str, Any], include: set[str]):
    if "line" in include:
        bboxes = row.get("ocr_bboxes") or []
        labels = row.get("ocr_labels") or []
        region_ids = row.get("ocr_region_ids") or []
        for idx, (bbox, text) in enumerate(zip(bboxes, labels, strict=False)):
            if not str(text).strip():
                continue
            yield {
                "crop_type": "line",
                "bbox": bbox,
                "text": str(text),
                "region_id": str(region_ids[idx]) if idx < len(region_ids) else f"line_{idx:04d}",
                "zone": None,
            }
    for zone in _zone_records(row):
        text = str(zone.get("text", "")).strip()
        bbox = zone.get("bbox")
        zone_type = str(zone.get("zone_type", zone.get("type", "zone")))
        if not text or not bbox:
            continue
        crop_type = "table_cell" if "cell" in zone_type or "table" in zone_type else "zone"
        if crop_type in include or "zone" in include and crop_type == "zone":
            yield {
                "crop_type": crop_type,
                "bbox": bbox,
                "text": text,
                "region_id": str(zone.get("zone_id", zone.get("id", zone_type))),
                "zone": zone,
            }


_WORKER_DS = None
_WORKER_DATASET_ROOT: Path | None = None
_WORKER_IMAGE_ROOT: Path | None = None
_WORKER_INCLUDE: set[str] = set()
_WORKER_DRY_RUN = False
_WORKER_RESUME = False


def _init_worker(
    dataset: str,
    config: str,
    dataset_root: str,
    image_root: str,
    include: tuple[str, ...],
    dry_run: bool,
    resume: bool,
) -> None:
    global _WORKER_DS
    global _WORKER_DATASET_ROOT
    global _WORKER_IMAGE_ROOT
    global _WORKER_INCLUDE
    global _WORKER_DRY_RUN
    global _WORKER_RESUME
    _WORKER_DS = _load_dataset(dataset, config, dataset_root)
    _WORKER_DATASET_ROOT = Path(dataset_root)
    _WORKER_IMAGE_ROOT = Path(image_root)
    _WORKER_INCLUDE = set(include)
    _WORKER_DRY_RUN = dry_run
    _WORKER_RESUME = resume


def _record_for_spec(row: dict[str, Any], out_split: str, page_id: str, spec_idx: int, spec: dict[str, Any]) -> dict[str, Any]:
    assert _WORKER_IMAGE_ROOT is not None
    sample_id = f"{out_split}_{page_id}_{spec['crop_type']}_{spec_idx:04d}"
    image_path = _WORKER_IMAGE_ROOT / out_split / f"{sample_id}.png"
    text = " ".join(str(spec["text"]).split()).strip()
    record = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "text": text,
        "language": _infer_language(row, spec.get("zone"), text),
        "layout_type": str(row.get("layout_id", "unknown")),
        "degradation_profile": "unknown",
        "crop_type": spec["crop_type"],
        "source_page_id": page_id,
        "bbox": spec["bbox"],
        "region_id": spec["region_id"],
        "contains_rare_chars": contains_rare_char(text),
        "rare_chars": [ch for ch in RARE_CHARS if ch in text.lower()],
        "text_length_bucket": text_length_bucket(text),
    }
    validate_recognition_record(record)
    return record


def _export_page(task: tuple[str, str, int]) -> dict[str, Any]:
    assert _WORKER_DS is not None
    assert _WORKER_DATASET_ROOT is not None
    split, out_split, page_idx = task
    row = dict(_WORKER_DS[split][page_idx])
    page_id = str(row.get("page_id", f"{split}_{page_idx:06d}"))
    specs = list(_iter_crop_specs(row, _WORKER_INCLUDE))
    records = [_record_for_spec(row, out_split, page_id, spec_idx, spec) for spec_idx, spec in enumerate(specs)]

    missing = []
    if not _WORKER_DRY_RUN:
        for record, spec in zip(records, specs, strict=False):
            image_path = Path(record["image_path"])
            if not (_WORKER_RESUME and image_path.exists()):
                missing.append((image_path, spec))
        if missing:
            image = _page_image(_WORKER_DATASET_ROOT, row)
            for image_path, spec in missing:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                _safe_crop(image, spec["bbox"]).save(image_path)

    char_counts: Counter[str] = Counter()
    manifest_lines = []
    openocr_lines = []
    for record in records:
        text = record["text"]
        char_counts.update(text)
        manifest_lines.append(json.dumps(record, ensure_ascii=False))
        openocr_lines.append(f"{record['image_path']}\t{text}")
    return {
        "page_idx": page_idx,
        "records": len(records),
        "manifest_lines": manifest_lines,
        "openocr_lines": openocr_lines,
        "char_counts": dict(char_counts),
    }


def _iter_parallel_results(
    executor: ProcessPoolExecutor,
    tasks,
    max_pending: int,
):
    task_iter = iter(tasks)
    pending = set()
    while True:
        while len(pending) < max_pending:
            try:
                pending.add(executor.submit(_export_page, next(task_iter)))
            except StopIteration:
                break
        if not pending:
            break
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    image_root = out_dir / "images"
    include = {item.strip() for item in args.include.split(",") if item.strip()}
    ds = _load_dataset(args.dataset, args.config, args.dataset_root)

    all_chars: Counter[str] = Counter()
    reports: dict[str, Any] = {"splits": {}, "dry_run": args.dry_run}
    split_map = {"train": "train", "validation": "val", "test": "test"}
    for split, out_split in split_map.items():
        limit = args.max_pages_per_split or len(ds[split])
        page_count = min(limit, len(ds[split]))
        record_count = 0
        manifest_path = manifest_dir / f"{out_split}_rec_large.jsonl"
        openocr_path = manifest_dir / f"{out_split}_openocr.txt"
        manifest_file = None
        openocr_file = None
        if not args.dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_path.open("w", encoding="utf-8")
            openocr_file = openocr_path.open("w", encoding="utf-8")
        try:
            tasks = ((split, out_split, page_idx) for page_idx in range(page_count))
            completed_pages = 0
            if args.num_workers == 1:
                _init_worker(
                    args.dataset,
                    args.config,
                    args.dataset_root,
                    str(image_root),
                    tuple(sorted(include)),
                    args.dry_run,
                    args.resume,
                )
                results = (_export_page(task) for task in tasks)
            else:
                max_pending = max(args.num_workers * 4, args.num_workers)
                executor = ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    initializer=_init_worker,
                    initargs=(
                        args.dataset,
                        args.config,
                        args.dataset_root,
                        str(image_root),
                        tuple(sorted(include)),
                        args.dry_run,
                        args.resume,
                    ),
                )
                results = _iter_parallel_results(executor, tasks, max_pending)
            with executor if args.num_workers > 1 else nullcontext():
                for result in results:
                    completed_pages += 1
                    record_count += int(result["records"])
                    all_chars.update(result["char_counts"])
                    if manifest_file and openocr_file:
                        for line in result["manifest_lines"]:
                            manifest_file.write(line + "\n")
                        for line in result["openocr_lines"]:
                            openocr_file.write(line + "\n")
                    if completed_pages % args.progress_every == 0:
                        reports["splits"][out_split] = {"pages": completed_pages, "records": record_count}
                        write_json(out_dir / "export_recognition_crops_report.json", reports)
        finally:
            if manifest_file:
                manifest_file.close()
            if openocr_file:
                openocr_file.close()
        reports["splits"][out_split] = {"pages": page_count, "records": record_count}

    charset = sorted(ch for ch in all_chars if ch not in {"\n", "\r"})
    if not args.dry_run:
        Path(args.charset_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.charset_out).write_text("\n".join(charset) + "\n", encoding="utf-8")
    reports["charset_size"] = len(charset)
    reports["rare_chars"] = {ch: all_chars.get(ch, 0) for ch in RARE_CHARS}
    write_json(out_dir / "export_recognition_crops_report.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
