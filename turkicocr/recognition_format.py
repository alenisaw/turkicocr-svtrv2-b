from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_RECOGNITION_FIELDS: tuple[str, ...] = (
    "sample_id",
    "image_path",
    "text",
    "language",
    "layout_type",
    "degradation_profile",
    "crop_type",
    "source_page_id",
)


def normalize_recognition_text(text: str | None) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split()).strip()


def text_length_bucket(text: str) -> str:
    n = len(normalize_recognition_text(text))
    if n < 32:
        return "short"
    if n < 128:
        return "medium"
    if n < 512:
        return "long"
    return "very_long"


def validate_recognition_record(record: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field
        for field in REQUIRED_RECOGNITION_FIELDS
        if field not in record or record[field] is None or record[field] == ""
    ]
    if missing:
        raise ValueError(f"Recognition record is missing required fields: {', '.join(missing)}")
    normalized = dict(record)
    normalized["sample_id"] = str(normalized["sample_id"])
    normalized["image_path"] = str(normalized["image_path"])
    normalized["text"] = normalize_recognition_text(str(normalized["text"]))
    for key in (
        "language",
        "layout_type",
        "degradation_profile",
        "crop_type",
        "source_page_id",
    ):
        normalized[key] = str(normalized[key])
    normalized.setdefault("text_length_bucket", text_length_bucket(normalized["text"]))
    return normalized


def load_recognition_manifest(path: str | Path, validate: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not validate:
        return rows
    return [validate_recognition_record(row) for row in rows]


def save_recognition_manifest(records: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(validate_recognition_record(row), ensure_ascii=False) + "\n")


def to_openocr_format(
    records: list[dict[str, Any]], out_path: str | Path | None = None
) -> list[str]:
    lines = [
        f"{row['image_path']}\t{row['text']}"
        for row in (validate_recognition_record(record) for record in records)
    ]
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def parse_openocr_format(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"OpenOCR line {idx + 1} must contain a tab separator")
            image_path, text = line.split("\t", 1)
            records.append(
                {
                    "sample_id": Path(image_path).stem or f"openocr_{idx:06d}",
                    "image_path": image_path,
                    "text": normalize_recognition_text(text),
                    "language": "unknown",
                    "layout_type": "unknown",
                    "degradation_profile": "unknown",
                    "crop_type": "unknown",
                    "source_page_id": "unknown",
                    "text_length_bucket": text_length_bucket(text),
                }
            )
    return records
