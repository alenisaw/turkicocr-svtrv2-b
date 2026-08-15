from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .utils import ensure_dir, first_present, write_json

DEFAULT_TEXT_FIELDS = ["text", "full_text", "page_text", "target_text", "ocr_text"]
DEFAULT_IMAGE_FIELDS = [
    "image",
    "image_path",
    "image_filename",
    "path",
    "file_name",
    "filename",
]


def load_hf_dataset(dataset_id: str, config: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets>=3.0.0 to load Hugging Face datasets.") from exc
    return load_dataset(dataset_id, name=config)


def extract_text(example: dict[str, Any], candidates: list[str] | None = None) -> str:
    value = first_present(example, candidates or DEFAULT_TEXT_FIELDS, "")
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value or "")


def extract_image_ref(example: dict[str, Any], candidates: list[str] | None = None) -> str:
    tar_path = example.get("tar_path")
    image_filename = example.get("image_filename")
    if tar_path and image_filename:
        return f"{tar_path}::{image_filename}"
    value = first_present(example, candidates or DEFAULT_IMAGE_FIELDS, "")
    if hasattr(value, "filename") and value.filename:
        return str(value.filename)
    return str(value or "")


def extract_metadata(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": example.get("language", example.get("lang", "unknown")),
        "layout_type": example.get("layout_type", example.get("layout", "unknown")),
        "degradation_profile": example.get(
            "degradation_profile", example.get("degradation", "unknown")
        ),
        "doc_type": example.get("doc_type", example.get("document_type", "unknown")),
    }


def split_counts(dataset_dict: Any) -> dict[str, int]:
    return {split: len(dataset_dict[split]) for split in dataset_dict.keys()}


def iter_split(dataset_dict: Any, split: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    ds = dataset_dict[split]
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = dict(ds[i])
        row["_index"] = i
        row["_split"] = split
        yield row


def audit_dataset(
    dataset_dict: Any, out_dir: str | Path, limit_per_split: int | None = None
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    counts = split_counts(dataset_dict)
    language = Counter()
    layout = Counter()
    degradation = Counter()
    doc_type = Counter()
    chars = Counter()

    for split in dataset_dict.keys():
        for row in iter_split(dataset_dict, split, limit_per_split):
            text = extract_text(row)
            meta = extract_metadata(row)
            language[str(meta["language"])] += 1
            layout[str(meta["layout_type"])] += 1
            degradation[str(meta["degradation_profile"])] += 1
            doc_type[str(meta["doc_type"])] += 1
            chars.update(text)

    rare_chars = {ch: chars.get(ch, 0) for ch in "әғқңөұүіһ"}
    stats = {
        "split_counts": counts,
        "language_distribution": dict(language),
        "layout_distribution": dict(layout),
        "degradation_distribution": dict(degradation),
        "doc_type_distribution": dict(doc_type),
        "rare_char_counts": rare_chars,
        "total_unique_chars": len(chars),
        "audit_limit_per_split": limit_per_split,
    }
    write_json(out / "dataset_stats.json", stats)
    write_counter_csv(out / "language_distribution.csv", language, "language")
    write_counter_csv(out / "layout_distribution.csv", layout, "layout_type")
    write_counter_csv(out / "degradation_distribution.csv", degradation, "degradation_profile")
    write_counter_csv(out / "char_distribution.csv", chars, "char")
    write_counter_csv(out / "rare_char_counts.csv", Counter(rare_chars), "char")
    write_dataset_report(out / "dataset_report.md", stats)
    return stats


def write_counter_csv(path: str | Path, counter: Counter, key_name: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(f"{key_name},count\n")
        for key, count in counter.most_common():
            safe = str(key).replace('"', '""')
            f.write(f'"{safe}",{count}\n')


def write_dataset_report(path: str | Path, stats: dict[str, Any]) -> None:
    lines = [
        "# TurkicOCR dataset audit",
        "",
        "## Split counts",
        "",
    ]
    for split, count in stats["split_counts"].items():
        lines.append(f"- `{split}`: {count}")
    lines.extend(["", "## Rare-character counts", ""])
    for ch, count in stats["rare_char_counts"].items():
        lines.append(f"- `{ch}`: {count}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This report is generated from recognition crop manifests.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
