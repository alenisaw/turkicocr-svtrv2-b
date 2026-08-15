from __future__ import annotations

import json
from pathlib import Path

from turkicocr.recognition_dataset import RecognitionDataset, sample_diagnostic_batch


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "sample_id": "plain",
            "image_path": "/tmp/plain.png",
            "text": "обычный текст",
            "language": "ru",
            "layout_type": "line",
            "degradation_profile": "clean",
            "crop_type": "line",
            "source_page_id": "p1",
        },
        {
            "sample_id": "rare",
            "image_path": "/tmp/rare.png",
            "text": "қазақ әліпбиі",
            "language": "kk",
            "layout_type": "line",
            "degradation_profile": "clean",
            "crop_type": "line",
            "source_page_id": "p2",
        },
        {
            "sample_id": "other",
            "image_path": "/tmp/other.png",
            "text": "тағы мәтін",
            "language": "kk",
            "layout_type": "zone",
            "degradation_profile": "scan",
            "crop_type": "zone",
            "source_page_id": "p3",
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_recognition_dataset_uses_loader_and_transform(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    dataset = RecognitionDataset(
        manifest,
        image_loader=lambda path: f"image:{Path(path).name}",
        transform=lambda value: value.upper(),
    )

    assert len(dataset) == 3
    item = dataset[0]
    assert item["image"] == "IMAGE:PLAIN.PNG"
    assert item["text"] == "обычный текст"
    assert item["record"]["sample_id"] == "plain"


def test_sample_diagnostic_batch_prefers_rare_chars(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)

    rows = sample_diagnostic_batch(manifest, n=2, seed=7, prefer_rare_chars=True)

    assert len(rows) == 2
    assert any("қ" in row["text"] or "ә" in row["text"] for row in rows)
