from pathlib import Path

from scripts.build_line_only_assets import _filter
from turkicocr.recognition_format import (
    load_recognition_manifest,
    save_recognition_manifest,
    to_openocr_format,
)


def _record(sample_id: str, crop_type: str) -> dict:
    return {
        "sample_id": sample_id,
        "image_path": f"/tmp/{sample_id}.png",
        "text": "text",
        "language": "kk",
        "layout_type": "book",
        "degradation_profile": "clean",
        "crop_type": crop_type,
        "source_page_id": "p1",
    }


def test_filter_keeps_only_line_rows():
    rows = [_record("a", "line"), _record("b", "zone")]

    assert [row["sample_id"] for row in _filter(rows, {"line"})] == ["a"]


def test_openocr_generation_from_filtered_manifest(tmp_path: Path):
    manifest = tmp_path / "rows.jsonl"
    save_recognition_manifest([_record("a", "line")], manifest)
    rows = load_recognition_manifest(manifest)
    out = tmp_path / "openocr.txt"

    to_openocr_format(rows, out)

    assert out.read_text(encoding="utf-8").strip() == "/tmp/a.png\ttext"
