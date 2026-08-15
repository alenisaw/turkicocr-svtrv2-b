import pytest

from turkicocr.recognition_format import (
    load_recognition_manifest,
    normalize_recognition_text,
    parse_openocr_format,
    save_recognition_manifest,
    text_length_bucket,
    to_openocr_format,
    validate_recognition_record,
)


def _record(text="Қазақстан"):
    return {
        "sample_id": "rec_001",
        "image_path": "/tmp/crop.png",
        "text": text,
        "language": "kazakh",
        "layout_type": "forms",
        "degradation_profile": "clean",
        "crop_type": "line",
        "source_page_id": "page_001",
    }


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    save_recognition_manifest([_record("  Қазақстан   Республикасы  ")], path)

    rows = load_recognition_manifest(path)

    assert rows[0]["sample_id"] == "rec_001"
    assert rows[0]["text"] == "Қазақстан Республикасы"
    assert rows[0]["text_length_bucket"] == "short"


def test_validate_recognition_record_missing_required_field():
    row = _record()
    row.pop("image_path")

    with pytest.raises(ValueError, match="image_path"):
        validate_recognition_record(row)


def test_openocr_format_roundtrip(tmp_path):
    txt_path = tmp_path / "openocr.txt"
    lines = to_openocr_format([_record()], txt_path)

    assert lines == ["/tmp/crop.png\tҚазақстан"]
    rows = parse_openocr_format(txt_path)
    assert rows[0]["image_path"] == "/tmp/crop.png"
    assert rows[0]["text"] == "Қазақстан"


def test_text_length_bucket_and_normalization():
    assert normalize_recognition_text(" a\u00a0  b ") == "a b"
    assert text_length_bucket("қысқа") == "short"
    assert text_length_bucket("x" * 80) == "medium"
    assert text_length_bucket("x" * 200) == "long"
    assert text_length_bucket("x" * 700) == "very_long"
