from pathlib import Path

from PIL import Image

from turkicocr.reports import character_diff, write_qualitative_report
from turkicocr.utils import write_jsonl


def test_character_diff_marks_missing_and_substitution():
    diff = character_diff("abc", "ax")

    assert diff[1]["status"] == "substitution"
    assert diff[2]["status"] == "missing"


def test_write_qualitative_report_outputs_csv_md_html(tmp_path: Path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 16), "white").save(image)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "sample_id": "s1",
                "image_path": str(image),
                "reference": "қазақ",
                "prediction": "казак",
                "metadata": {"crop_type": "line"},
            }
        ],
    )

    summary = write_qualitative_report(predictions, tmp_path / "report", per_group=2)

    assert summary["count"] >= 1
    assert Path(summary["csv"]).exists()
    assert Path(summary["markdown"]).exists()
    assert Path(summary["html"]).exists()
