from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .recognition_metrics import cer, chrf_score
from .utils import ensure_dir, iter_jsonl, write_json

RARE_CHARS = set("әғқңөұүіһӘҒҚҢӨҰҮІҺ")


def character_diff(reference: str, prediction: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    max_len = max(len(reference), len(prediction))
    for idx in range(max_len):
        ref = reference[idx] if idx < len(reference) else ""
        pred = prediction[idx] if idx < len(prediction) else ""
        if ref == pred:
            status = "match"
        elif not ref:
            status = "inserted"
        elif not pred:
            status = "missing"
        else:
            status = "substitution"
        rows.append({"index": str(idx), "reference": ref, "prediction": pred, "status": status})
    return rows


def _row_cer(row: dict[str, Any]) -> float:
    return cer(str(row.get("reference", "")), str(row.get("prediction", "")))


def _has_rare_success(row: dict[str, Any]) -> bool:
    ref = str(row.get("reference", ""))
    pred = str(row.get("prediction", ""))
    return any(ch in RARE_CHARS for ch in ref) and ref == pred


def _has_rare_substitution(row: dict[str, Any]) -> bool:
    for item in character_diff(str(row.get("reference", "")), str(row.get("prediction", ""))):
        if item["reference"] in RARE_CHARS and item["status"] != "match":
            return True
    return False


def select_qualitative_examples(rows: list[dict[str, Any]], per_group: int = 10) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [
        ("best_line_predictions", sorted(rows, key=_row_cer)),
        ("worst_line_failures", sorted(rows, key=_row_cer, reverse=True)),
        ("rare_character_successes", [row for row in rows if _has_rare_success(row)]),
        ("rare_character_substitutions", [row for row in rows if _has_rare_substitution(row)]),
        (
            "empty_or_too_short_failures",
            [
                row for row in rows
                if not str(row.get("prediction", "")).strip()
                or len(str(row.get("prediction", ""))) < len(str(row.get("reference", ""))) * 0.7
            ],
        ),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group, candidates in groups:
        for row in candidates[:per_group]:
            key = str(row.get("sample_id") or row.get("id") or row.get("image_path") or row.get("image_url") or "")
            out = dict(row)
            out["qualitative_group"] = group
            out["cer"] = _row_cer(row)
            out["chrf"] = chrf_score(str(row.get("reference", "")), str(row.get("prediction", "")))
            out["diff"] = character_diff(str(row.get("reference", "")), str(row.get("prediction", "")))
            if f"{group}:{key}" not in seen:
                selected.append(out)
                seen.add(f"{group}:{key}")
    return selected


def _make_example_image(row: dict[str, Any], out_path: Path) -> str | None:
    image_path = Path(str(row.get("image_path") or row.get("image_url") or ""))
    if not image_path.exists():
        return None
    try:
        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox = row.get("bbox") or (row.get("metadata") or {}).get("bbox")
        if bbox and len(bbox) == 4:
            draw.rectangle([float(x) for x in bbox], outline="red", width=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return str(out_path)
    except Exception:
        return None


def write_qualitative_report(
    predictions_path: str | Path,
    out_dir: str | Path,
    per_group: int = 10,
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    rows = list(iter_jsonl(predictions_path))
    examples = select_qualitative_examples(rows, per_group=per_group)
    figures_dir = ensure_dir(out / "examples")
    csv_path = out / "qualitative_examples.csv"
    md_path = out / "qualitative_examples.md"
    html_path = out / "qualitative_examples.html"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "qualitative_group",
                "sample_id",
                "image_path",
                "figure_path",
                "cer",
                "chrf",
                "reference",
                "prediction",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(examples):
            figure = _make_example_image(row, figures_dir / f"example_{idx:04d}.jpg")
            row["figure_path"] = figure or ""
            writer.writerow(
                {
                    "qualitative_group": row.get("qualitative_group", ""),
                    "sample_id": row.get("sample_id", row.get("id", "")),
                    "image_path": row.get("image_path", row.get("image_url", "")),
                    "figure_path": row.get("figure_path", ""),
                    "cer": f"{float(row.get('cer', 0.0)):.6f}",
                    "chrf": f"{float(row.get('chrf', 0.0)):.6f}",
                    "reference": row.get("reference", ""),
                    "prediction": row.get("prediction", ""),
                }
            )

    md_lines = ["# Qualitative OCR Examples", ""]
    html_lines = ["<html><body><h1>Qualitative OCR Examples</h1>"]
    for row in examples:
        title = f"{row.get('qualitative_group')} / {row.get('sample_id', row.get('id', ''))}"
        md_lines.extend([f"## {title}", "", f"- CER: {row.get('cer'):.6f}", f"- chrF: {row.get('chrf'):.6f}"])
        if row.get("figure_path"):
            md_lines.append(f"![example]({row['figure_path']})")
        md_lines.extend(["", "**Reference**", "", str(row.get("reference", "")), "", "**Prediction**", "", str(row.get("prediction", "")), ""])
        html_lines.append(f"<h2>{html.escape(str(title))}</h2>")
        if row.get("figure_path"):
            html_lines.append(f'<img src="{html.escape(str(row["figure_path"]))}" style="max-width:900px">')
        html_lines.append(f"<p><b>CER:</b> {float(row.get('cer', 0.0)):.6f} <b>chrF:</b> {float(row.get('chrf', 0.0)):.6f}</p>")
        html_lines.append(f"<p><b>Reference:</b> {html.escape(str(row.get('reference', '')))}</p>")
        html_lines.append(f"<p><b>Prediction:</b> {html.escape(str(row.get('prediction', '')))}</p>")
    html_lines.append("</body></html>")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    html_path.write_text("\n".join(html_lines), encoding="utf-8")
    write_json(out / "qualitative_examples.json", examples)
    return {
        "count": len(examples),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def write_checkpoint_metrics_report(metrics_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(metrics_root).glob("epoch_*/metrics_aggregate.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {"checkpoint": path.parent.name, **metrics}
        rows.append(row)
    csv_path = out / "checkpoint_metrics.csv"
    if rows:
        keys = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    md = ["# Checkpoint Metrics", ""]
    for row in rows:
        md.append(
            f"- `{row['checkpoint']}`: CER={float(row.get('val_rec_cer', 0.0)):.4f}, "
            f"chrF={float(row.get('val_rec_chrf', 0.0)):.4f}, "
            f"empty={float(row.get('val_rec_empty_prediction_rate', 0.0)):.4f}"
        )
    (out / "checkpoint_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"count": len(rows), "csv": str(csv_path), "markdown": str(out / "checkpoint_metrics.md")}
