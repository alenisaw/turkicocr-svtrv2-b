from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inference import run_manifest_inference
from .utils import ensure_dir, read_yaml, write_json


@dataclass(frozen=True)
class BaselineModel:
    id: str
    display_name: str
    type: str
    checkpoint: str | None = None
    languages: tuple[str, ...] = ()


def load_baseline_config(config_path: str | Path) -> list[BaselineModel]:
    cfg = read_yaml(config_path)
    models = []
    for item in cfg.get("models", []):
        models.append(
            BaselineModel(
                id=str(item["id"]),
                display_name=str(item.get("display_name", item["id"])),
                type=str(item.get("type", "unknown")),
                checkpoint=item.get("checkpoint"),
                languages=tuple(str(value) for value in item.get("languages", [])),
            )
        )
    return models


def run_baselines(
    config_path: str | Path,
    out_dir: str | Path,
    development_echo: bool = False,
    manifest_override: str | None = None,
    device: str | None = None,
    force: bool = False,
    model_ids: set[str] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    manifest = manifest_override or cfg.get("evaluation", {}).get("manifest")
    if not manifest:
        raise ValueError(f"Evaluation manifest must be set in {config_path}")
    out = ensure_dir(out_dir)
    status_path = out / "baseline_status.json"
    existing_rows: dict[str, dict[str, Any]] = {}
    if model_ids and status_path.exists():
        try:
            existing = read_yaml(status_path)
            existing_rows = {
                str(row.get("id")): row
                for row in existing.get("models", [])
                if row.get("id")
            }
        except Exception:
            existing_rows = {}
    result = {"models": []}
    updated_rows: dict[str, dict[str, Any]] = {}
    for model in load_baseline_config(config_path):
        if model_ids and model.id not in model_ids:
            continue
        pred_path = out / f"{model.id}_predictions.jsonl"
        if force and pred_path.exists():
            pred_path.unlink()
            summary = pred_path.with_suffix(".summary.json")
            if summary.exists():
                summary.unlink()
        if pred_path.exists() and pred_path.stat().st_size > 0:
            status = "development_echo" if development_echo else "completed"
            updated_rows[model.id] = {"id": model.id, "status": status, "predictions": str(pred_path)}
            continue
        try:
            checkpoint = model.checkpoint
            if model.type == "tesseract" and model.languages:
                checkpoint = "+".join(model.languages)
            run_manifest_inference(
                manifest=manifest,
                out_path=pred_path,
                checkpoint=checkpoint or "",
                backend_name=model.type,
                development_echo=development_echo,
                device=device,
                max_samples=max_samples,
            )
            status = "development_echo" if development_echo else "completed"
            error = None
        except Exception as exc:
            status = "failed"
            error = str(exc)
            pred_path.write_text("", encoding="utf-8")
        row = {"id": model.id, "status": status, "predictions": str(pred_path)}
        if error:
            row["error"] = error
        updated_rows[model.id] = row
    if model_ids and existing_rows:
        existing_rows.update(updated_rows)
        configured_order = [model.id for model in load_baseline_config(config_path)]
        result["models"] = [
            existing_rows[model_id]
            for model_id in configured_order
            if model_id in existing_rows
        ]
    else:
        result["models"] = list(updated_rows.values())
    write_json(status_path, result)
    return result
