#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from turkicocr.utils import write_json

REQUIRED_FILES = [
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-inference.txt",
    ".agent/PROJECT_PLAN.md",
    ".agent/policy/THIRD_PARTY_NOTICES.md",
    "configs/assets.yaml",
    "configs/dataset.yaml",
    "configs/eval_baselines.yaml",
    "configs/eval_page_oracle.yaml",
    "configs/eval_recognition.yaml",
    "configs/export_onnx.yaml",
    "configs/quantization.yaml",
    "configs/recognition_gate.yaml",
    "configs/server.yaml",
    "configs/train_svtrv2_b_rec.yaml",
    "scripts/build_recognition_diagnostic.py",
    "scripts/check_openocr_ready.py",
    "scripts/evaluate_openocr_checkpoints.py",
    "scripts/export_recognition_crops.py",
    "scripts/patch_openocr_runtime.py",
    "scripts/run_openocr_train.py",
    "scripts/run_recognition_eval.py",
    "scripts/run_recognition_inference.py",
    "scripts/train_svtrv2_b_rec.py",
    "docs/model_cards/collection/README.md",
    "docs/model_cards/turkicocr-svtrv2-b/README.md",
    "docs/model_cards/turkicocr-svtrv2-b-int8/README.md",
    "docs/model_cards/turkicocr-svtrv2-b-onnx/README.md",
    "data/samples/sample_manifest.jsonl",
]

REQUIRED_DIRS = ["configs", "scripts", "turkicocr", "tests", "docs", ".agent"]
MUST_BE_ABSENT = ["PROJECT_PLAN.md", "THIRD_PARTY_NOTICES.md"]
MUST_BE_IGNORED = [
    ".agent/logs/ACTION_LOG.md",
    ".venv/bin/python",
    "outputs/predictions/example.jsonl",
    "outputs/env/requirements.freeze.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TurkicOCR SVTRv2-B repository readiness.")
    parser.add_argument("--out", default="outputs/reports/repo_ready.json")
    return parser.parse_args()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _required_paths() -> list[dict[str, Any]]:
    rows = [{"path": path, "ok": Path(path).is_file()} for path in REQUIRED_FILES]
    rows.extend({"path": path, "ok": Path(path).is_dir(), "type": "dir"} for path in REQUIRED_DIRS)
    rows.extend({"path": path, "ok": not Path(path).exists(), "type": "absent"} for path in MUST_BE_ABSENT)
    return rows


def _ignored_paths() -> list[dict[str, Any]]:
    rows = []
    for path in MUST_BE_IGNORED:
        result = _run(["git", "check-ignore", "-q", path])
        rows.append({"path": path, "ok": result.returncode == 0})
    return rows


def _sample_manifest() -> list[dict[str, Any]]:
    path = Path("data/samples/sample_manifest.jsonl")
    required = {
        "sample_id",
        "image_path",
        "text",
        "language",
        "layout_type",
        "degradation_profile",
        "crop_type",
        "source_page_id",
    }
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append({"line": idx, "ok": False, "error": str(exc)})
            continue
        missing = sorted(required - set(row))
        rows.append({"line": idx, "ok": not missing, "missing": missing})
    return rows or [{"check": "non_empty", "ok": False}]


def _readme() -> list[dict[str, Any]]:
    text = Path("README.md").read_text(encoding="utf-8")
    checks = {
        "names_svtrv2_model": "TurkicOCR-SVTRv2-B" in text,
        "names_openocr_backend": "OpenOCR" in text,
        "states_not_official": "not an official OpenOCR" in text or "not an official OpenOCR, PaddleOCR" in text,
        "describes_document_pipeline": "document pipeline benchmark" in text.lower(),
        "uses_svtrv2_as_main_model": "TurkicOCR-SVTRv2-B" in text,
    }
    return [{"check": key, "ok": value} for key, value in checks.items()]


def _model_card() -> list[dict[str, Any]]:
    path = Path("docs/model_cards/turkicocr-svtrv2-b/README.md")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    checks = {
        "has_hf_front_matter": text.startswith("---\n"),
        "names_model": "TurkicOCR-SVTRv2-B" in text,
        "states_independent": "independent" in text.lower(),
        "mentions_openocr": "OpenOCR" in text,
        "has_evaluation_section": "Evaluation" in text,
    }
    return [{"check": key, "ok": value} for key, value in checks.items()]


def main() -> None:
    args = parse_args()
    groups = {
        "required_paths": _required_paths(),
        "ignored_paths": _ignored_paths(),
        "sample_manifest": _sample_manifest(),
        "readme": _readme(),
        "model_card": _model_card(),
    }
    failures = []
    for group, rows in groups.items():
        for row in rows:
            if not row.get("ok"):
                failures.append({"group": group, **row})
    report = {"ok": not failures, "failures": failures, "groups": groups}
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
