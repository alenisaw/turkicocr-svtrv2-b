#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from turkicocr.gpu import discover_nvidia_gpus_from_proc
from turkicocr.utils import write_json

TESSDATA_LANGS = ("kaz", "rus", "kir")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check local system dependencies for TurkicOCR runs.")
    p.add_argument("--out", default="outputs/system/deps_status.json")
    return p.parse_args()


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {"available": False, "error": "not found"}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _tesseract_status() -> dict[str, Any]:
    status = {
        "binary": shutil.which("tesseract"),
        "version": _run(["tesseract", "--version"]),
        "languages": _run(["tesseract", "--list-langs"]),
    }
    langs = set()
    output = status["languages"].get("stdout", "")
    for line in output.splitlines():
        item = line.strip()
        if item and not item.startswith("List of available languages"):
            langs.add(item)
    status["required_languages"] = {
        lang: lang in langs for lang in TESSDATA_LANGS
    }
    status["ready"] = bool(status["binary"]) and all(status["required_languages"].values())
    return status


def main() -> None:
    args = parse_args()
    proc_gpus = discover_nvidia_gpus_from_proc()
    status = {
        "python_modules": {
            "datasets": _module_available("datasets"),
            "huggingface_hub": _module_available("huggingface_hub"),
            "paddleocr": _module_available("paddleocr"),
            "paddle": _module_available("paddle"),
            "torch": _module_available("torch"),
            "transformers": _module_available("transformers"),
        },
        "executables": {
            "git_lfs": _run(["git", "lfs", "version"]),
            "nvidia_smi": _run(["nvidia-smi"]),
        },
        "gpu_inventory_from_proc": [
            {
                "index": gpu.index,
                "model": gpu.model,
                "bus_id": gpu.bus_id,
                "uuid": gpu.uuid,
            }
            for gpu in proc_gpus
        ],
        "tesseract": _tesseract_status(),
    }
    out = Path(args.out)
    write_json(out, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"Dependency status written to {out}")


if __name__ == "__main__":
    main()
