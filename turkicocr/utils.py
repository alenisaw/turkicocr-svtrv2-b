from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def get_asset_root(path: str | Path | None = None) -> Path:
    if path is not None and str(path).strip():
        return Path(path)
    env_root = os.environ.get("TURKICOCR_ASSET_ROOT")
    if env_root and env_root.strip():
        return Path(env_root)
    mnt_path = Path("/mnt/turkicocr-assets")
    if mnt_path.exists():
        return mnt_path
    return Path("assets")



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def first_present(mapping: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    for name in candidates:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default
