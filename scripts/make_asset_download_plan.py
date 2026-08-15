#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root

PHASES = [
    {
        "name": "metadata",
        "description": "Cards/config/tokenizers only; safe on small disks.",
        "asset_filters": [],
        "metadata_only": True,
        "include_datasets": True,
        "include_benchmarks": True,
    },
    {
        "name": "training_essentials",
        "description": "Dataset plus OpenOCR/SVTRv2-B assets needed before recognizer training.",
        "asset_filters": ["alenisaw/turkicocr-cyrillic", "OpenOCR/SVTRv2-B"],
    },
    {
        "name": "trocr_htr_baselines",
        "description": "Kazakh/Cyrillic TrOCR and HTR baselines.",
        "asset_filters": [
            "thekamilya/kazakh-trocr-fine-tuned",
            "kazars24/trocr-base-handwritten-ru",
            "Kansallisarkisto/cyrillic-htr-model",
        ],
    },
    {
        "name": "external_benchmarks",
        "description": "External validation datasets after core training/eval assets are local.",
        "asset_filters": [
            "henrygagnier/kazakh-ocr",
            "alenisaw/turkicocr-cyrillic",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a phased asset download plan.")
    p.add_argument("--size-estimate", default="outputs/assets/asset_size_estimate.json")
    p.add_argument("--asset-root", default=str(get_asset_root()))
    p.add_argument("--out-json", default="outputs/assets/asset_download_plan.json")
    p.add_argument("--out-md", default="outputs/assets/asset_download_plan.md")
    return p.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _asset_index(size_estimate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(asset.get("id")): asset for asset in size_estimate.get("assets", [])}


def _phase_assets(phase: dict[str, Any], assets_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if phase.get("metadata_only"):
        return []
    return [assets_by_id[item] for item in phase.get("asset_filters", []) if item in assets_by_id]


def _phase_total_gb(phase: dict[str, Any], assets_by_id: dict[str, dict[str, Any]]) -> float:
    return round(
        sum(float(asset.get("total_gb") or 0) for asset in _phase_assets(phase, assets_by_id)),
        3,
    )


def _prepare_command(phase: dict[str, Any], asset_root: str) -> str:
    base = [
        "python scripts/prepare_assets.py",
        "--config configs/assets.yaml",
        f"--asset-root {asset_root}",
    ]
    if phase.get("metadata_only"):
        base.extend(["--metadata-only", "--include-datasets", "--include-benchmarks"])
    else:
        base.append("--include-weights")
        for asset_filter in phase.get("asset_filters", []):
            base.append(f"--asset '{asset_filter}'")
    return " ".join(base)


def build_plan(size_estimate: dict[str, Any], asset_root: str) -> dict[str, Any]:
    assets_by_id = _asset_index(size_estimate)
    phases = []
    cumulative_gb = 0.0
    for phase in PHASES:
        total_gb = _phase_total_gb(phase, assets_by_id)
        cumulative_gb = round(cumulative_gb + total_gb, 3)
        phases.append(
            {
                "name": phase["name"],
                "description": phase["description"],
                "asset_filters": phase.get("asset_filters", []),
                "estimated_gb": total_gb,
                "cumulative_estimated_gb": cumulative_gb,
                "command": _prepare_command(phase, asset_root),
                "assets": [
                    {
                        "id": asset.get("id"),
                        "total_gb": asset.get("total_gb"),
                        "status": asset.get("status"),
                    }
                    for asset in _phase_assets(phase, assets_by_id)
                ],
            }
        )
    return {
        "source_size_estimate": size_estimate.get("config"),
        "asset_root": asset_root,
        "total_estimated_gb": size_estimate.get("summary", {}).get("total_estimated_gb"),
        "phases": phases,
    }


def _write_markdown(path: str | Path, plan: dict[str, Any]) -> None:
    lines = [
        "# TurkicOCR asset download plan",
        "",
        f"Asset root: `{plan['asset_root']}`",
        f"Total estimated HF assets: `{plan['total_estimated_gb']} GB`",
        "",
        "Run phases in order. Each phase can be resumed by rerunning the command.",
        "",
    ]
    for phase in plan["phases"]:
        lines.extend(
            [
                f"## {phase['name']}",
                "",
                phase["description"],
                "",
                f"Estimated phase size: `{phase['estimated_gb']} GB`",
                f"Cumulative estimate: `{phase['cumulative_estimated_gb']} GB`",
                "",
                "```bash",
                phase["command"],
                "```",
                "",
            ]
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    size_estimate = _read_json(args.size_estimate)
    plan = build_plan(size_estimate, args.asset_root)
    write_json(args.out_json, plan)
    _write_markdown(args.out_md, plan)
    print(f"Asset download plan written to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
