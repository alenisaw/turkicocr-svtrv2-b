#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from scripts.run_openocr_train import write_resolved_openocr_config
from turkicocr.recognition_format import load_recognition_manifest
from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recognition inference on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", default="openocr_svtrv2")
    parser.add_argument("--turkicocr-config", default="configs/train_svtrv2_b_rec.yaml")
    parser.add_argument("--openocr-root", default=str(get_asset_root() / "openocr/repo"))
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--development-echo", action="store_true", help="Write references as predictions for smoke tests.")
    return parser.parse_args()


def _checkpoint_path(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        for name in ("best.pth", "latest.pth"):
            candidate = p / name
            if candidate.exists():
                return str(candidate)
    return str(p)


def _openocr_config(args: argparse.Namespace) -> dict:
    resolved = write_resolved_openocr_config(args.turkicocr_config, args.openocr_root, args.asset_root)
    cfg = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
    cfg["Global"]["checkpoints"] = _checkpoint_path(args.checkpoint)
    cfg["Global"]["device"] = args.device
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False
    cfg["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    eval_transforms = cfg["Eval"]["dataset"]["transforms"]
    for item in eval_transforms:
        if "RecDynamicResize" in item:
            item["RecDynamicResize"]["padding"] = True
    return cfg


def _load_openocr_recognizer(args: argparse.Namespace):
    openocr_root = str(Path(args.openocr_root).resolve())
    if openocr_root not in sys.path:
        sys.path.insert(0, openocr_root)
    tools_root = str((Path(args.openocr_root) / "tools").resolve())
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    from tools.infer_rec import OpenRecognizer

    return OpenRecognizer(config=_openocr_config(args), mode="server", backend="torch", numId=args.device_id)


def main() -> None:
    args = parse_args()
    rows = load_recognition_manifest(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recognizer = None if args.development_echo else _load_openocr_recognizer(args)
    failures: list[dict[str, str]] = []
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            started = time.perf_counter()
            prediction = row["text"]
            score = None
            if recognizer is not None:
                try:
                    result = recognizer(img_path=row["image_path"], batch_num=args.batch_size)[0]
                    prediction = str(result.get("text", ""))
                    score = float(result.get("score", 0.0))
                    latency_ms = float(result.get("elapse", 0.0)) * 1000.0
                except Exception as exc:
                    prediction = ""
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    failures.append({"sample_id": str(row["sample_id"]), "error": repr(exc)})
            else:
                latency_ms = (time.perf_counter() - started) * 1000.0
            pred = {
                "checkpoint": args.checkpoint,
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "reference": row["text"],
                "prediction": prediction,
                "score": score,
                "metadata": {
                    "language": row.get("language", "unknown"),
                    "layout_type": row.get("layout_type", "unknown"),
                    "degradation_profile": row.get("degradation_profile", "unknown"),
                    "crop_type": row.get("crop_type", "unknown"),
                    "source_page_id": row.get("source_page_id", "unknown"),
                    "bbox": row.get("bbox"),
                    "text_length_bucket": row.get("text_length_bucket", "unknown"),
                    "latency_ms": latency_ms,
                },
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    if failures:
        write_json(out.with_suffix(".failures.json"), {"count": len(failures), "failures": failures})
    print(f"wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
