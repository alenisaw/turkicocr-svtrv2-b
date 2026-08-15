#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

from scripts.run_openocr_train import write_resolved_openocr_config
from turkicocr.recognition_format import load_recognition_manifest
from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX recognition inference on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--turkicocr-config", default="configs/train_svtrv2_b_rec_line.yaml")
    parser.add_argument("--openocr-root", default=str(get_asset_root() / "openocr/repo"))
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _openocr_config(args: argparse.Namespace) -> dict:
    resolved = write_resolved_openocr_config(args.turkicocr_config, args.openocr_root, args.asset_root)
    cfg = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
    cfg["Global"]["checkpoints"] = None
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["device"] = args.device
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False
    cfg["Global"]["backend"] = "onnx"
    cfg["Global"]["onnx_model_path"] = str(Path(args.onnx_model).resolve())
    cfg["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    for item in cfg["Eval"]["dataset"]["transforms"]:
        if "RecDynamicResize" in item:
            item["RecDynamicResize"]["padding"] = True
    return cfg


def _load_recognizer(args: argparse.Namespace):
    openocr_root = str(Path(args.openocr_root).resolve())
    tools_root = str((Path(args.openocr_root) / "tools").resolve())
    for item in (openocr_root, tools_root):
        if item not in sys.path:
            sys.path.insert(0, item)
    from tools.infer_rec import OpenRecognizer

    return OpenRecognizer(config=_openocr_config(args), mode="mobile", backend="onnx", use_gpu="true" if args.device == "gpu" else "false")


def main() -> None:
    args = parse_args()
    rows = load_recognition_manifest(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recognizer = _load_recognizer(args)
    failures: list[dict[str, str]] = []
    with out.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            started = time.perf_counter()
            try:
                images = [cv2.imread(row["image_path"]) for row in batch_rows]
                if any(image is None for image in images):
                    missing = [row["image_path"] for row, image in zip(batch_rows, images, strict=True) if image is None]
                    raise RuntimeError(f"Failed to read images: {missing[:5]}")
                results = recognizer(img_numpy_list=images, batch_num=args.batch_size)
                latency_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(batch_rows))
            except Exception as exc:
                results = [{"text": "", "score": 0.0, "elapse": 0.0} for _ in batch_rows]
                latency_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(batch_rows))
                failures.append({"start": str(start), "error": repr(exc)})
            for row, result in zip(batch_rows, results, strict=True):
                pred = {
                    "checkpoint": args.onnx_model,
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "reference": row["text"],
                    "prediction": str(result.get("text", "")),
                    "score": float(result.get("score", 0.0)),
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
                handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
    if failures:
        write_json(out.with_suffix(".failures.json"), {"count": len(failures), "failures": failures})
    print(f"wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
