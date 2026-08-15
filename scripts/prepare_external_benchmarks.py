#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from turkicocr.utils import ensure_dir, read_yaml, write_json, write_jsonl
from turkicocr.utils import get_asset_root

# Set max field size for CSV parsing
csv.field_size_limit(sys.maxsize)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SFT-style manifests for external OCR benchmark datasets locally."
    )
    parser.add_argument("--config", default="configs/assets.yaml")
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--dataset", default=None, help="Only prepare this dataset ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    asset_root = Path(args.asset_root).resolve()
    out_root = ensure_dir(args.out or asset_root / "outputs" / "external_benchmarks" / "manifests")
    image_root = ensure_dir(asset_root / "outputs" / "external_benchmarks" / "images")
    report: dict[str, Any] = {"benchmarks": []}

    for bench in cfg.get("external_benchmarks", []):
        dataset_id = str(bench["id"])
        if args.dataset and dataset_id != args.dataset:
            continue
        name = dataset_id.replace("/", "__")
        manifest = out_root / f"{name}.jsonl"
        images_dir = ensure_dir(image_root / name)
        local_dir = Path(bench["local_dir"])

        row_report: dict[str, Any] = {
            "id": dataset_id,
            "manifest": str(manifest),
            "status": "pending",
            "rows": 0,
        }

        print(f"Preparing {dataset_id} from {local_dir}...")

        try:
            rows = []
            if dataset_id == "henrygagnier/kazakh-ocr":
                # Path: local_dir / "metadata.csv"
                metadata_path = local_dir / "metadata.csv"
                if not metadata_path.exists():
                    raise FileNotFoundError(f"Metadata file {metadata_path} not found")
                
                with open(metadata_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    idx = 0
                    for row in reader:
                        if row.get("script") != "cyrillic":
                            continue
                        if idx >= args.max_samples:
                            break
                        
                        img_path = local_dir / row["image_path"]
                        if not img_path.exists():
                            continue
                            
                        # Use shutil.copy for fast binary copy
                        out_img = images_dir / f"image_{idx:06d}.png"
                        shutil.copy(img_path, out_img)
                            
                        text = row["text"]
                        rows.append({
                            "image_info": [{"image_url": str(out_img), "matched_text_index": 0}],
                            "text_info": [
                                {"text": "OCR:", "tag": "mask"},
                                {"text": str(text), "tag": "no_mask"},
                            ],
                            "metadata": {
                                "benchmark": dataset_id,
                                "source_index": idx,
                                "language": "kazakh",
                                "layout_type": "unknown",
                                "degradation_profile": "unknown",
                            },
                        })
                        idx += 1

            elif dataset_id == "alenisaw/turkicocr-cyrillic":
                import tarfile

                import pandas as pd
                parquet_path = local_dir / "indexes" / "large" / "validation.parquet"
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Parquet file {parquet_path} not found")
                
                df = pd.read_parquet(parquet_path)
                
                tar_cache = {}
                idx = 0
                for _, row in df.iterrows():
                    if idx >= args.max_samples:
                        break
                    
                    tar_rel = row["tar_path"]
                    tar_p = local_dir / tar_rel
                    img_fn = row["image_filename"]
                    text = row["ocr_text"]
                    
                    if tar_p not in tar_cache:
                        if not tar_p.exists():
                            continue
                        tar_cache[tar_p] = tarfile.open(tar_p)
                    
                    tf = tar_cache[tar_p]
                    try:
                        member = tf.getmember(img_fn)
                        f_img = tf.extractfile(member)
                        if f_img is None:
                            continue
                        
                        ext = os.path.splitext(img_fn)[1] or ".png"
                        out_img = images_dir / f"image_{idx:06d}{ext}"
                        with open(out_img, "wb") as out_f:
                            out_f.write(f_img.read())
                    except Exception as e:
                        print(f"Error extracting {img_fn} from {tar_p}: {e}")
                        continue
                    
                    rows.append({
                        "image_info": [{"image_url": str(out_img), "matched_text_index": 0}],
                        "text_info": [
                            {"text": "OCR:", "tag": "mask"},
                            {"text": str(text), "tag": "no_mask"},
                        ],
                        "metadata": {
                            "benchmark": dataset_id,
                            "source_index": idx,
                            "language": "kazakh",
                            "layout_type": row.get("layout_id", "unknown"),
                            "degradation_profile": "synthetic_clean",
                        },
                    })
                    idx += 1
                
                for tf in tar_cache.values():
                    tf.close()

            if not rows:
                raise ValueError("No rows were prepared for this benchmark")
                
            write_jsonl(manifest, rows)
            row_report["status"] = "created"
            row_report["rows"] = len(rows)
            print(f"Successfully prepared {len(rows)} samples for {dataset_id}")
            
        except Exception as exc:
            row_report["status"] = "failed"
            row_report["error"] = str(exc)
            print(f"Failed to prepare {dataset_id}: {exc}")
            
        report["benchmarks"].append(row_report)

    write_json(out_root / "manifest_report.json", report)
    print(f"External benchmark manifest report written to {out_root / 'manifest_report.json'}")


if __name__ == "__main__":
    main()
