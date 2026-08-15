#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from turkicocr.utils import get_asset_root


def _load_metrics(path: str | None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def _metrics_block(metrics: dict) -> str:
    if not metrics:
        return "Results are pending the post-training recognition evaluation run."
    rows = []
    for key in [
        "val_rec_count",
        "val_rec_cer",
        "val_rec_wer",
        "val_rec_chrf",
        "val_rec_rare_char_cer",
        "val_rec_exact_match",
        "val_rec_latency_ms_per_crop",
    ]:
        if key in metrics:
            rows.append(f"| {key} | `{metrics[key]}` |")
    if not rows:
        return "```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```"
    return "| Metric | Value |\n|---|---:|\n" + "\n".join(rows)


def build_card(metrics: dict) -> str:
    return f"""---
license: apache-2.0
base_model: OpenOCR/SVTRv2-B
library_name: openocr
pipeline_tag: image-to-text
language:
  - kk
  - ky
  - ru
tags:
  - ocr
  - text-recognition
  - turkicocr
  - svtrv2
  - openocr
  - kazakh
  - kyrgyz
  - cyrillic
datasets:
  - alenisaw/turkicocr-cyrillic
metrics:
  - cer
  - wer
  - chrf
model-index:
  - name: TurkicOCR-SVTRv2-B
    results: []
---

# TurkicOCR-SVTRv2-B

TurkicOCR-SVTRv2-B is an SVTRv2-B/OpenOCR-based OCR recognizer adapted for Kazakh, Kyrgyz, Russian, and mixed Turkic Cyrillic text from line, zone, and table-cell crops.

This is an independent project. It is not an official OpenOCR, PaddleOCR, or PaddlePaddle release.

## Model Details

| Field | Value |
|---|---|
| Base model | `OpenOCR/SVTRv2-B` |
| Training dataset | `alenisaw/turkicocr-cyrillic`, `large` |
| Training strategy | OpenOCR recognizer fine-tuning |
| Active run | `turkicocr_svtrv2_b_rec` |
| Languages | Kazakh, Kyrgyz, Russian, RU-KZ mixed, RU-KY mixed |

## Evaluation

{_metrics_block(metrics)}

## Reproducibility

```bash
python scripts/export_recognition_crops.py --dataset alenisaw/turkicocr-cyrillic --config large --out-dir ./assets/outputs/recognition --charset-out ./assets/outputs/recognition/charset_turkic_cyrillic.txt --include line,zone,table_cell --resume
python scripts/build_recognition_diagnostic.py --validation ./assets/outputs/recognition/manifests/val_rec_large.jsonl --out-dir ./assets/outputs/recognition/manifests --force
python scripts/train_svtrv2_b_rec.py --config configs/train_svtrv2_b_rec.yaml --asset-root ./assets --run-name turkicocr_svtrv2_b_rec --launch
```

## Attribution

- Base recognizer: SVTRv2-B/OpenOCR.
- Training dataset: `alenisaw/turkicocr-cyrillic`, CC BY 4.0, DOI `10.57967/hf/9255`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the TurkicOCR-SVTRv2-B model card.")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--out", default="docs/model_cards/turkicocr-svtrv2-b/README.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_card(_load_metrics(args.metrics)), encoding="utf-8")
    print(f"Model card written to {out}")


if __name__ == "__main__":
    main()
