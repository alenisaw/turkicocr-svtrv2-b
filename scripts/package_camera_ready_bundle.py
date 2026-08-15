#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
from turkicocr.utils import get_asset_root

ROOT = Path(".")
BUNDLE = ROOT / "camera_ready_results_bundle"
CANONICAL = BUNDLE / "canonical"
BASELINES = BUNDLE / "baselines"
DERIVED = BUNDLE / "derived"

def assemble_bundle():
    print("Assembling camera_ready_results_bundle...")
    os.makedirs(CANONICAL, exist_ok=True)
    os.makedirs(BASELINES / "configs", exist_ok=True)
    os.makedirs(BASELINES / "existing_predictions", exist_ok=True)
    os.makedirs(DERIVED, exist_ok=True)
    
    # 1. Copy Canonical Files
    print("Copying canonical files...")
    src_pred = Path(".agent/paper_archive/2026-07-12/assets/recognition_predictions/turkicocr_svtrv2_b_rec_line_v1/release_epoch9/test_line_full.jsonl")
    dest_pred = CANONICAL / "full_test_predictions.jsonl"
    if not dest_pred.exists() or dest_pred.stat().st_size != src_pred.stat().st_size:
        shutil.copy2(src_pred, dest_pred)
        print(f"Copied {dest_pred} ({dest_pred.stat().st_size / 1e6:.1f} MB)")
        
    src_metrics_dir = Path(".agent/paper_archive/2026-07-12/assets/recognition_release_package/turkicocr_svtrv2_b_epoch9_release_20260705/metrics/test_line_full")
    for fname in ["metrics_aggregate.json", "metrics_by_language.json", "by_degradation.json", "by_layout.json", "by_crop_type.json", "recognition_gate_result.json"]:
        if (src_metrics_dir / fname).exists():
            shutil.copy2(src_metrics_dir / fname, CANONICAL / fname)
            
    # Copy additional canonical metrics
    src_clean_metrics = Path(".agent/paper_archive/2026-07-12/assets/benchmark_clean_metrics")
    if (src_clean_metrics / "page_oracle/metrics_aggregate.json").exists():
        shutil.copy2(src_clean_metrics / "page_oracle/metrics_aggregate.json", CANONICAL / "metrics_page_oracle.json")
    if (src_clean_metrics / "zone_from_lines/metrics_aggregate.json").exists():
        shutil.copy2(src_clean_metrics / "zone_from_lines/metrics_aggregate.json", CANONICAL / "metrics_zone_from_lines.json")
    if (src_clean_metrics / "page_detected/layout_aware/henrygagnier__kazakh-ocr/leaderboard.csv").exists():
        shutil.copy2(src_clean_metrics / "page_detected/layout_aware/henrygagnier__kazakh-ocr/leaderboard.csv", CANONICAL / "metrics_page_detected_layout_aware.csv")

    # Copy CSV metrics if present in benchmark_clean_metrics
    src_rec_full = src_clean_metrics / "recognition_line_full"
    if src_rec_full.exists():
        for csv_file in src_rec_full.glob("*.csv"):
            shutil.copy2(csv_file, CANONICAL / csv_file.name)
            
    # 2. Copy Baselines
    print("Copying baseline files...")
    # Configs
    for cfg in ["configs/eval_baselines.yaml", "configs/eval_recognition.yaml", "configs/eval_page_oracle.yaml"]:
        if Path(cfg).exists():
            shutil.copy2(Path(cfg), BASELINES / "configs" / Path(cfg).name)
            
    # Existing predictions & summaries from pipeline_eval
    src_pipe_pred = Path(".agent/paper_archive/2026-07-12/assets/pipeline_eval/alenisaw__turkicocr-cyrillic/predictions/line_models")
    if src_pipe_pred.exists():
        for ffile in src_pipe_pred.glob("*"):
            shutil.copy2(ffile, BASELINES / "existing_predictions" / ffile.name)
            
    # Copy page-level detected baseline predictions
    src_page_pred = Path(".agent/paper_archive/2026-07-12/assets/benchmark_clean_metrics/page_detected/layout_aware/henrygagnier__kazakh-ocr/predictions/line_models")
    if src_page_pred.exists():
        for ffile in src_page_pred.glob("*"):
            if not (BASELINES / "existing_predictions" / ffile.name).exists():
                shutil.copy2(ffile, BASELINES / "existing_predictions" / ffile.name)

    # Generate baseline_summary.csv and baseline_summary.json
    baseline_summary_rows = [
        {
            "model_id": "turkicocr_svtrv2_b_rec",
            "display_name": "TurkicOCR-SVTRv2-B (Torch)",
            "checkpoint": "/mnt/turkicocr-assets/checkpoints/turkicocr_svtrv2_b_rec_line_v1/epoch_9.pth",
            "backend": "openocr_svtrv2",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 21.31,
            "WER_percent": 65.50,
            "chrF": 0.7455,
            "exact_match_SR_percent": 0.40,
        },
        {
            "model_id": "turkicocr_svtrv2_b_int8",
            "display_name": "TurkicOCR-SVTRv2-B (ONNX INT8)",
            "checkpoint": "/mnt/turkicocr-assets/variants/int8/turkicocr-svtrv2-b/model.int8.onnx",
            "backend": "openocr_onnx",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 21.56,
            "WER_percent": 67.29,
            "chrF": 0.7425,
            "exact_match_SR_percent": 0.30,
        },
        {
            "model_id": "kazakh_trocr",
            "display_name": "Kazakh TrOCR",
            "checkpoint": "/mnt/turkicocr-assets/models/thekamilya/kazakh-trocr-fine-tuned",
            "backend": "transformers_vision_encoder_decoder",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 31.96,
            "WER_percent": 72.13,
            "chrF": 0.6408,
            "exact_match_SR_percent": 0.00,
        },
        {
            "model_id": "tesseract",
            "display_name": "Tesseract OCR",
            "checkpoint": "kaz+rus+kir",
            "backend": "tesseract",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 43.52,
            "WER_percent": 74.04,
            "chrF": 0.6413,
            "exact_match_SR_percent": 1.70,
        },
        {
            "model_id": "russian_cyrillic_trocr",
            "display_name": "Russian/Cyrillic TrOCR",
            "checkpoint": "/mnt/turkicocr-assets/models/kazars24/trocr-base-handwritten-ru",
            "backend": "transformers_vision_encoder_decoder",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 61.57,
            "WER_percent": 97.59,
            "chrF": 0.2507,
            "exact_match_SR_percent": 0.00,
        },
        {
            "model_id": "ppocr",
            "display_name": "PP-OCR / PaddleOCR",
            "checkpoint": "PP-OCRv4 server/mobile rec",
            "backend": "ppocr",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 91.91,
            "WER_percent": 95.01,
            "chrF": 0.0497,
            "exact_match_SR_percent": 0.00,
        },
        {
            "model_id": "cyrillic_htr",
            "display_name": "Cyrillic HTR",
            "checkpoint": "/mnt/turkicocr-assets/models/Kansallisarkisto/cyrillic-htr-model",
            "backend": "transformers_vision_encoder_decoder",
            "eval_dataset": "henrygagnier/kazakh-ocr (1,000 pages, layout-aware)",
            "n_samples_or_pages": 1000,
            "CER_percent": 94.31,
            "WER_percent": 96.57,
            "chrF": 0.0399,
            "exact_match_SR_percent": 0.00,
        },
    ]

    with open(BASELINES / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(baseline_summary_rows, f, ensure_ascii=False, indent=2)

    import csv
    with open(BASELINES / "baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_summary_rows)

    print("Wrote baseline summary CSV and JSON.")

if __name__ == "__main__":
    assemble_bundle()
