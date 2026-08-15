#!/usr/bin/env python3
"""Compute derived CPU-only analyses for TurkicOCR-SVTRv2-B camera-ready submission package."""

import json
import math
import os
import random
import shutil
import zipfile
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import rapidfuzz
from sklearn.metrics import roc_auc_score

PRED_PATH = Path(".agent/paper_archive/2026-07-12/assets/recognition_predictions/turkicocr_svtrv2_b_rec_line_v1/release_epoch9/test_line_full.jsonl")
BUNDLE_DIR = Path("camera_ready_results_bundle")
CANONICAL_DIR = BUNDLE_DIR / "canonical"
BASELINES_DIR = BUNDLE_DIR / "baselines"
DERIVED_DIR = BUNDLE_DIR / "derived"

RARE_CHARS_PAIRS = [
    ("Ә/ә", ["ә"]),
    ("Ғ/ғ", ["ғ"]),
    ("Қ/қ", ["қ"]),
    ("Ң/ң", ["ң"]),
    ("Ө/ө", ["ө"]),
    ("Ұ/ұ", ["ұ"]),
    ("Ү/ү", ["ү"]),
    ("Һ/һ", ["һ"]),
    ("І/і_or_I/i", ["і", "i"]),
]

CHAR_TO_LABEL = {}
for label, clist in RARE_CHARS_PAIRS:
    for c in clist:
        CHAR_TO_LABEL[c] = label

def normalize_recognition_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split()).strip()

def chrf_score(ref_norm: str, hyp_norm: str, beta: float = 1.0, max_n: int = 6) -> float:
    if ref_norm == hyp_norm:
        return 1.0
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0
        
    scores = []
    beta2 = beta * beta
    ref_l = ref_norm.lower()
    hyp_l = hyp_norm.lower()
    
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ref_l[i:i+n] for i in range(len(ref_l)-n+1)) if len(ref_l) >= n else (Counter([ref_l]) if ref_l else Counter())
        hyp_ngrams = Counter(hyp_l[i:i+n] for i in range(len(hyp_l)-n+1)) if len(hyp_l) >= n else (Counter([hyp_l]) if hyp_l else Counter())
        
        if not ref_ngrams and not hyp_ngrams:
            scores.append(1.0)
            continue
        
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        prec = overlap / max(1, sum(hyp_ngrams.values()))
        rec = overlap / max(1, sum(ref_ngrams.values()))
        denom = beta2 * prec + rec
        scores.append((1 + beta2) * prec * rec / denom if denom else 0.0)
    return sum(scores) / len(scores)

def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% confidence z-score
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin) * 100.0, min(100.0, centre + margin) * 100.0)

def classify_text_length(length: int) -> str:
    if length < 32:
        return "short"
    elif length <= 127:
        return "medium"
    elif length <= 511:
        return "long"
    else:
        return "very_long"

def run_all_analyses():
    print(f"Loading predictions from {PRED_PATH}...")
    
    samples = []
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            
    total_samples = len(samples)
    print(f"Loaded {total_samples:,} samples.")
    
    print("Computing per-sample metrics, edit alignments, and rare char counts...")
    
    sample_records = []
    page_rare_counts = defaultdict(lambda: defaultdict(lambda: {"ref": 0, "err": 0, "del": 0, "sub": 0, "ins": 0}))
    
    for idx, s in enumerate(samples):
        ref = s.get("reference", "")
        pred = s.get("prediction", "")
        score = s.get("score", None)
        meta = s.get("metadata", {}) or {}
        
        ref_norm = normalize_recognition_text(ref)
        pred_norm = normalize_recognition_text(pred)
        ref_len = len(ref_norm)
        pred_len = len(pred_norm)
        
        # Edit distance & ops
        opcodes = rapidfuzz.distance.Levenshtein.opcodes(ref_norm, pred_norm)
        dels = subs = ins = 0
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'delete':
                dels += (i2 - i1)
            elif tag == 'replace':
                subs += (i2 - i1)
            elif tag == 'insert':
                ins += (j2 - j1)
                
        dist = dels + subs + ins
        cer_val = dist / max(1, ref_len)
        
        # Word error
        ref_words = ref_norm.split()
        pred_words = pred_norm.split()
        word_dist = rapidfuzz.distance.Levenshtein.distance(ref_words, pred_words)
        wer_val = word_dist / max(1, len(ref_words))
        
        chrf_val = chrf_score(ref_norm, pred_norm)
        exact = float(ref_norm == pred_norm)
        len_ratio = pred_len / max(1, ref_len)
        is_empty = float(pred_len == 0)
        is_too_short = float(len_ratio < 0.70)
        
        lang = meta.get("language", "unknown")
        layout = meta.get("layout_type", "unknown")
        deg = meta.get("degradation_profile", "unknown")
        crop_type = meta.get("crop_type", "unknown")
        page_id = meta.get("source_page_id", "unknown")
        length_bucket = classify_text_length(ref_len)
        
        # Single-pass rare char alignment check
        ref_lower = ref_norm.lower()
        pred_lower = pred_norm.lower()
        
        for ch in ref_lower:
            if ch in CHAR_TO_LABEL:
                page_rare_counts[page_id][CHAR_TO_LABEL[ch]]["ref"] += 1
                
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'delete':
                for i in range(i1, i2):
                    ch = ref_lower[i]
                    if ch in CHAR_TO_LABEL:
                        lbl = CHAR_TO_LABEL[ch]
                        page_rare_counts[page_id][lbl]["err"] += 1
                        page_rare_counts[page_id][lbl]["del"] += 1
            elif tag == 'replace':
                for i in range(i1, i2):
                    ch = ref_lower[i]
                    if ch in CHAR_TO_LABEL:
                        lbl = CHAR_TO_LABEL[ch]
                        page_rare_counts[page_id][lbl]["err"] += 1
                        page_rare_counts[page_id][lbl]["sub"] += 1
            elif tag == 'insert':
                for j in range(j1, j2):
                    ch = pred_lower[j]
                    if ch in CHAR_TO_LABEL:
                        lbl = CHAR_TO_LABEL[ch]
                        page_rare_counts[page_id][lbl]["ins"] += 1

        rec = {
            "idx": idx,
            "ref_norm": ref_norm,
            "pred_norm": pred_norm,
            "ref_len": ref_len,
            "pred_len": pred_len,
            "score": score,
            "cer": cer_val,
            "wer": wer_val,
            "chrf": chrf_val,
            "exact_match": exact,
            "length_ratio": len_ratio,
            "empty": is_empty,
            "too_short": is_too_short,
            "dels": dels,
            "subs": subs,
            "ins": ins,
            "edit_errors": dist,
            "language": lang,
            "layout_type": layout,
            "degradation_profile": deg,
            "crop_type": crop_type,
            "source_page_id": page_id,
            "text_length_bucket": length_bucket,
        }
        sample_records.append(rec)
        if (idx + 1) % 50000 == 0 or (idx + 1) == total_samples:
            print(f"  Processed {idx + 1:,} / {total_samples:,} samples")

    os.makedirs(DERIVED_DIR, exist_ok=True)

    # A. Degradation
    print("\n--- A. Degradation Analysis ---")
    deg_groups = defaultdict(list)
    for r in sample_records:
        deg_groups[r["degradation_profile"]].append(r)
        
    deg_results = []
    for deg, rows in sorted(deg_groups.items()):
        n = len(rows)
        row_dict = {
            "degradation_profile": deg,
            "N": n,
            "CER": float(np.mean([r["cer"] for r in rows]) * 100),
            "WER": float(np.mean([r["wer"] for r in rows]) * 100),
            "chrF": float(np.mean([r["chrf"] for r in rows])),
            "exact_match_SR": float(np.mean([r["exact_match"] for r in rows]) * 100),
            "empty_rate": float(np.mean([r["empty"] for r in rows]) * 100),
            "too_short_rate": float(np.mean([r["too_short"] for r in rows]) * 100),
        }
        deg_results.append(row_dict)
        
    with open(DERIVED_DIR / "by_degradation_detailed.json", "w", encoding="utf-8") as f:
        json.dump(deg_results, f, ensure_ascii=False, indent=2)
        
    with open(DERIVED_DIR / "by_degradation_detailed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(deg_results[0].keys()))
        writer.writeheader()
        writer.writerows(deg_results)
    print(f"Saved {DERIVED_DIR / 'by_degradation_detailed.csv'}")

    # B. Text length
    print("\n--- B. Text-Length Analysis ---")
    len_groups = defaultdict(list)
    for r in sample_records:
        len_groups[r["text_length_bucket"]].append(r)
        
    bucket_order = ["short", "medium", "long", "very_long"]
    len_results = []
    for b in bucket_order:
        rows = len_groups.get(b, [])
        if not rows:
            continue
        n = len(rows)
        row_dict = {
            "text_length_bucket": b,
            "N": n,
            "mean_ref_length": float(np.mean([r["ref_len"] for r in rows])),
            "CER": float(np.mean([r["cer"] for r in rows]) * 100),
            "WER": float(np.mean([r["wer"] for r in rows]) * 100),
            "chrF": float(np.mean([r["chrf"] for r in rows])),
            "exact_match_SR": float(np.mean([r["exact_match"] for r in rows]) * 100),
            "empty_rate": float(np.mean([r["empty"] for r in rows]) * 100),
            "too_short_rate": float(np.mean([r["too_short"] for r in rows]) * 100),
            "mean_length_ratio": float(np.mean([r["length_ratio"] for r in rows])),
        }
        len_results.append(row_dict)
        
    with open(DERIVED_DIR / "by_text_length.json", "w", encoding="utf-8") as f:
        json.dump(len_results, f, ensure_ascii=False, indent=2)
        
    with open(DERIVED_DIR / "by_text_length.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(len_results[0].keys()))
        writer.writeheader()
        writer.writerows(len_results)
    print(f"Saved {DERIVED_DIR / 'by_text_length.csv'}")

    # C. Edit operations
    print("\n--- C. Edit-Operation Analysis ---")
    tot_dels = sum(r["dels"] for r in sample_records)
    tot_subs = sum(r["subs"] for r in sample_records)
    tot_ins = sum(r["ins"] for r in sample_records)
    tot_errors = tot_dels + tot_subs + tot_ins
    tot_ref_chars = sum(r["ref_len"] for r in sample_records)
    
    global_edit_ops = {
        "sample_count": total_samples,
        "total_reference_chars": tot_ref_chars,
        "total_deletions": tot_dels,
        "total_substitutions": tot_subs,
        "total_insertions": tot_ins,
        "total_edit_errors": tot_errors,
        "deletion_share_of_errors": float(tot_dels / max(1, tot_errors)),
        "substitution_share_of_errors": float(tot_subs / max(1, tot_errors)),
        "insertion_share_of_errors": float(tot_ins / max(1, tot_errors)),
        "deletions_per_ref_char": float(tot_dels / max(1, tot_ref_chars)),
        "substitutions_per_ref_char": float(tot_subs / max(1, tot_ref_chars)),
        "insertions_per_ref_char": float(tot_ins / max(1, tot_ref_chars)),
        "errors_per_ref_char": float(tot_errors / max(1, tot_ref_chars)),
    }
    with open(DERIVED_DIR / "edit_operations_global.json", "w", encoding="utf-8") as f:
        json.dump(global_edit_ops, f, ensure_ascii=False, indent=2)
        
    def compute_edit_breakdown(group_key_func, group_name):
        groups = defaultdict(list)
        for r in sample_records:
            groups[group_key_func(r)].append(r)
            
        rows = []
        for g_val, r_list in sorted(groups.items()):
            g_dels = sum(r["dels"] for r in r_list)
            g_subs = sum(r["subs"] for r in r_list)
            g_ins = sum(r["ins"] for r in r_list)
            g_errs = g_dels + g_subs + g_ins
            g_ref_chars = sum(r["ref_len"] for r in r_list)
            
            rows.append({
                group_name: g_val,
                "N": len(r_list),
                "total_ref_chars": g_ref_chars,
                "total_errors": g_errs,
                "deletions": g_dels,
                "substitutions": g_subs,
                "insertions": g_ins,
                "deletion_share": float(g_dels / max(1, g_errs)),
                "substitution_share": float(g_subs / max(1, g_errs)),
                "insertion_share": float(g_ins / max(1, g_errs)),
                "errors_per_ref_char": float(g_errs / max(1, g_ref_chars)),
            })
        return rows
        
    by_len_ops = compute_edit_breakdown(lambda r: r["text_length_bucket"], "text_length_bucket")
    by_lang_ops = compute_edit_breakdown(lambda r: r["language"], "language")
    by_deg_ops = compute_edit_breakdown(lambda r: r["degradation_profile"], "degradation_profile")
    
    for path, data in [
        (DERIVED_DIR / "edit_operations_by_length.csv", by_len_ops),
        (DERIVED_DIR / "edit_operations_by_language.csv", by_lang_ops),
        (DERIVED_DIR / "edit_operations_by_degradation.csv", by_deg_ops),
    ]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    print("Saved edit operations global and breakdown files.")

    # D. Page cluster bootstrap
    print("\n--- D. Page-Clustered Statistics (1000 Bootstrap Iterations) ---")
    page_data = defaultdict(lambda: {
        "crops": 0,
        "cer_num": 0,
        "cer_den": 0,
        "wer_num": 0,
        "wer_den": 0,
        "chrf_sum": 0.0,
        "exact_match_count": 0,
    })
    
    for r in sample_records:
        p = page_data[r["source_page_id"]]
        p["crops"] += 1
        p["cer_num"] += r["edit_errors"]
        p["cer_den"] += max(1, r["ref_len"])
        
        ref_w_cnt = max(1, len(r["ref_norm"].split()))
        p["wer_num"] += r["wer"] * ref_w_cnt
        p["wer_den"] += ref_w_cnt
        
        p["chrf_sum"] += r["chrf"]
        p["exact_match_count"] += int(r["exact_match"])
        
    page_ids = list(page_data.keys())
    n_pages = len(page_ids)
    print(f"Aggregated {n_pages:,} unique pages.")
    
    page_crops = np.array([page_data[pid]["crops"] for pid in page_ids], dtype=np.float64)
    page_cer_num = np.array([page_data[pid]["cer_num"] for pid in page_ids], dtype=np.float64)
    page_cer_den = np.array([page_data[pid]["cer_den"] for pid in page_ids], dtype=np.float64)
    page_wer_num = np.array([page_data[pid]["wer_num"] for pid in page_ids], dtype=np.float64)
    page_wer_den = np.array([page_data[pid]["wer_den"] for pid in page_ids], dtype=np.float64)
    page_chrf_sum = np.array([page_data[pid]["chrf_sum"] for pid in page_ids], dtype=np.float64)
    page_exact_cnt = np.array([page_data[pid]["exact_match_count"] for pid in page_ids], dtype=np.float64)
    
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_cer = np.zeros(n_boot)
    boot_wer = np.zeros(n_boot)
    boot_chrf = np.zeros(n_boot)
    boot_sr = np.zeros(n_boot)
    
    for b_idx in range(n_boot):
        sample_p_idx = rng.choice(n_pages, size=n_pages, replace=True)
        
        sum_cer_n = np.sum(page_cer_num[sample_p_idx])
        sum_cer_d = np.sum(page_cer_den[sample_p_idx])
        boot_cer[b_idx] = (sum_cer_n / max(1.0, sum_cer_d)) * 100.0
        
        sum_wer_n = np.sum(page_wer_num[sample_p_idx])
        sum_wer_d = np.sum(page_wer_den[sample_p_idx])
        boot_wer[b_idx] = (sum_wer_n / max(1.0, sum_wer_d)) * 100.0
        
        tot_crops = np.sum(page_crops[sample_p_idx])
        boot_chrf[b_idx] = np.sum(page_chrf_sum[sample_p_idx]) / max(1.0, tot_crops)
        boot_sr[b_idx] = (np.sum(page_exact_cnt[sample_p_idx]) / max(1.0, tot_crops)) * 100.0
        
    def summary_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "ci_95_lower": float(np.percentile(arr, 2.5)),
            "ci_95_upper": float(np.percentile(arr, 97.5)),
        }
        
    page_cluster_boot_results = {
        "resampling_unit": "source_page_id",
        "n_pages": n_pages,
        "n_samples": total_samples,
        "n_bootstrap_iterations": n_boot,
        "seed": 42,
        "metrics": {
            "CER_percent": summary_stats(boot_cer),
            "WER_percent": summary_stats(boot_wer),
            "chrF": summary_stats(boot_chrf),
            "SuccessRate_percent": summary_stats(boot_sr),
        },
        "paper_sample_level_ci_comparison": {
            "CER_percent": {"paper_sample_ci": [4.45, 4.56], "page_clustered_ci": [summary_stats(boot_cer)["ci_95_lower"], summary_stats(boot_cer)["ci_95_upper"]]},
            "WER_percent": {"paper_sample_ci": [9.04, 9.24], "page_clustered_ci": [summary_stats(boot_wer)["ci_95_lower"], summary_stats(boot_wer)["ci_95_upper"]]},
            "chrF": {"paper_sample_ci": [0.9370, 0.9384], "page_clustered_ci": [summary_stats(boot_chrf)["ci_95_lower"], summary_stats(boot_chrf)["ci_95_upper"]]},
            "SR_percent": {"paper_sample_ci": [85.88, 86.14], "page_clustered_ci": [summary_stats(boot_sr)["ci_95_lower"], summary_stats(boot_sr)["ci_95_upper"]]},
        }
    }
    with open(DERIVED_DIR / "page_cluster_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(page_cluster_boot_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {DERIVED_DIR / 'page_cluster_bootstrap.json'}")

    # E. Rare character stats
    print("\n--- E. Rare-Character Robustness Analysis ---")
    rare_char_stats = []
    
    for label, char_list in RARE_CHARS_PAIRS:
        tot_ref = sum(page_rare_counts[pid][label]["ref"] for pid in page_ids)
        tot_err = sum(page_rare_counts[pid][label]["err"] for pid in page_ids)
        tot_del = sum(page_rare_counts[pid][label]["del"] for pid in page_ids)
        tot_sub = sum(page_rare_counts[pid][label]["sub"] for pid in page_ids)
        tot_ins = sum(page_rare_counts[pid][label]["ins"] for pid in page_ids)
        
        err_rate = (tot_err / max(1, tot_ref)) * 100.0
        del_share = (tot_del / max(1, tot_err)) * 100.0
        sub_share = (tot_sub / max(1, tot_err)) * 100.0
        
        w_low, w_high = wilson_score_interval(tot_err, tot_ref)
        
        p_refs = np.array([page_rare_counts[pid][label]["ref"] for pid in page_ids], dtype=np.float64)
        p_errs = np.array([page_rare_counts[pid][label]["err"] for pid in page_ids], dtype=np.float64)
        
        rare_boot_err_rates = np.zeros(n_boot)
        for b_idx in range(n_boot):
            sample_p_idx = rng.choice(n_pages, size=n_pages, replace=True)
            b_r = np.sum(p_refs[sample_p_idx])
            b_e = np.sum(p_errs[sample_p_idx])
            rare_boot_err_rates[b_idx] = (b_e / max(1.0, b_r)) * 100.0
            
        boot_ci_low = float(np.percentile(rare_boot_err_rates, 2.5))
        boot_ci_high = float(np.percentile(rare_boot_err_rates, 97.5))
        
        row_dict = {
            "Character": label.split("_")[0],
            "ReferenceCount": tot_ref,
            "ErrorCount": tot_err,
            "ObservedErrorRate_percent": float(err_rate),
            "DeletionCount": tot_del,
            "DeletionShare_percent": float(del_share),
            "SubstitutionCount": tot_sub,
            "SubstitutionShare_percent": float(sub_share),
            "InsertionCount": tot_ins,
            "Wilson_CI_lower": float(w_low),
            "Wilson_CI_upper": float(w_high),
            "PageCluster_Bootstrap_CI_lower": boot_ci_low,
            "PageCluster_Bootstrap_CI_upper": boot_ci_high,
        }
        rare_char_stats.append(row_dict)
        
    with open(DERIVED_DIR / "rare_character_extended.json", "w", encoding="utf-8") as f:
        json.dump(rare_char_stats, f, ensure_ascii=False, indent=2)
        
    with open(DERIVED_DIR / "rare_character_extended.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rare_char_stats[0].keys()))
        writer.writeheader()
        writer.writerows(rare_char_stats)
    print(f"Saved {DERIVED_DIR / 'rare_character_extended.csv'}")

    # F. Confidence Analysis
    print("\n--- F. Confidence Analysis ---")
    scores = np.array([r["score"] for r in sample_records if r["score"] is not None], dtype=np.float64)
    exact_matches = np.array([r["exact_match"] == 1.0 for r in sample_records], dtype=bool)
    cers = np.array([r["cer"] for r in sample_records], dtype=np.float64)
    
    exact_scores = scores[exact_matches]
    err_scores = scores[~exact_matches]
    
    y_true = (~exact_matches).astype(int)
    y_score = 1.0 - scores
    auroc = float(roc_auc_score(y_true, y_score))
    
    cer_b0 = scores[cers == 0.0]
    cer_b1 = scores[(cers > 0.0) & (cers <= 0.05)]
    cer_b2 = scores[(cers > 0.05) & (cers <= 0.10)]
    cer_b3 = scores[(cers > 0.10) & (cers <= 0.20)]
    cer_b4 = scores[cers > 0.20]
    
    n_bins = 10
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        in_bin = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i+1] if i < n_bins - 1 else scores <= bin_boundaries[i+1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(exact_matches[in_bin])
            bin_conf = np.mean(scores[in_bin])
            abs_diff = abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * abs_diff
            bin_details.append({
                "bin": f"[{bin_boundaries[i]:.1f}, {bin_boundaries[i+1]:.1f}]",
                "count": int(bin_size),
                "accuracy_exact_match": float(bin_acc),
                "mean_confidence": float(bin_conf),
                "abs_diff": float(abs_diff),
            })
            
    conf_analysis = {
        "score_available": True,
        "sample_count": total_samples,
        "exact_match_predictions": {
            "count": int(len(exact_scores)),
            "mean_confidence": float(np.mean(exact_scores)),
            "median_confidence": float(np.median(exact_scores)),
            "std_confidence": float(np.std(exact_scores)),
        },
        "erroneous_predictions": {
            "count": int(len(err_scores)),
            "mean_confidence": float(np.mean(err_scores)),
            "median_confidence": float(np.median(err_scores)),
            "std_confidence": float(np.std(err_scores)),
        },
        "confidence_by_cer_buckets": {
            "cer_0_exact_match": {"count": len(cer_b0), "mean_confidence": float(np.mean(cer_b0)) if len(cer_b0) else None, "median_confidence": float(np.median(cer_b0)) if len(cer_b0) else None},
            "cer_0_to_0.05": {"count": len(cer_b1), "mean_confidence": float(np.mean(cer_b1)) if len(cer_b1) else None, "median_confidence": float(np.median(cer_b1)) if len(cer_b1) else None},
            "cer_0.05_to_0.10": {"count": len(cer_b2), "mean_confidence": float(np.mean(cer_b2)) if len(cer_b2) else None, "median_confidence": float(np.median(cer_b2)) if len(cer_b2) else None},
            "cer_0.10_to_0.20": {"count": len(cer_b3), "mean_confidence": float(np.mean(cer_b3)) if len(cer_b3) else None, "median_confidence": float(np.median(cer_b3)) if len(cer_b3) else None},
            "cer_above_0.20": {"count": len(cer_b4), "mean_confidence": float(np.mean(cer_b4)) if len(cer_b4) else None, "median_confidence": float(np.median(cer_b4)) if len(cer_b4) else None},
        },
        "auroc_error_detection": auroc,
        "ece_expected_calibration_error": float(ece),
        "calibration_bins": bin_details,
        "score_interpretation_note": (
            "Model score is calculated by the OpenOCR / SVTRv2 recognizer backend as the sequence path "
            "softmax probability product or geometric mean over decoding steps. While scores strongly correlate "
            "with exact-match correctness (AUROC = {:.4f}), CTC path scores tend to be overconfident (mean confidence "
            "~0.97 for erroneous predictions), making raw ECE ({:.4f}) a reflection of CTC sequence probability skew "
            "rather than well-calibrated posterior confidence probabilities."
        ).format(auroc, ece),
    }
    
    with open(DERIVED_DIR / "confidence_analysis.json", "w", encoding="utf-8") as f:
        json.dump(conf_analysis, f, ensure_ascii=False, indent=2)
    print(f"Saved {DERIVED_DIR / 'confidence_analysis.json'}")

if __name__ == "__main__":
    run_all_analyses()
