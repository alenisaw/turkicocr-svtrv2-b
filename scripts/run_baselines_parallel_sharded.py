#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from turkicocr.baselines import load_baseline_config
from turkicocr.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run configured OCR baselines sharded in parallel across multiple GPUs.")
    p.add_argument("--config", default="configs/eval_baselines.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=None, help="Override evaluation manifest/glob.")
    p.add_argument("--gpus", default="0,1,2,3", help="Comma-separated list of GPU device IDs to use, e.g. 0,1,2,3.")
    p.add_argument("--max-samples", type=int, default=None, help="Stop after this many manifest rows total.")
    p.add_argument("--model-id", action="append", default=[], help="Run only the selected model id. Repeatable.")
    p.add_argument("--force", action="store_true", help="Regenerate predictions even if files exist.")
    p.add_argument(
        "--development-echo",
        action="store_true",
        help="Write echo predictions to validate pipeline only.",
    )
    return p.parse_args()


def load_manifest_lines(manifest_path: Path, max_samples: int | None) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if max_samples is not None:
        lines = lines[:max_samples]
    return lines


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    out_dir = ensure_dir(args.out)

    # 1. Resolve manifest path
    manifest_override = args.manifest
    if not manifest_override:
        with config_path.open("r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)
            manifest_override = cfg.get("evaluation", {}).get("manifest")
    if not manifest_override:
        raise ValueError(f"Manifest must be specified via --manifest or in {args.config}")
    
    manifest_path = Path(manifest_override)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # 2. Parse GPUs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
    num_gpus = len(gpu_ids)
    if num_gpus == 0:
        raise ValueError("At least one GPU must be specified in --gpus.")

    print(f"Loading manifest: {manifest_path}")
    lines = load_manifest_lines(manifest_path, args.max_samples)
    total_samples = len(lines)
    print(f"Total samples to process: {total_samples}")

    if total_samples == 0:
        print("Manifest is empty. Nothing to do.")
        return

    # Check if final merged predictions already exist for all models
    selected_model_ids = set(args.model_id) if args.model_id else None
    models = [
        model
        for model in load_baseline_config(config_path)
        if selected_model_ids is None or model.id in selected_model_ids
    ]
    if not models:
        raise ValueError(f"No models selected from {args.config}: {sorted(selected_model_ids or [])}")
    all_completed = True
    for model in models:
        final_pred_path = out_dir / f"{model.id}_predictions.jsonl"
        if not (final_pred_path.exists() and final_pred_path.stat().st_size > 0):
            all_completed = False
            break

    if all_completed and not args.force:
        print(f"All models already completed in output directory {out_dir}. Skipping sharded parallel execution.")
        final_status_rows = []
        for model in models:
            final_pred_path = out_dir / f"{model.id}_predictions.jsonl"
            status = "development_echo" if args.development_echo else "completed"
            final_status_rows.append({
                "id": model.id,
                "status": status,
                "predictions": str(final_pred_path)
            })
        status_path = out_dir / "baseline_status.json"
        write_json(status_path, {"models": final_status_rows})
        return

    # Adjust number of shards if samples are very few
    num_shards = min(num_gpus, total_samples)
    print(f"Sharding dataset into {num_shards} chunks for GPUs: {[gpu_ids[i] for i in range(num_shards)]}")

    # 3. Create shard manifests
    shards_manifest_dir = ensure_dir(out_dir / "tmp_manifest_shards")
    shard_paths: list[Path] = []
    
    # Calculate slice sizes
    base_size = total_samples // num_shards
    remainder = total_samples % num_shards

    start_idx = 0
    for i in range(num_shards):
        size = base_size + (1 if i < remainder else 0)
        shard_lines = lines[start_idx : start_idx + size]
        start_idx += size

        shard_path = shards_manifest_dir / f"shard_{i}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(shard_lines) + "\n")
        shard_paths.append(shard_path)

    # 4. Launch parallel subprocesses
    processes = []
    shard_out_dirs: list[Path] = []

    for i in range(num_shards):
        gpu_id = gpu_ids[i]
        shard_out = out_dir / f"tmp_shard_{i}_out"
        shard_out_dirs.append(shard_out)
        if args.force and shard_out.exists():
            shutil.rmtree(shard_out)
        shard_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/run_baselines.py",
            "--config", str(config_path),
            "--out", str(shard_out),
            "--manifest", str(shard_paths[i]),
            # CUDA_VISIBLE_DEVICES below already restricts this subprocess to a
            # single physical GPU, so from inside it that GPU is always local
            # ordinal 0 -- NOT gpu_id. Passing gpu:{gpu_id} here caused
            # "invalid device ordinal" on every shard except gpu_id==0.
            "--device", "gpu:0",
            "--isolated",
        ]
        if args.force:
            cmd.append("--force")
        if args.development_echo:
            cmd.append("--development-echo")
        for model_id in args.model_id:
            cmd.extend(["--model-id", model_id])

        # Set CUDA_VISIBLE_DEVICES dynamically
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["FLAGS_selected_gpus"] = "0"
        env["TURKICOCR_ONNX_USE_GPU"] = "1"

        import site
        site_pkgs = site.getsitepackages() + [sys.prefix]
        nvidia_paths = []
        for sp in site_pkgs:
            p1 = Path(sp) / "nvidia/cudnn/lib"
            p2 = Path(sp) / "nvidia/cublas/lib"
            if p1.exists():
                nvidia_paths.append(str(p1))
            if p2.exists():
                nvidia_paths.append(str(p2))
        if nvidia_paths:
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(nvidia_paths) + (f":{old_ld}" if old_ld else "")

        print(f"Launching shard {i} (samples: {len(shard_lines)}) on GPU {gpu_id}...", flush=True)
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

    # Wait for all processes to complete
    print("Waiting for all shards to complete...")
    for idx, proc in enumerate(processes):
        proc.wait()
        if proc.returncode != 0:
            print(f"Error: Shard process {idx} failed with exit code {proc.returncode}")

    # 5. Merge predictions and summaries for each configured model
    models = load_baseline_config(config_path)
    final_status_rows = []

    for model in models:
        model_id = model.id
        final_pred_path = out_dir / f"{model_id}_predictions.jsonl"
        final_summary_path = out_dir / f"{model_id}_predictions.summary.json"

        # If final prediction file already exists and we are not forcing, skip sharded merging
        if final_pred_path.exists() and final_pred_path.stat().st_size > 0 and not args.force:
            status = "development_echo" if args.development_echo else "completed"
            final_status_rows.append({
                "id": model_id,
                "status": status,
                "predictions": str(final_pred_path)
            })
            continue

        # Check if the predictions exist in all shards
        all_shards_ok = True
        shard_predictions = []
        shard_summaries = []

        for i in range(num_shards):
            shard_out = shard_out_dirs[i]
            pred_file = shard_out / f"{model_id}_predictions.jsonl"
            summary_file = shard_out / f"{model_id}_predictions.summary.json"

            if pred_file.exists() and pred_file.stat().st_size > 0:
                shard_predictions.append(pred_file)
            else:
                all_shards_ok = False
                break

            if summary_file.exists():
                shard_summaries.append(summary_file)

        if not all_shards_ok:
            print(f"Model {model_id} failed on one or more shards. Skipping merge for this model.")
            final_status_rows.append({
                "id": model_id,
                "status": "failed",
                "predictions": str(final_pred_path),
                "error": "Failed during sharded parallel execution."
            })
            continue

        # Concatenate prediction files
        print(f"Merging predictions for model: {model_id}")
        with final_pred_path.open("w", encoding="utf-8") as outfile:
            for pred_file in shard_predictions:
                with pred_file.open("r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)

        # Merge summary metrics
        if shard_summaries:
            total_count = 0
            max_elapsed = 0.0
            peak_memory = None
            backend_name = ""
            checkpoint = ""

            for s_file in shard_summaries:
                try:
                    with s_file.open("r", encoding="utf-8") as sf:
                        s_data = json.load(sf)
                        total_count += s_data.get("count", 0)
                        max_elapsed = max(max_elapsed, s_data.get("elapsed_sec", 0.0))
                        mem = s_data.get("peak_cuda_memory_bytes")
                        if mem is not None:
                            if peak_memory is None:
                                peak_memory = mem
                            else:
                                peak_memory = max(peak_memory, mem)
                        backend_name = s_data.get("backend", backend_name)
                        checkpoint = s_data.get("checkpoint", checkpoint)
                except Exception as e:
                    print(f"Warning: Failed to parse summary file {s_file}: {e}")

            latency = round((max_elapsed / total_count) * 1000.0, 3) if total_count else None
            final_summary = {
                "checkpoint": checkpoint,
                "backend": backend_name,
                "device": f"parallel_gpus_{args.gpus}",
                "count": total_count,
                "elapsed_sec": round(max_elapsed, 3),
                "latency_per_page_ms": latency,
                "peak_cuda_memory_bytes": peak_memory,
            }
            write_json(final_summary_path, final_summary)

        status = "development_echo" if args.development_echo else "completed"
        final_status_rows.append({
            "id": model_id,
            "status": status,
            "predictions": str(final_pred_path)
        })

    # Write overall baseline_status.json
    status_path = out_dir / "baseline_status.json"
    write_json(status_path, {"models": final_status_rows})
    print(f"Sharded parallel baseline execution complete. Status written to: {status_path}")

    # 6. Clean up temporary directories
    print("Cleaning up temporary shard files...")
    shutil.rmtree(shards_manifest_dir, ignore_errors=True)
    for shard_out in shard_out_dirs:
        shutil.rmtree(shard_out, ignore_errors=True)


if __name__ == "__main__":
    main()
