#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _artifact_size_mb(path: Path) -> float:
    total = path.stat().st_size
    sidecar = path.with_name(path.name + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total / (1024 * 1024)


def _bench(path: Path, shape: tuple[int, int, int, int], warmup: int, iters: int, providers: list[str]) -> dict:
    session = ort.InferenceSession(str(path), providers=providers)
    rng = np.random.default_rng(42)
    sample = rng.normal(size=shape).astype("float32")
    for _ in range(warmup):
        session.run(None, {"input": sample})
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        output = session.run(None, {"input": sample})[0]
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "model": str(path.resolve()),
        "providers": session.get_providers(),
        "input_shape": list(shape),
        "output_shape": list(output.shape),
        "artifact_size_mb": _artifact_size_mb(path),
        "iterations": iters,
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exported ONNX OCR variants.")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--provider", action="append", default=["CPUExecutionProvider"])
    args = parser.parse_args()

    shape = (args.batch_size, 3, args.height, args.width)
    rows = [_bench(Path(model), shape, args.warmup, args.iters, args.provider) for model in args.model]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmarks": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"benchmarks": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
