#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def _onnx_artifact_size(path: Path) -> int:
    total = path.stat().st_size
    sidecar = path.with_name(path.name + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total


def _validate(fp32_model: Path, int8_model: Path, input_shape: tuple[int, int, int, int]) -> dict:
    onnx.checker.check_model(onnx.load(str(int8_model)))
    rng = np.random.default_rng(42)
    dummy = rng.normal(size=input_shape).astype("float32")
    fp32_session = ort.InferenceSession(str(fp32_model), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_model), providers=["CPUExecutionProvider"])
    fp32_out = fp32_session.run(None, {"input": dummy})[0]
    int8_out = int8_session.run(None, {"input": dummy})[0]
    abs_diff = np.abs(fp32_out - int8_out)
    fp32_size = _onnx_artifact_size(fp32_model)
    int8_size = _onnx_artifact_size(int8_model)
    return {
        "status": "PASS",
        "input_shape": list(input_shape),
        "fp32_output_shape": list(fp32_out.shape),
        "int8_output_shape": list(int8_out.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "fp32_size_mb": fp32_size / (1024 * 1024),
        "int8_size_mb": int8_size / (1024 * 1024),
        "size_ratio": int8_size / max(1, fp32_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize an exported TurkicOCR ONNX recognizer.")
    parser.add_argument("--source-onnx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    parser.add_argument(
        "--op-types",
        default="MatMul,Gemm",
        help="Comma-separated op types to quantize. Dynamic ConvInteger is not supported by CPUExecutionProvider.",
    )
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--copy-runtime-from", default=None, help="Directory with config.yaml/charset/export_metadata.json.")
    args = parser.parse_args()

    source = Path(args.source_onnx)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model.int8.onnx"
    op_types = [item.strip() for item in args.op_types.split(",") if item.strip()]
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8,
        op_types_to_quantize=op_types,
    )
    if args.copy_runtime_from:
        runtime_dir = Path(args.copy_runtime_from)
        for name in ("config.yaml", "charset_turkic_cyrillic.txt", "export_metadata.json"):
            src = runtime_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)
    validation = _validate(source, target, (1, 3, args.height, args.width))
    metadata = {
        "source_onnx": str(source.resolve()),
        "quantized_onnx": str(target.resolve()),
        "method": "onnxruntime.quantization.quantize_dynamic",
        "weight_type": args.weight_type,
        "op_types_to_quantize": op_types,
        "validation": validation,
    }
    (out_dir / "quantization_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
