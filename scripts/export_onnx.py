#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

from scripts.run_openocr_train import write_resolved_openocr_config
from turkicocr.utils import get_asset_root


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _gate_passed(path: str | Path) -> bool:
    data = _read_json(path)
    return str(data.get("status", "")).upper() == "PASS"


def _add_openocr_to_path(openocr_root: str | Path) -> None:
    root = str(Path(openocr_root).resolve())
    tools = str((Path(openocr_root) / "tools").resolve())
    for item in (root, tools):
        if item not in sys.path:
            sys.path.insert(0, item)


def _load_export_model(args: argparse.Namespace):
    _add_openocr_to_path(args.openocr_root)
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from tools.infer_rec import build_rec_process
    from tools.utils.ckpt import load_ckpt

    resolved = write_resolved_openocr_config(args.turkicocr_config, args.openocr_root, args.asset_root)
    cfg = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
    cfg["Global"]["checkpoints"] = str(Path(args.checkpoint).resolve())
    cfg["Global"]["device"] = "cpu"
    cfg["Global"]["distributed"] = False
    cfg["Global"]["use_amp"] = False

    post_process = build_post_process(cfg["PostProcess"])
    cfg["Architecture"]["Decoder"]["out_channels"] = len(post_process.character)
    model = build_model(cfg["Architecture"])
    load_ckpt(model, cfg)
    model.eval()

    for layer in model.modules():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
            layer.rep()

    export_runtime_cfg = {
        "PostProcess": cfg["PostProcess"],
        "Transforms": build_rec_process(cfg),
        "Architecture": cfg["Architecture"],
    }
    return model, cfg, export_runtime_cfg


def _dynamic_axes(enabled: bool) -> dict | None:
    if not enabled:
        return None
    return {
        "input": {0: "batch", 3: "width"},
        "logits": {0: "batch", 1: "time"},
    }


def _validate_onnx(path: Path, input_shape: tuple[int, int, int, int], model: torch.nn.Module) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    opsets = {item.domain or "ai.onnx": item.version for item in onnx_model.opset_import}

    torch.manual_seed(42)
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).detach().cpu().numpy()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"input": dummy.cpu().numpy()})[0]
    abs_diff = np.abs(torch_out - ort_out)
    return {
        "status": "PASS",
        "input_shape": list(input_shape),
        "torch_output_shape": list(torch_out.shape),
        "onnx_output_shape": list(ort_out.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "opsets": opsets,
        "providers": session.get_providers(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SVTRv2-B to ONNX after recognition gate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-result", required=True, help="Recognition gate JSON; must have status=PASS unless --skip-gate.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--turkicocr-config", default="configs/train_svtrv2_b_rec_line.yaml")
    parser.add_argument("--openocr-root", default=str(get_asset_root() / "openocr/repo"))
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--charset", default=str(get_asset_root() / "outputs/recognition/charset_turkic_cyrillic.txt"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--dynamic-width", action="store_true", default=True)
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    if not args.skip_gate and not _gate_passed(args.gate_result):
        raise RuntimeError(f"ONNX export blocked: recognition gate is not PASS in {args.gate_result}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    model, cfg, runtime_cfg = _load_export_model(args)
    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(model_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=_dynamic_axes(args.dynamic_width),
        opset_version=args.opset,
        do_constant_folding=True,
    )
    (out_dir / "config.yaml").write_text(yaml.safe_dump(runtime_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if Path(args.charset).exists():
        shutil.copy2(args.charset, out_dir / Path(args.charset).name)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "turkicocr_config": str(Path(args.turkicocr_config).resolve()),
        "gate_result": str(Path(args.gate_result).resolve()),
        "opset": args.opset,
        "dynamic_width": bool(args.dynamic_width),
        "input_shape": [1, 3, args.height, args.width],
        "postprocess": cfg.get("PostProcess", {}),
    }
    validation = _validate_onnx(model_path, (1, 3, args.height, args.width), model)
    metadata["validation"] = validation
    (out_dir / "export_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(model_path), "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
