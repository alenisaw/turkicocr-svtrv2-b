#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply TurkicOCR runtime compatibility patches to OpenOCR.")
    parser.add_argument("--openocr-root", default=str(get_asset_root() / "openocr/repo"))
    return parser.parse_args()


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.openocr_root)
    trainer = root / "tools" / "engine" / "trainer.py"
    train_rec = root / "tools" / "train_rec.py"

    changed = []
    if replace_once(
        trainer,
        "torch.distributed.init_process_group(backend='nccl')",
        "distributed_backend = os.environ.get('OPENOCR_DISTRIBUTED_BACKEND', 'nccl')\n"
        "            torch.distributed.init_process_group(backend=distributed_backend)",
    ):
        changed.append(str(trainer))

    if replace_once(
        trainer,
        "self.model, [self.local_rank], find_unused_parameters=False)",
        "self.model, [self.local_rank],\n"
        "                find_unused_parameters=os.environ.get('OPENOCR_FIND_UNUSED_PARAMETERS', 'true').lower() == 'true')",
    ):
        changed.append(str(trainer))

    if replace_once(
        train_rec,
        "    args = parser.parse_args()\n    return args\n",
        "    parser.add_argument(\n"
        "        '--no-eval',\n"
        "        dest='eval',\n"
        "        action='store_false',\n"
        "        help='Disable evaluation during train',\n"
        "    )\n"
        "    args = parser.parse_args()\n"
        "    return args\n",
    ):
        changed.append(str(train_rec))

    infer_rec = root / "tools" / "infer_rec.py"
    if replace_once(
        infer_rec,
        "        if algorithm_name in ['SVTRv2_mobile', 'SVTRv2_server']:\n"
        "            self.cfg['Global']['character_dict_path'] = DEFAULT_DICT_PATH_REC\n",
        "        if algorithm_name in ['SVTRv2_mobile', 'SVTRv2_server'] and not self.cfg['Global'].get('character_dict_path'):\n"
        "            self.cfg['Global']['character_dict_path'] = DEFAULT_DICT_PATH_REC\n",
    ):
        changed.append(str(infer_rec))

    onnx_engine = root / "tools" / "infer" / "onnx_engine.py"
    if replace_once(
        onnx_engine,
        "            providers = [\n"
        "                'TensorrtExecutionProvider',\n"
        "                'CUDAExecutionProvider',\n"
        "                'CPUExecutionProvider',\n"
        "            ]",
        "            providers = [\n"
        "                p for p in [\n"
        "                    'TensorrtExecutionProvider',\n"
        "                    'CUDAExecutionProvider',\n"
        "                    'CPUExecutionProvider',\n"
        "                ]\n"
        "                if os.environ.get('OPENOCR_ONNX_SKIP_TENSORRT', '0') != '1' or p != 'TensorrtExecutionProvider'\n"
        "            ]",
    ):
        changed.append(str(onnx_engine))

    print({"openocr_root": str(root), "changed": changed})


if __name__ == "__main__":
    main()
