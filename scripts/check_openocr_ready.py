#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from turkicocr.utils import write_json
from turkicocr.utils import get_asset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OpenOCR/SVTRv2-B readiness.")
    parser.add_argument("--openocr-root", default=str(get_asset_root() / "openocr/repo"))
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--charset")
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    openocr_root = Path(args.openocr_root)
    asset_root = Path(args.asset_root)
    model_dir = asset_root / "models" / "OpenOCR" / "SVTRv2-B"
    checks = {
        "openocr_root_exists": openocr_root.exists(),
        "train_rec_exists": (openocr_root / "tools" / "train_rec.py").exists(),
        "svtrv2_config_exists": (openocr_root / "configs" / "rec" / "svtrv2" / "svtrv2_ch.yml").exists(),
        "model_dir_exists": model_dir.exists(),
        "svtrv2_weight_exists": any(model_dir.glob("*svtrv2*.pth")),
        "openocr_import": importlib.util.find_spec("openocr") is not None or importlib.util.find_spec("openrec") is not None,
        "torch_import": importlib.util.find_spec("torch") is not None,
    }
    if args.charset:
        checks["charset_exists"] = Path(args.charset).exists()
    proc_gpus = sorted(Path("/proc/driver/nvidia/gpus").glob("*/information"))
    checks["gpu_count_from_proc"] = len(proc_gpus)
    checks["nvidia_smi_healthy"] = False
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        checks["nvidia_smi_healthy"] = True
    except Exception:
        pass
    ok = all(value for key, value in checks.items() if key not in {"nvidia_smi_healthy", "openocr_import"})
    report = {"ok": ok, "checks": checks, "openocr_root": str(openocr_root), "model_dir": str(model_dir)}
    out = Path(args.out) if args.out else asset_root / "outputs" / "audit" / "openocr_ready.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
