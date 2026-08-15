from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPUDevice:
    index: int
    model: str
    bus_id: str
    uuid: str | None = None


def discover_nvidia_gpus_from_proc(
    proc_root: str | Path = "/proc/driver/nvidia/gpus",
) -> list[GPUDevice]:
    root = Path(proc_root)
    if not root.exists():
        return []
    devices: list[GPUDevice] = []
    for idx, info_path in enumerate(sorted(root.glob("*/information"))):
        info = _parse_nvidia_info(info_path)
        devices.append(
            GPUDevice(
                index=idx,
                model=info.get("Model", "NVIDIA GPU"),
                bus_id=info.get("Bus Location", info_path.parent.name),
                uuid=info.get("GPU UUID"),
            )
        )
    return devices


def _parse_nvidia_info(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def resolve_devices(devices: str = "auto") -> list[str]:
    if devices == "auto":
        found = discover_nvidia_gpus_from_proc()
        return [str(gpu.index) for gpu in found] or ["cpu"]
    return [item.strip() for item in devices.split(",") if item.strip()]
