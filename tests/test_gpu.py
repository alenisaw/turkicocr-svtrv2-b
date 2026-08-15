from pathlib import Path

from turkicocr.gpu import discover_nvidia_gpus_from_proc, resolve_devices


def test_discover_nvidia_gpus_from_proc(tmp_path: Path):
    info = tmp_path / "0000:01:00.0" / "information"
    info.parent.mkdir(parents=True)
    info.write_text(
        "Model: \t\t NVIDIA L4\n"
        "GPU UUID: \t GPU-test\n"
        "Bus Location: \t 0000:01:00.0\n",
        encoding="utf-8",
    )
    devices = discover_nvidia_gpus_from_proc(tmp_path)
    assert devices[0].model == "NVIDIA L4"
    assert devices[0].bus_id == "0000:01:00.0"


def test_resolve_devices_explicit():
    assert resolve_devices("0,2") == ["0", "2"]
