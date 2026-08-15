import importlib.util
from pathlib import Path


def _load_planner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_asset_download_plan.py"
    spec = importlib.util.spec_from_file_location("make_asset_download_plan", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_plan_training_essentials_phase():
    module = _load_planner()
    size_estimate = {
        "config": "configs/assets.yaml",
        "summary": {"total_estimated_gb": 64.0},
        "assets": [
            {"id": "alenisaw/turkicocr-cyrillic", "status": "estimated", "total_gb": 62.0},
            {"id": "OpenOCR/SVTRv2-B", "status": "estimated", "total_gb": 2.0},
        ],
    }
    plan = module.build_plan(size_estimate, "/mnt/assets")
    phase = next(item for item in plan["phases"] if item["name"] == "training_essentials")
    assert phase["estimated_gb"] == 64.0
    assert phase["cumulative_estimated_gb"] == 64.0
    assert "--asset-root /mnt/assets" in phase["command"]
    assert "--include-weights" in phase["command"]
    assert "--asset 'OpenOCR/SVTRv2-B'" in phase["command"]


def test_download_plan_metadata_phase_command():
    module = _load_planner()
    plan = module.build_plan({"summary": {"total_estimated_gb": 0}, "assets": []}, "/mnt/assets")
    phase = plan["phases"][0]
    assert phase["name"] == "metadata"
    assert "--metadata-only" in phase["command"]
    assert "--include-datasets" in phase["command"]
    assert "--include-benchmarks" in phase["command"]
