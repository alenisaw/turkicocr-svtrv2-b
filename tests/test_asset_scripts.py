import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_assets_filter_and_asset_root():
    module = _load_script("prepare_assets.py")
    cfg = {
        "dataset": {
            "id": "owner/dataset",
            "local_dir": "external/datasets/owner/dataset",
        },
        "models": [
            {
                "id": "OpenOCR/SVTRv2-B",
                "role": "detector_backbone",
                "local_dir": "external/models/OpenOCR/SVTRv2-B",
            },
            {
                "id": "alenisaw/turkicocr-svtrv2-b",
                "role": "ours",
                "local_dir": "checkpoints/turkicocr_svtrv2_b_rec",
            },
        ],
    }
    args = SimpleNamespace(
        include_datasets=True,
        include_benchmarks=False,
        include_future_checkpoint=False,
        asset=["OpenOCR/*"],
        asset_root="/mnt/turkicocr-assets",
    )
    plan = module._asset_plan(cfg, args)
    assert len(plan) == 1
    assert plan[0]["id"] == "OpenOCR/SVTRv2-B"
    assert plan[0]["local_dir"] == (
        "/mnt/turkicocr-assets/models/OpenOCR/SVTRv2-B"
    )


def test_check_assets_asset_root_rewrite():
    module = _load_script("check_assets.py")
    assert module._local_dir_for_asset(
        "external/datasets/owner/dataset", "/mnt/turkicocr-assets"
    ) == "/mnt/turkicocr-assets/datasets/owner/dataset"
    assert module._local_dir_for_asset(
        "checkpoints/model", "/mnt/turkicocr-assets"
    ) == "checkpoints/model"
