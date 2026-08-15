import importlib.util
from pathlib import Path


def _load_estimator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "estimate_asset_sizes.py"
    spec = importlib.util.spec_from_file_location("estimate_asset_sizes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimator_filters_and_rewrites_asset_root():
    module = _load_estimator()
    cfg = {
        "dataset": {
            "id": "owner/dataset",
            "repo_type": "dataset",
            "local_dir": "external/datasets/owner/dataset",
        },
        "models": [
            {
                "id": "owner/model",
                "role": "baseline",
                "local_dir": "external/models/owner/model",
            },
            {"id": "system", "source": "system package"},
        ],
    }
    rows = module._asset_rows(cfg, "/mnt/assets", include_future=False)
    assert [row["id"] for row in rows] == ["owner/dataset", "owner/model"]
    assert rows[0]["local_dir"] == "/mnt/assets/datasets/owner/dataset"
    assert rows[1]["local_dir"] == "/mnt/assets/models/owner/model"
    assert module._matches_asset(rows[1], ["baseline"])


def test_estimator_suffix_bytes():
    module = _load_estimator()
    files = [
        {"path": "model.safetensors", "size": 10},
        {"path": "data.parquet", "size": 7},
        {"path": "README.md", "size": 3},
    ]
    assert module._suffix_bytes(files, module.WEIGHT_SUFFIXES) == 10
    assert module._suffix_bytes(files, module.DATA_SUFFIXES) == 7
