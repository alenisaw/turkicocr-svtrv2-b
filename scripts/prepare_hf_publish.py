#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from turkicocr.utils import ensure_dir, write_json
from turkicocr.utils import get_asset_root

COPY_NAMES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "trainer_state.json",
    "variant_config.json",
    # OpenOCR ONNX/INT8 export variants.
    "config.yaml",
    "charset_turkic_cyrillic.txt",
    "export_metadata.json",
    "quantization_metadata.json",
}
COPY_SUFFIXES = (".safetensors", ".onnx", ".onnx.data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage trained TurkicOCR model directories for later Hugging Face upload."
    )
    parser.add_argument("--asset-root", default=str(get_asset_root()))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-name", default="turkicocr_svtrv2_b_rec_line_v1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--namespace", default="alenisaw")
    parser.add_argument("--include-variants", action="store_true")
    return parser.parse_args()


def _latest_checkpoint(run_dir: Path) -> Path:
    for name in ("best.pth", "latest.pth"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate.resolve()
    checkpoints = sorted(run_dir.glob("epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    return checkpoints[-1].resolve()


def _copy_publish_files(src: Path, dst: Path) -> list[dict[str, str | int]]:
    copied = []
    ensure_dir(dst)
    if src.is_file():
        target = dst / src.name
        shutil.copy2(src, target)
        return [{"source": str(src), "target": str(target), "size_bytes": target.stat().st_size}]
    for item in sorted(src.iterdir()):
        if item.name not in COPY_NAMES and not item.name.endswith(COPY_SUFFIXES):
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(
            {
                "source": str(item),
                "target": str(target),
                "size_bytes": target.stat().st_size if target.is_file() else 0,
            }
        )
    return copied


def _write_card(card: Path, dst: Path) -> None:
    if not card.exists():
        raise FileNotFoundError(f"Model card not found: {card}")
    shutil.copy2(card, dst / "README.md")
    for png in card.parent.glob("*.png"):
        shutil.copy2(png, dst / png.name)


def _stage_one(
    source: Path,
    card: Path,
    out_root: Path,
    repo_id: str,
    kind: str,
) -> dict[str, object]:
    target = ensure_dir(out_root / repo_id.replace("/", "__"))
    copied = _copy_publish_files(source, target)
    _write_card(card, target)
    manifest = {
        "repo_id": repo_id,
        "kind": kind,
        "source": str(source),
        "card": str(card),
        "target": str(target),
        "files": copied,
        "upload_command": f"hf upload {repo_id} {target} . --repo-type model",
    }
    write_json(target / "publish_manifest.json", manifest)
    return manifest


VARIANT_LAYOUT = (
    # (asset subdir under variants/, model card folder name / repo suffix)
    ("onnx", "turkicocr-svtrv2-b-onnx"),
    ("int8", "turkicocr-svtrv2-b-int8"),
)


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    run_dir = asset_root / "checkpoints" / args.run_name
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else _latest_checkpoint(run_dir)
    out_root = ensure_dir(args.out or asset_root / "publish" / "hf")

    staged: list[dict[str, object]] = []
    staged.append(
        _stage_one(
            source=checkpoint,
            card=repo_root / "docs" / "model_cards" / "turkicocr-svtrv2-b" / "README.md",
            out_root=out_root,
            repo_id=f"{args.namespace}/turkicocr-svtrv2-b",
            kind="svtrv2_b_openocr_checkpoint",
        )
    )

    collection_dir = ensure_dir(out_root / f"{args.namespace}__turkicocr-svtrv2-b-collection-card")
    collection_card = repo_root / "docs" / "model_cards" / "collection" / "README.md"
    _write_card(collection_card, collection_dir)
    collection_manifest = {
        "repo_id": f"{args.namespace}/turkicocr-svtrv2-b-collection",
        "kind": "collection_card",
        "target": str(collection_dir),
    }
    write_json(collection_dir / "publish_manifest.json", collection_manifest)
    staged.append(collection_manifest)

    if args.include_variants:
        for subdir, card_name in VARIANT_LAYOUT:
            variant_dir = asset_root / "variants" / subdir / "turkicocr-svtrv2-b"
            if not variant_dir.exists():
                print(f"WARNING: variant directory not found, skipping: {variant_dir}")
                continue
            staged.append(
                _stage_one(
                    source=variant_dir,
                    card=repo_root / "docs" / "model_cards" / card_name / "README.md",
                    out_root=out_root,
                    repo_id=f"{args.namespace}/{card_name}",
                    kind="variant",
                )
            )

    write_json(out_root / "publish_index.json", {"staged": staged})
    print(f"Staged {len(staged)} publish target(s) under {out_root}")


if __name__ == "__main__":
    main()
