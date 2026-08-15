from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .rare_chars import contains_rare_char
from .recognition_format import load_recognition_manifest


class RecognitionDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        image_loader: Callable[[str], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.records = load_recognition_manifest(manifest_path)
        self.image_loader = image_loader or self._default_image_loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        image = self.image_loader(record["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "text": record["text"], "record": record}

    @staticmethod
    def _default_image_loader(path: str) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install pillow to load recognition crop images.") from exc
        return Image.open(path).convert("RGB")


def build_recognition_dataloader(
    manifest_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Callable[[Any], Any] | None = None,
):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install torch to build a recognition dataloader.") from exc
    dataset = RecognitionDataset(manifest_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def sample_diagnostic_batch(
    manifest_path: str | Path,
    n: int = 256,
    seed: int = 42,
    prefer_rare_chars: bool = True,
) -> list[dict[str, Any]]:
    records = load_recognition_manifest(manifest_path)
    rng = random.Random(seed)
    if n >= len(records):
        return list(records)
    rare = [row for row in records if contains_rare_char(row["text"])]
    other = [row for row in records if not contains_rare_char(row["text"])]
    if not prefer_rare_chars or not rare:
        return rng.sample(records, n)
    rare_n = min(len(rare), max(1, n // 3))
    sampled = rng.sample(rare, rare_n)
    sampled.extend(rng.sample(other, min(len(other), n - rare_n)))
    if len(sampled) < n:
        remaining = [row for row in records if row not in sampled]
        sampled.extend(rng.sample(remaining, n - len(sampled)))
    rng.shuffle(sampled)
    return sampled
