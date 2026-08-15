from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


def split_tar_image_ref(image_ref: str) -> tuple[str, str] | None:
    if "::" not in image_ref:
        return None
    tar_path, member = image_ref.split("::", 1)
    if not tar_path or not member:
        return None
    return tar_path, member


@contextmanager
def open_image(image_ref: str) -> Iterator[Image.Image]:
    tar_ref = split_tar_image_ref(image_ref)
    if tar_ref is None:
        with Image.open(image_ref) as image:
            yield image.convert("RGB")
        return

    tar_path, member = tar_ref
    with tarfile.open(Path(tar_path)) as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member!r} not found in {tar_path!r}")
        data = extracted.read()
    with Image.open(io.BytesIO(data)) as image:
        yield image.convert("RGB")
