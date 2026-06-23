from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray


def load_tiff(path: str | Path) -> NDArray[np.float64]:
    """
    Load a TIFF image and return it as float64.

    Expected output shape:
        (height, width, channels)

    Channel-first stacks are automatically moved to channel-last format.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {file_path}")

    if file_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("File must have a .tif or .tiff extension")

    image = np.asarray(tifffile.imread(file_path))

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D TIFF stack, received shape {image.shape}"
        )

    # Typical microscopy stack: channels, height, width
    if image.shape[0] < image.shape[-1] and image.shape[0] < image.shape[1]:
        image = np.moveaxis(image, 0, -1)

    return image.astype(np.float64, copy=False)