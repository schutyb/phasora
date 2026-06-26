from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_phasor(
    image: NDArray[np.floating],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Compute DC, g, and s phasor components for a multichannel image.

    Parameters
    ----------
    image:
        Array with shape (height, width, channels).

    Returns
    -------
    dc:
        Sum of intensities across channels.
    g:
        Real phasor component.
    s:
        Imaginary phasor component.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, channels)")

    n_channels = image.shape[-1]

    if n_channels < 2:
        raise ValueError("image must contain at least two channels")

    data = np.asarray(image, dtype=np.float64)

    angles = 2.0 * np.pi * np.arange(n_channels) / n_channels
    cos_weights = np.cos(angles)
    sin_weights = np.sin(angles)

    dc = np.sum(data, axis=-1)

    weighted_cos = np.sum(data * cos_weights, axis=-1)
    weighted_sin = np.sum(data * sin_weights, axis=-1)

    g = np.divide(
        weighted_cos,
        dc,
        out=np.zeros_like(weighted_cos),
        where=dc != 0,
    )

    s = np.divide(
        weighted_sin,
        dc,
        out=np.zeros_like(weighted_sin),
        where=dc != 0,
    )

    return dc, g, s
