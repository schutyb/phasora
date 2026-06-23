from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def circular_phasor_mask(
    g: NDArray[np.floating],
    s: NDArray[np.floating],
    center_g: float,
    center_s: float,
    radius: float,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """
    Return a boolean mask for pixels inside a circular phasor region.
    """
    if g.shape != s.shape:
        raise ValueError("g and s must have the same shape")

    if radius < 0:
        raise ValueError("radius must be non-negative")

    finite = np.isfinite(g) & np.isfinite(s)

    if valid_mask is not None:
        if valid_mask.shape != g.shape:
            raise ValueError("valid_mask must have the same shape as g and s")
        finite &= valid_mask

    distance_squared = (
        (g - center_g) ** 2
        + (s - center_s) ** 2
    )

    return finite & (distance_squared <= radius**2)