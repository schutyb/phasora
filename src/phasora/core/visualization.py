from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class PhasorMode(StrEnum):
    SPECTRAL = "Spectral"
    LIFETIME = "Lifetime"


def phasor_plot_ranges(
    mode: PhasorMode,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return the standard display ranges for each phasor mode.
    """
    if mode == PhasorMode.SPECTRAL:
        return (-1.0, 1.0), (-1.0, 1.0)

    if mode == PhasorMode.LIFETIME:
        return (0.0, 1.0), (0.0, 0.55)

    raise ValueError(f"Unsupported phasor mode: {mode}")


def compute_phasor_histogram(
    g: NDArray[np.floating],
    s: NDArray[np.floating],
    *,
    bins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    valid_mask: NDArray[np.bool_] | None = None,
    log_scale: bool = True,
) -> NDArray[np.float64]:
    """
    Compute a 2D histogram in phasor coordinates.

    The returned array is transposed so it can be displayed directly
    with a row-major ImageItem.
    """
    if g.shape != s.shape:
        raise ValueError("g and s must have the same shape")

    if bins < 8:
        raise ValueError("bins must be at least 8")

    valid = np.isfinite(g) & np.isfinite(s)

    if valid_mask is not None:
        if valid_mask.shape != g.shape:
            raise ValueError(
                "valid_mask must have the same shape as g and s"
            )
        valid &= valid_mask

    valid &= (
        (g >= x_range[0])
        & (g <= x_range[1])
        & (s >= y_range[0])
        & (s <= y_range[1])
    )

    histogram, _, _ = np.histogram2d(
        g[valid],
        s[valid],
        bins=bins,
        range=[x_range, y_range],
    )

    histogram = histogram.astype(np.float64, copy=False)

    if log_scale:
        histogram = np.log1p(histogram)

    return histogram.T


def universal_semicircle(
    number_of_points: int = 512,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return the standard lifetime universal semicircle.

    The circle has center (0.5, 0) and radius 0.5.
    """
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least 2")

    theta = np.linspace(0.0, np.pi, number_of_points)

    g = 0.5 + 0.5 * np.cos(theta)
    s = 0.5 * np.sin(theta)

    return g, s