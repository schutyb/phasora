import numpy as np
import pytest

from phasora.core.visualization import (
    PhasorMode,
    compute_phasor_histogram,
    phasor_plot_ranges,
    universal_semicircle,
)


def test_spectral_ranges_cover_full_phasor_plane() -> None:
    x_range, y_range = phasor_plot_ranges(PhasorMode.SPECTRAL)

    assert x_range == (-1.0, 1.0)
    assert y_range == (-1.0, 1.0)


def test_lifetime_ranges_cover_universal_semicircle() -> None:
    x_range, y_range = phasor_plot_ranges(PhasorMode.LIFETIME)

    assert x_range == (0.0, 1.0)
    assert y_range == (0.0, 0.55)


def test_histogram_has_requested_number_of_bins() -> None:
    g = np.array([[0.1, 0.2], [0.3, 0.4]])
    s = np.array([[0.1, 0.2], [0.3, 0.4]])

    histogram = compute_phasor_histogram(
        g,
        s,
        bins=64,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
    )

    assert histogram.shape == (64, 64)


def test_histogram_requires_matching_shapes() -> None:
    g = np.zeros((2, 2))
    s = np.zeros((3, 3))

    with pytest.raises(ValueError, match="same shape"):
        compute_phasor_histogram(
            g,
            s,
            bins=64,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        )


def test_universal_semicircle_endpoints() -> None:
    g, s = universal_semicircle()

    np.testing.assert_allclose(
        sorted([g[0], g[-1]]),
        [0.0, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [s[0], s[-1]],
        [0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.max(s),
        0.5,
        atol=1e-5,
    )
