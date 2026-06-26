import numpy as np
import pytest

from phasora.core.phasor import compute_phasor


def test_uniform_signal_is_at_origin() -> None:
    image = np.ones((4, 5, 8), dtype=np.float64)

    dc, g, s = compute_phasor(image)

    assert dc.shape == (4, 5)
    assert g.shape == (4, 5)
    assert s.shape == (4, 5)

    np.testing.assert_allclose(dc, 8.0)
    np.testing.assert_allclose(g, 0.0, atol=1e-12)
    np.testing.assert_allclose(s, 0.0, atol=1e-12)


def test_signal_in_first_channel_is_at_g_one() -> None:
    image = np.zeros((2, 3, 8), dtype=np.float64)
    image[..., 0] = 10.0

    dc, g, s = compute_phasor(image)

    np.testing.assert_allclose(dc, 10.0)
    np.testing.assert_allclose(g, 1.0)
    np.testing.assert_allclose(s, 0.0, atol=1e-12)


def test_zero_signal_returns_zero_phasor() -> None:
    image = np.zeros((2, 2, 8), dtype=np.float64)

    dc, g, s = compute_phasor(image)

    np.testing.assert_allclose(dc, 0.0)
    np.testing.assert_allclose(g, 0.0)
    np.testing.assert_allclose(s, 0.0)


def test_invalid_shape_raises_error() -> None:
    image = np.ones((10, 10), dtype=np.float64)

    with pytest.raises(ValueError, match="height, width, channels"):
        compute_phasor(image)
