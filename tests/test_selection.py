import numpy as np
import pytest

from phasora.core.selection import circular_phasor_mask


def test_circular_mask_selects_points_inside_radius() -> None:
    g = np.array([[0.0, 0.5], [1.0, 0.0]])
    s = np.array([[0.0, 0.0], [0.0, 1.0]])

    mask = circular_phasor_mask(
        g=g,
        s=s,
        center_g=0.0,
        center_s=0.0,
        radius=0.5,
    )

    expected = np.array([[True, True], [False, False]])

    np.testing.assert_array_equal(mask, expected)


def test_valid_mask_is_applied() -> None:
    g = np.zeros((2, 2))
    s = np.zeros((2, 2))
    valid = np.array([[True, False], [True, False]])

    mask = circular_phasor_mask(
        g=g,
        s=s,
        center_g=0.0,
        center_s=0.0,
        radius=1.0,
        valid_mask=valid,
    )

    np.testing.assert_array_equal(mask, valid)


def test_negative_radius_raises_error() -> None:
    g = np.zeros((2, 2))
    s = np.zeros((2, 2))

    with pytest.raises(ValueError, match="non-negative"):
        circular_phasor_mask(
            g=g,
            s=s,
            center_g=0.0,
            center_s=0.0,
            radius=-0.1,
        )


def test_mismatched_shapes_raise_error() -> None:
    g = np.zeros((2, 2))
    s = np.zeros((3, 3))

    with pytest.raises(ValueError, match="same shape"):
        circular_phasor_mask(
            g=g,
            s=s,
            center_g=0.0,
            center_s=0.0,
            radius=0.5,
        )
