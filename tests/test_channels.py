import numpy as np
import pytest

from phasora.core.channels import ChannelGroup, extract_channel_group


def test_extract_channel_group() -> None:
    image = np.zeros((10, 12, 20), dtype=np.float64)

    group = ChannelGroup(
        name="Group A",
        start=4,
        stop=12,
    )

    extracted = extract_channel_group(image, group)

    assert extracted.shape == (10, 12, 8)


def test_group_stop_is_exclusive() -> None:
    image = np.arange(10, dtype=np.float64).reshape(1, 1, 10)

    group = ChannelGroup(
        name="Selection",
        start=2,
        stop=5,
    )

    extracted = extract_channel_group(image, group)

    np.testing.assert_array_equal(
        extracted,
        np.array([[[2.0, 3.0, 4.0]]]),
    )


def test_invalid_group_range_raises_error() -> None:
    image = np.zeros((10, 10, 8), dtype=np.float64)

    group = ChannelGroup(
        name="Invalid",
        start=4,
        stop=12,
    )

    with pytest.raises(ValueError, match="contains only 8 channels"):
        extract_channel_group(image, group)