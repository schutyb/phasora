from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChannelGroup:
    name: str
    start: int
    stop: int

    def validate(self, n_channels: int) -> None:
        if not self.name.strip():
            raise ValueError("Channel group name cannot be empty")

        if self.start < 0:
            raise ValueError("Channel group start must be non-negative")

        if self.stop <= self.start:
            raise ValueError("Channel group stop must be greater than start")

        if self.stop > n_channels:
            raise ValueError(
                f"Channel group '{self.name}' ends at channel {self.stop}, "
                f"but the image contains only {n_channels} channels"
            )


def extract_channel_group(
    image: NDArray[np.floating],
    group: ChannelGroup,
) -> NDArray[np.float64]:
    """
    Extract a channel range from a channel-last image.

    The interval follows Python slicing convention:
    start is included, stop is excluded.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width, channels)")

    n_channels = image.shape[-1]
    group.validate(n_channels)

    return np.asarray(
        image[..., group.start : group.stop],
        dtype=np.float64,
    )