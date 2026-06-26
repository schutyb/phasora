from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class DCImageWidget(QWidget):
    """Display a DC intensity image and a transparent selection overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.graphics = pg.GraphicsLayoutWidget()

        self.plot = self.graphics.addPlot(
            title="Intensity image (DC)",
        )
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")

        self.image_item = pg.ImageItem()
        self.overlay_item = pg.ImageItem()

        self.plot.addItem(self.image_item)
        self.plot.addItem(self.overlay_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def set_image(self, image: np.ndarray) -> None:
        """Display a 2D DC image using robust percentile contrast."""

        array = np.asarray(image)

        if array.ndim != 2:
            raise ValueError(f"DC image must be a 2D array, received shape {array.shape}.")

        finite_values = array[np.isfinite(array)]

        if finite_values.size == 0:
            lower = 0.0
            upper = 1.0
        else:
            lower, upper = np.percentile(
                finite_values,
                [1, 99],
            )

            lower = float(lower)
            upper = float(upper)

            if lower == upper:
                upper = lower + 1.0

        self.image_item.setImage(
            array,
            autoLevels=False,
            levels=(lower, upper),
        )

        height, width = array.shape
        self.plot.setRange(
            xRange=(0, width),
            yRange=(0, height),
            padding=0.02,
        )

    def set_overlay(self, overlay: np.ndarray) -> None:
        """Display an RGBA overlay aligned with the DC image."""

        array = np.asarray(overlay)

        if array.ndim != 3 or array.shape[-1] != 4:
            raise ValueError(
                f"Overlay must have shape (height, width, 4), received shape {array.shape}."
            )

        self.overlay_item.setImage(
            array,
            autoLevels=False,
        )

    def clear_overlay(self, image_shape: Sequence[int]) -> None:
        """Replace the overlay with a transparent RGBA image."""

        if len(image_shape) != 2:
            raise ValueError("Image shape must contain height and width.")

        height = int(image_shape[0])
        width = int(image_shape[1])

        self.set_overlay(
            np.zeros(
                (height, width, 4),
                dtype=np.uint8,
            )
        )
