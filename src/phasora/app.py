from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from phasora.core.io import load_tiff
from phasora.core.phasor import compute_phasor
from phasora.core.selection import circular_phasor_mask


pg.setConfigOptions(imageAxisOrder="row-major")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.current_path: Path | None = None
        self.dc: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.s: np.ndarray | None = None

        self.setWindowTitle("Phasora")
        self.resize(1450, 850)

        self.open_button = QPushButton("Open TIFF")
        self.open_button.setMinimumHeight(42)
        self.open_button.clicked.connect(self.open_tiff)

        self.status_label = QLabel(
            "Open a preprocessed multichannel TIFF image."
        )
        self.status_label.setWordWrap(True)

        self.graphics = pg.GraphicsLayoutWidget()

        # Intensity panel
        self.dc_plot = self.graphics.addPlot(
            row=0,
            col=0,
            title="Intensity image (DC)",
        )
        self.dc_plot.setAspectLocked(True)
        self.dc_plot.hideAxis("left")
        self.dc_plot.hideAxis("bottom")

        self.dc_image_item = pg.ImageItem()
        self.dc_plot.addItem(self.dc_image_item)

        # Transparent selection overlay
        self.overlay_item = pg.ImageItem()
        self.dc_plot.addItem(self.overlay_item)

        # Phasor panel
        self.phasor_plot = self.graphics.addPlot(
            row=0,
            col=1,
            title="Phasor histogram",
        )
        self.phasor_plot.setAspectLocked(True)
        self.phasor_plot.setLabel("bottom", "g")
        self.phasor_plot.setLabel("left", "s")
        self.phasor_plot.setXRange(-1.0, 1.0)
        self.phasor_plot.setYRange(-1.0, 1.0)

        self.phasor_image_item = pg.ImageItem()
        self.phasor_plot.addItem(self.phasor_image_item)

        # Interactive circular selection
        self.phasor_roi = pg.CircleROI(
            pos=(-0.1, -0.1),
            size=(0.2, 0.2),
            movable=True,
            resizable=True,
            pen=pg.mkPen(width=2),
        )
        self.phasor_roi.setVisible(False)
        self.phasor_roi.sigRegionChanged.connect(
            self.update_selection
        )
        self.phasor_plot.addItem(self.phasor_roi)

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.graphics, stretch=1)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def open_tiff(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF image",
            "",
            "TIFF images (*.tif *.tiff)",
        )

        if not file_name:
            return

        try:
            image = load_tiff(file_name)
            dc, g, s = compute_phasor(image)
        except (FileNotFoundError, ValueError, OSError) as error:
            QMessageBox.critical(
                self,
                "Unable to open TIFF",
                str(error),
            )
            return

        self.current_path = Path(file_name)
        self.dc = dc
        self.g = g
        self.s = s

        self.update_dc_view()
        self.update_phasor_view()

        self.phasor_roi.setVisible(True)
        self.update_selection()

    def update_dc_view(self) -> None:
        if self.dc is None:
            return

        finite_values = self.dc[np.isfinite(self.dc)]

        if finite_values.size == 0:
            lower = 0.0
            upper = 1.0
        else:
            lower, upper = np.percentile(
                finite_values,
                [1, 99],
            )

            if lower == upper:
                upper = lower + 1.0

        self.dc_image_item.setImage(
            self.dc,
            autoLevels=False,
            levels=(float(lower), float(upper)),
        )

        height, width = self.dc.shape

        self.dc_plot.setRange(
            xRange=(0, width),
            yRange=(0, height),
            padding=0.02,
        )

    def update_phasor_view(self) -> None:
        if self.dc is None or self.g is None or self.s is None:
            return

        valid = (
            (self.dc > 0)
            & np.isfinite(self.g)
            & np.isfinite(self.s)
            & (self.g >= -1.0)
            & (self.g <= 1.0)
            & (self.s >= -1.0)
            & (self.s <= 1.0)
        )

        if np.any(valid):
            histogram, _, _ = np.histogram2d(
                self.g[valid],
                self.s[valid],
                bins=256,
                range=[[-1.0, 1.0], [-1.0, 1.0]],
            )
        else:
            histogram = np.zeros(
                (256, 256),
                dtype=np.float64,
            )

        log_histogram = np.log1p(histogram)

        self.phasor_image_item.setImage(
            log_histogram.T,
            autoLevels=True,
        )

        self.phasor_image_item.setRect(
            QRectF(-1.0, -1.0, 2.0, 2.0)
        )

        self.phasor_plot.setXRange(
            -1.0,
            1.0,
            padding=0,
        )
        self.phasor_plot.setYRange(
            -1.0,
            1.0,
            padding=0,
        )

    def update_selection(self) -> None:
        if self.dc is None or self.g is None or self.s is None:
            return

        roi_position = self.phasor_roi.pos()
        roi_size = self.phasor_roi.size()

        center_g = float(
            roi_position.x() + roi_size.x() / 2.0
        )
        center_s = float(
            roi_position.y() + roi_size.y() / 2.0
        )
        radius = float(
            min(roi_size.x(), roi_size.y()) / 2.0
        )

        valid = (
            (self.dc > 0)
            & np.isfinite(self.g)
            & np.isfinite(self.s)
        )

        mask = circular_phasor_mask(
            g=self.g,
            s=self.s,
            center_g=center_g,
            center_s=center_s,
            radius=radius,
            valid_mask=valid,
        )

        overlay = np.zeros(
            (*mask.shape, 4),
            dtype=np.uint8,
        )

        overlay[mask, 0] = 255
        overlay[mask, 1] = 80
        overlay[mask, 2] = 80
        overlay[mask, 3] = 150

        self.overlay_item.setImage(
            overlay,
            autoLevels=False,
        )

        selected_pixels = int(np.count_nonzero(mask))
        valid_pixels = int(np.count_nonzero(valid))

        if valid_pixels > 0:
            selected_percentage = (
                100.0 * selected_pixels / valid_pixels
            )
        else:
            selected_percentage = 0.0

        file_name = (
            self.current_path.name
            if self.current_path is not None
            else "Unknown"
        )

        self.status_label.setText(
            f"File: {file_name} | "
            f"Selection center: "
            f"g={center_g:.3f}, s={center_s:.3f} | "
            f"Radius: {radius:.3f} | "
            f"Selected pixels: {selected_pixels:,} "
            f"({selected_percentage:.2f}%)"
        )


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())