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


pg.setConfigOptions(imageAxisOrder="row-major")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Phasora")
        self.resize(1400, 850)

        self.open_button = QPushButton("Open TIFF")
        self.open_button.setMinimumHeight(42)
        self.open_button.clicked.connect(self.open_tiff)

        self.status_label = QLabel(
            "Open a multichannel TIFF image to calculate its phasor."
        )
        self.status_label.setWordWrap(True)

        self.graphics = pg.GraphicsLayoutWidget()

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

        self.update_dc_view(dc)
        self.update_phasor_view(dc, g, s)
        self.update_status(
            path=Path(file_name),
            image=image,
            dc=dc,
            g=g,
            s=s,
        )

    def update_dc_view(self, dc: np.ndarray) -> None:
        finite_values = dc[np.isfinite(dc)]

        if finite_values.size == 0:
            lower = 0.0
            upper = 1.0
        else:
            lower, upper = np.percentile(finite_values, [1, 99])

            if lower == upper:
                upper = lower + 1.0

        self.dc_image_item.setImage(
            dc,
            autoLevels=False,
            levels=(float(lower), float(upper)),
        )

        height, width = dc.shape
        self.dc_plot.setRange(
            xRange=(0, width),
            yRange=(0, height),
            padding=0.02,
        )

    def update_phasor_view(
        self,
        dc: np.ndarray,
        g: np.ndarray,
        s: np.ndarray,
    ) -> None:
        valid = (
            (dc > 0)
            & np.isfinite(g)
            & np.isfinite(s)
            & (g >= -1.0)
            & (g <= 1.0)
            & (s >= -1.0)
            & (s <= 1.0)
        )

        if not np.any(valid):
            histogram = np.zeros((256, 256), dtype=np.float64)
        else:
            histogram, _, _ = np.histogram2d(
                g[valid],
                s[valid],
                bins=256,
                range=[[-1.0, 1.0], [-1.0, 1.0]],
            )

        log_histogram = np.log1p(histogram)

        self.phasor_image_item.setImage(
            log_histogram.T,
            autoLevels=True,
        )

        self.phasor_image_item.setRect(
            QRectF(-1.0, -1.0, 2.0, 2.0)
        )

        self.phasor_plot.setXRange(-1.0, 1.0, padding=0)
        self.phasor_plot.setYRange(-1.0, 1.0, padding=0)

    def update_status(
        self,
        path: Path,
        image: np.ndarray,
        dc: np.ndarray,
        g: np.ndarray,
        s: np.ndarray,
    ) -> None:
        nonzero_pixels = dc > 0

        if np.any(nonzero_pixels):
            mean_g = float(np.mean(g[nonzero_pixels]))
            mean_s = float(np.mean(s[nonzero_pixels]))
            mean_dc = float(np.mean(dc[nonzero_pixels]))
        else:
            mean_g = 0.0
            mean_s = 0.0
            mean_dc = 0.0

        height, width, channels = image.shape

        summary = (
            f"File: {path.name} | "
            f"Shape: {height} × {width} × {channels} | "
            f"Mean DC: {mean_dc:.4f} | "
            f"Mean g: {mean_g:.4f} | "
            f"Mean s: {mean_s:.4f}"
        )

        self.status_label.setText(summary)


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())