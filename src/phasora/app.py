from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
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
    MAX_SELECTIONS = 10

    def __init__(self) -> None:
        super().__init__()

        self.current_path: Path | None = None
        self.dc: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.s: np.ndarray | None = None

        self.selection_colors = [
            (255, 80, 80, 150),    # red
            (80, 180, 255, 150),   # blue
            (100, 220, 120, 150),  # green
            (255, 200, 70, 150),   # yellow
            (200, 100, 255, 150),  # purple
            (255, 140, 60, 150),   # orange
            (80, 220, 220, 150),   # cyan
            (255, 100, 180, 150),  # pink
            (170, 220, 70, 150),   # lime
            (160, 120, 80, 150),   # brown
        ]

        self.phasor_rois: list[pg.CircleROI] = []

        self.setWindowTitle("Phasora")
        self.resize(1450, 850)

        self.open_button = QPushButton("Open TIFF")
        self.open_button.setMinimumHeight(42)
        self.open_button.clicked.connect(self.open_tiff)

        self.add_selection_button = QPushButton("Add selection")
        self.add_selection_button.setMinimumHeight(42)
        self.add_selection_button.setEnabled(False)
        self.add_selection_button.clicked.connect(self.add_selection)

        self.remove_selection_button = QPushButton("Remove last selection")
        self.remove_selection_button.setMinimumHeight(42)
        self.remove_selection_button.setEnabled(False)
        self.remove_selection_button.clicked.connect(
            self.remove_last_selection
        )

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.add_selection_button)
        button_layout.addWidget(self.remove_selection_button)

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

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
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

        self.clear_selections()
        self.add_selection()

    def add_selection(self) -> None:
        if self.g is None or self.s is None:
            return

        if len(self.phasor_rois) >= self.MAX_SELECTIONS:
            QMessageBox.information(
                self,
                "Selection limit",
                f"A maximum of {self.MAX_SELECTIONS} selections is allowed.",
            )
            return

        selection_index = len(self.phasor_rois)
        color = self.selection_colors[selection_index]

        offset = 0.04 * selection_index

        roi = pg.CircleROI(
            pos=(-0.1 + offset, -0.1 + offset),
            size=(0.2, 0.2),
            movable=True,
            resizable=True,
            pen=pg.mkPen(
                color=color[:3],
                width=2,
            ),
        )

        roi.sigRegionChanged.connect(self.update_selection)

        self.phasor_plot.addItem(roi)
        self.phasor_rois.append(roi)

        self.remove_selection_button.setEnabled(True)
        self.add_selection_button.setEnabled(
            len(self.phasor_rois) < self.MAX_SELECTIONS
        )

        self.update_selection()

    def remove_last_selection(self) -> None:
        if not self.phasor_rois:
            return

        roi = self.phasor_rois.pop()
        self.phasor_plot.removeItem(roi)

        self.remove_selection_button.setEnabled(
            len(self.phasor_rois) > 0
        )
        self.add_selection_button.setEnabled(True)

        self.update_selection()

    def clear_selections(self) -> None:
        for roi in self.phasor_rois:
            self.phasor_plot.removeItem(roi)

        self.phasor_rois.clear()

        self.remove_selection_button.setEnabled(False)
        self.add_selection_button.setEnabled(
            self.g is not None and self.s is not None
        )

        if self.dc is not None:
            empty_overlay = np.zeros(
                (*self.dc.shape, 4),
                dtype=np.uint8,
            )
            self.overlay_item.setImage(
                empty_overlay,
                autoLevels=False,
            )

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

        valid = (
            (self.dc > 0)
            & np.isfinite(self.g)
            & np.isfinite(self.s)
        )

        overlay = np.zeros(
            (*self.dc.shape, 4),
            dtype=np.uint8,
        )

        selection_summaries: list[str] = []

        for index, roi in enumerate(self.phasor_rois):
            roi_position = roi.pos()
            roi_size = roi.size()

            center_g = float(
                roi_position.x() + roi_size.x() / 2.0
            )
            center_s = float(
                roi_position.y() + roi_size.y() / 2.0
            )
            radius = float(
                min(roi_size.x(), roi_size.y()) / 2.0
            )

            mask = circular_phasor_mask(
                g=self.g,
                s=self.s,
                center_g=center_g,
                center_s=center_s,
                radius=radius,
                valid_mask=valid,
            )

            color = self.selection_colors[index]

            overlay[mask, 0] = color[0]
            overlay[mask, 1] = color[1]
            overlay[mask, 2] = color[2]
            overlay[mask, 3] = color[3]

            selected_pixels = int(np.count_nonzero(mask))
            valid_pixels = int(np.count_nonzero(valid))

            if valid_pixels > 0:
                selected_percentage = (
                    100.0 * selected_pixels / valid_pixels
                )
            else:
                selected_percentage = 0.0

            selection_summaries.append(
                f"S{index + 1}: "
                f"g={center_g:.3f}, "
                f"s={center_s:.3f}, "
                f"r={radius:.3f}, "
                f"{selected_pixels:,} px "
                f"({selected_percentage:.2f}%)"
            )

        self.overlay_item.setImage(
            overlay,
            autoLevels=False,
        )

        file_name = (
            self.current_path.name
            if self.current_path is not None
            else "Unknown"
        )

        if selection_summaries:
            selection_text = " | ".join(selection_summaries)
        else:
            selection_text = "No active selections"

        self.status_label.setText(
            f"File: {file_name} | {selection_text}"
        )


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())