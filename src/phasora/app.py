from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from phasora.core.io import load_tiff
from phasora.core.phasor import compute_phasor
from phasora.core.selection import circular_phasor_mask
from phasora.core.visualization import (
    PhasorMode,
    compute_phasor_histogram,
    phasor_plot_ranges,
    universal_semicircle,
)
from phasora.export.results import SelectionResult, export_selections


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
        self.resize(1500, 900)

        # File and cursor controls
        self.open_button = QPushButton("Open TIFF")
        self.open_button.setMinimumHeight(42)
        self.open_button.clicked.connect(self.open_tiff)

        self.add_selection_button = QPushButton("Add cursor")
        self.add_selection_button.setMinimumHeight(42)
        self.add_selection_button.setEnabled(False)
        self.add_selection_button.clicked.connect(self.add_selection)

        self.remove_selection_button = QPushButton("Remove last cursor")
        self.remove_selection_button.setMinimumHeight(42)
        self.remove_selection_button.setEnabled(False)
        self.remove_selection_button.clicked.connect(
            self.remove_last_selection
        )

        self.export_button = QPushButton("Export selections")
        self.export_button.setMinimumHeight(42)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(
            self.export_current_selections
        )

        # Phasor display controls
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                PhasorMode.SPECTRAL.value,
                PhasorMode.LIFETIME.value,
            ]
        )
        self.mode_combo.currentTextChanged.connect(
            self.update_display_settings
        )

        self.histogram_bins_spinbox = QSpinBox()
        self.histogram_bins_spinbox.setRange(32, 1024)
        self.histogram_bins_spinbox.setSingleStep(32)
        self.histogram_bins_spinbox.setValue(256)
        self.histogram_bins_spinbox.valueChanged.connect(
            self.update_display_settings
        )

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            [
                "Inferno",
                "Viridis",
                "Magma",
                "Cyan",
                "Grayscale",
            ]
        )
        self.colormap_combo.currentTextChanged.connect(
            self.update_colormap
        )

        display_form = QFormLayout()
        display_form.addRow("Phasor type:", self.mode_combo)
        display_form.addRow(
            "Histogram bins:",
            self.histogram_bins_spinbox,
        )
        display_form.addRow(
            "Colormap:",
            self.colormap_combo,
        )

        display_box = QGroupBox("Phasor display")
        display_box.setLayout(display_form)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.add_selection_button)
        button_layout.addWidget(self.remove_selection_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(display_box)

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
            title="Spectral phasor histogram",
        )
        self.phasor_plot.setAspectLocked(True)
        self.phasor_plot.setLabel("bottom", "g")
        self.phasor_plot.setLabel("left", "s")
        self.phasor_plot.setXRange(-1.0, 1.0)
        self.phasor_plot.setYRange(-1.0, 1.0)

        self.phasor_image_item = pg.ImageItem()
        self.phasor_plot.addItem(self.phasor_image_item)

        # Spectral unit circle or lifetime universal semicircle
        self.reference_curve_item = pg.PlotDataItem(
            pen=pg.mkPen(
                color=(230, 230, 230),
                width=2,
            )
        )
        self.phasor_plot.addItem(self.reference_curve_item)

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.graphics, stretch=1)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        self.update_reference_curve()
        self.update_colormap()

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
                "Cursor limit",
                f"A maximum of {self.MAX_SELECTIONS} cursors is allowed.",
            )
            return

        selection_index = len(self.phasor_rois)
        color = self.selection_colors[selection_index]
        mode = PhasorMode(self.mode_combo.currentText())

        if mode == PhasorMode.LIFETIME:
            offset = 0.025 * selection_index
            position = (
                0.40 + offset,
                0.10 + offset,
            )
            size = (0.12, 0.12)
        else:
            offset = 0.04 * selection_index
            position = (
                -0.10 + offset,
                -0.10 + offset,
            )
            size = (0.20, 0.20)

        roi = pg.CircleROI(
            pos=position,
            size=size,
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
        self.export_button.setEnabled(True)

        self.update_selection()

    def remove_last_selection(self) -> None:
        if not self.phasor_rois:
            return

        roi = self.phasor_rois.pop()
        self.phasor_plot.removeItem(roi)

        has_selections = len(self.phasor_rois) > 0

        self.remove_selection_button.setEnabled(has_selections)
        self.export_button.setEnabled(has_selections)
        self.add_selection_button.setEnabled(True)

        self.update_selection()

    def clear_selections(self) -> None:
        for roi in self.phasor_rois:
            self.phasor_plot.removeItem(roi)

        self.phasor_rois.clear()

        self.remove_selection_button.setEnabled(False)
        self.export_button.setEnabled(False)
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

    def export_current_selections(self) -> None:
        if self.dc is None or self.g is None or self.s is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Open a TIFF image before exporting selections.",
            )
            return

        if not self.phasor_rois:
            QMessageBox.warning(
                self,
                "No selections",
                "Add at least one phasor cursor before exporting.",
            )
            return

        output_directory = QFileDialog.getExistingDirectory(
            self,
            "Select export directory",
            "",
        )

        if not output_directory:
            return

        valid = (
            (self.dc > 0)
            & np.isfinite(self.g)
            & np.isfinite(self.s)
        )

        selections: list[SelectionResult] = []

        for index, roi in enumerate(self.phasor_rois, start=1):
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

            selections.append(
                SelectionResult(
                    index=index,
                    center_g=center_g,
                    center_s=center_s,
                    radius=radius,
                    mask=mask,
                )
            )

        try:
            export_selections(
                output_directory=output_directory,
                selections=selections,
            )
        except (ValueError, OSError) as error:
            QMessageBox.critical(
                self,
                "Export failed",
                str(error),
            )
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Selections were exported to:\n{output_directory}",
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

        mode = PhasorMode(self.mode_combo.currentText())
        x_range, y_range = phasor_plot_ranges(mode)

        valid = (
            (self.dc > 0)
            & np.isfinite(self.g)
            & np.isfinite(self.s)
        )

        histogram = compute_phasor_histogram(
            self.g,
            self.s,
            bins=self.histogram_bins_spinbox.value(),
            x_range=x_range,
            y_range=y_range,
            valid_mask=valid,
            log_scale=True,
        )

        self.phasor_image_item.setImage(
            histogram,
            autoLevels=True,
        )

        self.phasor_image_item.setRect(
            QRectF(
                x_range[0],
                y_range[0],
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
            )
        )

        self.phasor_plot.setXRange(
            *x_range,
            padding=0,
        )
        self.phasor_plot.setYRange(
            *y_range,
            padding=0,
        )

        self.phasor_plot.setTitle(
            f"{mode.value} phasor histogram"
        )

        self.update_reference_curve()

    def update_reference_curve(self) -> None:
        mode = PhasorMode(self.mode_combo.currentText())

        if mode == PhasorMode.LIFETIME:
            curve_g, curve_s = universal_semicircle()
        else:
            theta = np.linspace(
                0.0,
                2.0 * np.pi,
                512,
            )
            curve_g = np.cos(theta)
            curve_s = np.sin(theta)

        self.reference_curve_item.setData(
            curve_g,
            curve_s,
        )

    def update_display_settings(
        self,
        _value: object | None = None,
    ) -> None:
        self.update_reference_curve()

        if self.g is None or self.s is None:
            return

        self.update_phasor_view()
        self.update_selection()

    def update_colormap(
        self,
        _value: object | None = None,
    ) -> None:
        colormaps = {
            "Inferno": pg.ColorMap(
                [0.0, 0.35, 0.70, 1.0],
                [
                    (0, 0, 4),
                    (87, 15, 109),
                    (249, 142, 8),
                    (252, 255, 164),
                ],
            ),
            "Viridis": pg.ColorMap(
                [0.0, 0.35, 0.70, 1.0],
                [
                    (68, 1, 84),
                    (49, 104, 142),
                    (53, 183, 121),
                    (253, 231, 37),
                ],
            ),
            "Magma": pg.ColorMap(
                [0.0, 0.35, 0.70, 1.0],
                [
                    (0, 0, 4),
                    (114, 31, 129),
                    (241, 96, 93),
                    (252, 253, 191),
                ],
            ),
            "Cyan": pg.ColorMap(
                [0.0, 0.50, 1.0],
                [
                    (0, 0, 0),
                    (0, 100, 130),
                    (180, 255, 255),
                ],
            ),
            "Grayscale": pg.ColorMap(
                [0.0, 1.0],
                [
                    (0, 0, 0),
                    (255, 255, 255),
                ],
            ),
        }

        selected_name = self.colormap_combo.currentText()

        color_map = colormaps.get(
            selected_name,
            colormaps["Inferno"],
        )

        lookup_table = color_map.getLookupTable(
            start=0.0,
            stop=1.0,
            nPts=256,
        )

        self.phasor_image_item.setLookupTable(
            lookup_table
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
        valid_pixels = int(np.count_nonzero(valid))

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

            if valid_pixels > 0:
                selected_percentage = (
                    100.0 * selected_pixels / valid_pixels
                )
            else:
                selected_percentage = 0.0

            selection_summaries.append(
                f"C{index + 1}: "
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

        mode = self.mode_combo.currentText()
        histogram_bins = self.histogram_bins_spinbox.value()

        if selection_summaries:
            selection_text = " | ".join(selection_summaries)
        else:
            selection_text = "No active cursors"

        self.status_label.setText(
            f"File: {file_name} | "
            f"Mode: {mode} | "
            f"Histogram bins: {histogram_bins} | "
            f"{selection_text}"
        )


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())