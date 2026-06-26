from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class DCHistogramWidget(QWidget):
    """DC histogram with optional upper threshold and line/bar display."""

    threshold_changed = Signal(float)
    upper_threshold_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._centers = np.array([], dtype=np.float64)
        self._counts = np.array([], dtype=np.float64)
        self._bar_width = 1.0
        self._x_max = 1.0

        self.graphics = pg.GraphicsLayoutWidget()
        self.plot = self.graphics.addPlot(
            title="DC intensity histogram",
        )
        self.plot.setLabel("bottom", "DC intensity")
        self.plot.setLabel("left", "Pixel count")
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        self.curve = pg.PlotDataItem(pen=pg.mkPen(color=(220, 220, 220), width=2))
        self.plot.addItem(self.curve)

        self.bars = pg.BarGraphItem(
            x=[],
            height=[],
            width=1.0,
            brush=pg.mkBrush(180, 180, 180, 180),
            pen=pg.mkPen(220, 220, 220, 220),
        )
        self.plot.addItem(self.bars)
        self.bars.setVisible(False)

        self.threshold_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color=(255, 80, 80), width=2),
            label="Minimum",
            labelOpts={"position": 0.90},
        )
        self.threshold_line.setVisible(False)
        self.threshold_line.sigPositionChangeFinished.connect(self._emit_threshold_changed)
        self.plot.addItem(self.threshold_line)

        self.upper_threshold_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color=(255, 180, 40), width=2),
            label="Maximum",
            labelOpts={"position": 0.78},
        )
        self.upper_threshold_line.setVisible(False)
        self.upper_threshold_line.sigPositionChangeFinished.connect(
            self._emit_upper_threshold_changed
        )
        self.plot.addItem(self.upper_threshold_line)

        self.style_combo = QComboBox()
        self.style_combo.addItems(["Line", "Bars"])
        self.style_combo.currentTextChanged.connect(self._update_display_style)

        self.log_y_checkbox = QCheckBox("Logarithmic Y axis")
        self.log_y_checkbox.setToolTip("Display pixel counts as log10(count).")
        self.log_y_checkbox.toggled.connect(self.set_log_y)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(8, 4, 8, 4)
        controls_layout.addWidget(QLabel("Display:"))
        controls_layout.addWidget(self.style_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.log_y_checkbox)
        controls_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls_layout)
        layout.addWidget(self.graphics, stretch=1)

    def set_histogram(
        self,
        *,
        centers: np.ndarray,
        counts: np.ndarray,
        x_max: float,
    ) -> None:
        centers_array = np.asarray(centers, dtype=np.float64)
        counts_array = np.asarray(counts, dtype=np.float64)

        if centers_array.ndim != 1 or counts_array.ndim != 1:
            raise ValueError("Histogram centers and counts must be one-dimensional.")
        if centers_array.shape != counts_array.shape:
            raise ValueError("Histogram centers and counts must have matching shapes.")

        self._centers = centers_array
        self._counts = counts_array
        self._x_max = max(float(x_max), 1.0)

        if centers_array.size >= 2:
            differences = np.diff(centers_array)
            positive_differences = differences[differences > 0]
            self._bar_width = (
                float(np.median(positive_differences)) if positive_differences.size > 0 else 1.0
            )
        else:
            self._bar_width = 1.0

        self._render_histogram()

    def set_threshold(
        self,
        value: float,
        *,
        visible: bool = True,
    ) -> None:
        self.threshold_line.setValue(float(value))
        self.threshold_line.setVisible(visible)

    def set_upper_threshold(
        self,
        value: float,
        *,
        visible: bool,
    ) -> None:
        self.upper_threshold_line.setValue(float(value))
        self.upper_threshold_line.setVisible(bool(visible))

    def threshold_value(self) -> float:
        return float(self.threshold_line.value())

    def upper_threshold_value(self) -> float:
        return float(self.upper_threshold_line.value())

    def set_log_y(self, enabled: bool) -> None:
        self.plot.setLabel(
            "left",
            "log10(Pixel count)" if enabled else "Pixel count",
        )
        self._render_histogram()

    def clear(self) -> None:
        self._centers = np.array([], dtype=np.float64)
        self._counts = np.array([], dtype=np.float64)
        self.curve.setData([], [])
        self.bars.setOpts(x=[], height=[], width=self._bar_width)
        self.threshold_line.setVisible(False)
        self.upper_threshold_line.setVisible(False)

    def _update_display_style(self, _style_name: str) -> None:
        self._render_histogram()

    def _display_counts(self) -> np.ndarray:
        if self.log_y_checkbox.isChecked():
            return np.log10(np.maximum(self._counts, 1.0))
        return self._counts

    def _render_histogram(self) -> None:
        display_counts = self._display_counts()

        if self.style_combo.currentText() == "Bars":
            self.curve.setVisible(False)
            self.bars.setVisible(True)
            self.bars.setOpts(
                x=self._centers,
                height=display_counts,
                width=self._bar_width * 0.95,
            )
        else:
            self.bars.setVisible(False)
            self.curve.setVisible(True)
            self.curve.setData(self._centers, display_counts)

        self.plot.setXRange(
            0.0,
            self._x_max,
            padding=0.01,
        )

        finite_counts = display_counts[np.isfinite(display_counts)]
        y_max = float(np.max(finite_counts)) if finite_counts.size > 0 else 1.0
        if y_max <= 0:
            y_max = 1.0

        self.plot.setYRange(
            0.0,
            y_max * 1.10,
            padding=0.0,
        )

    def _emit_threshold_changed(self) -> None:
        self.threshold_changed.emit(self.threshold_value())

    def _emit_upper_threshold_changed(self) -> None:
        self.upper_threshold_changed.emit(self.upper_threshold_value())
