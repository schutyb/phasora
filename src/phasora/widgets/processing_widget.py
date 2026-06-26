from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from phasora.core.visualization import PhasorMode


class ProcessingWidget(QWidget):
    """General phasor-processing controls.

    This widget owns only the interface and emits user intentions.
    Scientific calculations remain in MainWindow/core modules.
    """

    mode_changed = Signal(str)
    histogram_bins_changed = Signal(int)
    colormap_changed = Signal(str)

    filter_requested = Signal(str, int, int)
    filter_reset_requested = Signal()

    threshold_value_changed = Signal(float)
    upper_threshold_enabled_changed = Signal(bool)
    upper_threshold_value_changed = Signal(float)
    threshold_requested = Signal(float, bool, float)
    threshold_reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._build_display_controls()
        self._build_filter_controls()
        self._build_threshold_controls()
        self._build_layout()
        self._connect_signals()
        self.update_filter_control_state()

    def _build_display_controls(self) -> None:
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                PhasorMode.SPECTRAL.value,
                PhasorMode.LIFETIME.value,
            ]
        )

        self.histogram_bins_spinbox = QSpinBox()
        self.histogram_bins_spinbox.setRange(32, 1024)
        self.histogram_bins_spinbox.setSingleStep(32)
        self.histogram_bins_spinbox.setValue(256)
        self.histogram_bins_spinbox.setToolTip(
            "Number of bins used to render the 2D phasor histogram."
        )

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            [
                "Inferno",
                "Viridis",
                "Magma",
                "Cyan",
                "Grayscale",
                "Spectral",
                "Red-Green",
            ]
        )

        display_form = QFormLayout()
        display_form.addRow("Phasor type:", self.mode_combo)
        display_form.addRow("Histogram bins:", self.histogram_bins_spinbox)
        display_form.addRow("Colormap:", self.colormap_combo)

        self.display_box = QGroupBox("Phasor display")
        self.display_box.setLayout(display_form)

    def _build_filter_controls(self) -> None:
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Median"])

        self.filter_size_spinbox = QSpinBox()
        self.filter_size_spinbox.setRange(1, 15)
        self.filter_size_spinbox.setSingleStep(2)
        self.filter_size_spinbox.setValue(3)
        self.filter_size_spinbox.setToolTip("Spatial median-filter kernel size. Use an odd value.")

        self.filter_repeat_spinbox = QSpinBox()
        self.filter_repeat_spinbox.setRange(1, 10)
        self.filter_repeat_spinbox.setValue(1)
        self.filter_repeat_spinbox.setToolTip("Number of times the median filter is applied.")

        self.apply_filter_button = QPushButton("Apply filter")
        self.apply_filter_button.setEnabled(False)

        self.reset_filter_button = QPushButton("Reset filter")
        self.reset_filter_button.setEnabled(False)

        filter_form = QFormLayout()
        filter_form.addRow("Filter:", self.filter_combo)
        filter_form.addRow("Kernel size:", self.filter_size_spinbox)
        filter_form.addRow("Repetitions:", self.filter_repeat_spinbox)
        filter_form.addRow(self.apply_filter_button)
        filter_form.addRow(self.reset_filter_button)

        self.filter_box = QGroupBox("Filters")
        self.filter_box.setLayout(filter_form)

    def _build_threshold_controls(self) -> None:
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setDecimals(2)
        self.threshold_spinbox.setSingleStep(1.0)
        self.threshold_spinbox.setValue(0.0)
        self.threshold_spinbox.setEnabled(False)
        self.threshold_spinbox.setToolTip("Minimum DC intensity retained in the phasor analysis.")

        self.upper_threshold_checkbox = QCheckBox("Enable maximum DC")
        self.upper_threshold_checkbox.setEnabled(False)
        self.upper_threshold_checkbox.setChecked(False)
        self.upper_threshold_checkbox.setToolTip(
            "Enable an upper DC threshold to remove saturated pixels."
        )

        self.upper_threshold_spinbox = QDoubleSpinBox()
        self.upper_threshold_spinbox.setRange(0.0, 1.0)
        self.upper_threshold_spinbox.setDecimals(2)
        self.upper_threshold_spinbox.setSingleStep(1.0)
        self.upper_threshold_spinbox.setValue(1.0)
        self.upper_threshold_spinbox.setEnabled(False)
        self.upper_threshold_spinbox.setToolTip(
            "Pixels with DC greater than or equal to this value are removed."
        )

        self.apply_threshold_button = QPushButton("Apply thresholds")
        self.apply_threshold_button.setEnabled(False)

        self.reset_threshold_button = QPushButton("Reset thresholds")
        self.reset_threshold_button.setEnabled(False)

        threshold_form = QFormLayout()
        threshold_form.addRow("Minimum DC:", self.threshold_spinbox)
        threshold_form.addRow(self.upper_threshold_checkbox)
        threshold_form.addRow(
            "Maximum DC:",
            self.upper_threshold_spinbox,
        )
        threshold_form.addRow(self.apply_threshold_button)
        threshold_form.addRow(self.reset_threshold_button)

        self.threshold_box = QGroupBox("DC thresholds")
        self.threshold_box.setLayout(threshold_form)

    def _build_layout(self) -> None:
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(12)
        content_layout.addWidget(self.display_box)
        content_layout.addWidget(self.filter_box)
        content_layout.addWidget(self.threshold_box)
        content_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)

    def _connect_signals(self) -> None:
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        self.histogram_bins_spinbox.valueChanged.connect(self.histogram_bins_changed)
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed)

        self.filter_combo.currentTextChanged.connect(self.update_filter_control_state)
        self.apply_filter_button.clicked.connect(self._emit_filter_requested)
        self.reset_filter_button.clicked.connect(self.filter_reset_requested)

        self.threshold_spinbox.valueChanged.connect(self.threshold_value_changed)
        self.upper_threshold_checkbox.toggled.connect(self._upper_threshold_toggled)
        self.upper_threshold_spinbox.valueChanged.connect(self.upper_threshold_value_changed)
        self.apply_threshold_button.clicked.connect(self._emit_threshold_requested)
        self.reset_threshold_button.clicked.connect(self.threshold_reset_requested)

    def _emit_filter_requested(self) -> None:
        self.filter_requested.emit(
            self.filter_name(),
            self.filter_size(),
            self.filter_repetitions(),
        )

    def update_filter_control_state(self, _value: object | None = None) -> None:
        median_selected = self.filter_name() == "Median"
        self.filter_size_spinbox.setEnabled(median_selected)
        self.filter_repeat_spinbox.setEnabled(median_selected)

    def set_image_loaded(self, loaded: bool) -> None:
        self.apply_filter_button.setEnabled(loaded)
        self.reset_filter_button.setEnabled(loaded)
        self.threshold_spinbox.setEnabled(loaded)
        self.upper_threshold_checkbox.setEnabled(loaded)
        self.upper_threshold_spinbox.setEnabled(
            loaded and self.upper_threshold_checkbox.isChecked()
        )
        self.apply_threshold_button.setEnabled(loaded)
        self.reset_threshold_button.setEnabled(loaded)

    def configure_threshold(
        self,
        *,
        maximum: float,
        value: float,
    ) -> None:
        maximum = max(float(maximum), 1.0)
        value = min(max(float(value), 0.0), maximum)

        step = max(maximum / 1000.0, 1.0)

        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setRange(0.0, maximum)
        self.threshold_spinbox.setSingleStep(step)
        self.threshold_spinbox.setValue(value)
        self.threshold_spinbox.blockSignals(False)

        self.upper_threshold_spinbox.blockSignals(True)
        self.upper_threshold_spinbox.setRange(0.0, maximum)
        self.upper_threshold_spinbox.setSingleStep(step)
        self.upper_threshold_spinbox.setValue(maximum)
        self.upper_threshold_spinbox.blockSignals(False)

        self.upper_threshold_checkbox.blockSignals(True)
        self.upper_threshold_checkbox.setChecked(False)
        self.upper_threshold_checkbox.blockSignals(False)
        self.upper_threshold_spinbox.setEnabled(False)

    def set_threshold_value(self, value: float) -> None:
        value = min(
            max(float(value), self.threshold_spinbox.minimum()),
            self.threshold_spinbox.maximum(),
        )
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(value)
        self.threshold_spinbox.blockSignals(False)

    def _upper_threshold_toggled(self, enabled: bool) -> None:
        self.upper_threshold_spinbox.setEnabled(bool(enabled))
        self.upper_threshold_enabled_changed.emit(bool(enabled))

    def _emit_threshold_requested(self) -> None:
        self.threshold_requested.emit(
            self.threshold_value(),
            self.upper_threshold_enabled(),
            self.upper_threshold_value(),
        )

    def set_upper_threshold_value(self, value: float) -> None:
        value = min(
            max(float(value), self.upper_threshold_spinbox.minimum()),
            self.upper_threshold_spinbox.maximum(),
        )
        self.upper_threshold_spinbox.blockSignals(True)
        self.upper_threshold_spinbox.setValue(value)
        self.upper_threshold_spinbox.blockSignals(False)

    def set_upper_threshold_enabled(self, enabled: bool) -> None:
        self.upper_threshold_checkbox.setChecked(bool(enabled))

    def set_mode(self, mode: str) -> None:
        index = self.mode_combo.findText(mode)
        if index < 0:
            raise ValueError(f"Unknown phasor mode: {mode}")

        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)

    def set_histogram_bins(self, bins: int) -> None:
        value = int(bins)
        value = min(
            max(value, self.histogram_bins_spinbox.minimum()),
            self.histogram_bins_spinbox.maximum(),
        )

        self.histogram_bins_spinbox.blockSignals(True)
        self.histogram_bins_spinbox.setValue(value)
        self.histogram_bins_spinbox.blockSignals(False)

    def set_colormap_name(self, name: str) -> None:
        index = self.colormap_combo.findText(name)
        if index < 0:
            raise ValueError(f"Unknown colormap: {name}")

        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentIndex(index)
        self.colormap_combo.blockSignals(False)

    def set_filter_settings(
        self,
        *,
        name: str,
        size: int,
        repetitions: int,
    ) -> None:
        index = self.filter_combo.findText(name)
        if index < 0:
            raise ValueError(f"Unknown filter: {name}")

        self.filter_combo.blockSignals(True)
        self.filter_combo.setCurrentIndex(index)
        self.filter_combo.blockSignals(False)

        self.filter_size_spinbox.blockSignals(True)
        self.filter_size_spinbox.setValue(int(size))
        self.filter_size_spinbox.blockSignals(False)

        self.filter_repeat_spinbox.blockSignals(True)
        self.filter_repeat_spinbox.setValue(int(repetitions))
        self.filter_repeat_spinbox.blockSignals(False)

        self.update_filter_control_state()

    def mode(self) -> PhasorMode:
        return PhasorMode(self.mode_combo.currentText())

    def histogram_bins(self) -> int:
        return self.histogram_bins_spinbox.value()

    def colormap_name(self) -> str:
        return self.colormap_combo.currentText()

    def filter_name(self) -> str:
        return self.filter_combo.currentText()

    def filter_size(self) -> int:
        return self.filter_size_spinbox.value()

    def filter_repetitions(self) -> int:
        return self.filter_repeat_spinbox.value()

    def threshold_value(self) -> float:
        return float(self.threshold_spinbox.value())

    def upper_threshold_enabled(self) -> bool:
        return self.upper_threshold_checkbox.isChecked()

    def upper_threshold_value(self) -> float:
        return float(self.upper_threshold_spinbox.value())

    def threshold_maximum(self) -> float:
        return float(self.threshold_spinbox.maximum())
