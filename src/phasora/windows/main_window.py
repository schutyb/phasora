from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from phasorpy.filter import phasor_filter_median, phasor_threshold
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from phasora.core.io import load_image
from phasora.core.phasor import compute_phasor
from phasora.core.selection import circular_phasor_mask
from phasora.core.visualization import (
    PhasorMode,
    compute_phasor_histogram,
    phasor_plot_ranges,
    universal_semicircle,
)
from phasora.export.results import SelectionResult, export_selections
from phasora.models.session import (
    CursorState,
    PhasoraSession,
    ViewRangeState,
)
from phasora.export.statistics import (
    CursorStatistics,
    export_cursor_statistics,
)
from phasora.widgets.dc_histogram_widget import DCHistogramWidget
from phasora.widgets.dc_image_widget import DCImageWidget
from phasora.widgets.phasor_widget import PhasorWidget
from phasora.widgets.processing_widget import ProcessingWidget


pg.setConfigOptions(imageAxisOrder="row-major")


class MainWindow(QMainWindow):
    MAX_SELECTIONS = 10
    DC_HISTOGRAM_BINS = 256

    def __init__(self) -> None:
        super().__init__()

        self.current_path: Path | None = None
        self.dc: np.ndarray | None = None
        self.original_g: np.ndarray | None = None
        self.original_s: np.ndarray | None = None
        self.filtered_g: np.ndarray | None = None
        self.filtered_s: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.s: np.ndarray | None = None

        self.active_filter_description = "No filter"
        self.active_threshold: float | None = None
        self.active_upper_threshold: float | None = None

        self.selection_colors = [
            (255, 80, 80, 150),
            (80, 180, 255, 150),
            (100, 220, 120, 150),
            (255, 200, 70, 150),
            (200, 100, 255, 150),
            (255, 140, 60, 150),
            (80, 220, 220, 150),
            (255, 100, 180, 150),
            (170, 220, 70, 150),
            (160, 120, 80, 150),
        ]
        self.phasor_rois: list[pg.CircleROI] = []

        self.setWindowTitle("Phasora")
        self.resize(1450, 900)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        self._build_action_buttons()
        self._build_processing_dock()
        self._build_phasor_view()
        self._build_dc_image_dock()
        self._build_dc_histogram_dock()
        self._build_central_widget()
        self._build_menus()
        self._connect_processing_signals()

        self.update_reference_curve()
        self.update_colormap()

    def _build_action_buttons(self) -> None:
        self.open_button = QPushButton("Open image")
        self.open_button.setMinimumHeight(42)
        self.open_button.clicked.connect(self.open_image)

        self.add_selection_button = QPushButton("Add cursor")
        self.add_selection_button.setMinimumHeight(42)
        self.add_selection_button.setEnabled(False)
        self.add_selection_button.clicked.connect(self.add_selection)

        self.remove_selection_button = QPushButton("Remove last cursor")
        self.remove_selection_button.setMinimumHeight(42)
        self.remove_selection_button.setEnabled(False)
        self.remove_selection_button.clicked.connect(self.remove_last_selection)

        self.export_button = QPushButton("Export selections")
        self.export_button.setMinimumHeight(42)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_current_selections)

        self.export_statistics_button = QPushButton("Export cursor statistics")
        self.export_statistics_button.setMinimumHeight(42)
        self.export_statistics_button.setEnabled(False)
        self.export_statistics_button.clicked.connect(self.export_current_cursor_statistics)

        self.export_phasor_button = QPushButton("Export phasor image")
        self.export_phasor_button.setMinimumHeight(42)
        self.export_phasor_button.setEnabled(False)
        self.export_phasor_button.clicked.connect(self.export_phasor_image)

    def _build_processing_dock(self) -> None:
        self.processing_widget = ProcessingWidget()

        self.processing_dock = QDockWidget("Processing", self)
        self.processing_dock.setObjectName("processing_dock")
        self.processing_dock.setWidget(self.processing_widget)
        self.processing_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.processing_dock.setMinimumWidth(285)

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.processing_dock,
        )

    def _build_phasor_view(self) -> None:
        self.phasor_widget = PhasorWidget()

    def _build_dc_image_dock(self) -> None:
        self.dc_image_widget = DCImageWidget()

        self.dc_image_dock = QDockWidget("DC Image", self)
        self.dc_image_dock.setObjectName("dc_image_dock")
        self.dc_image_dock.setWidget(self.dc_image_widget)
        self.dc_image_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.dc_image_dock,
        )
        self.dc_image_dock.setFloating(True)
        self.dc_image_dock.resize(700, 700)

    def _build_dc_histogram_dock(self) -> None:
        self.dc_histogram_widget = DCHistogramWidget()
        self.dc_histogram_widget.threshold_changed.connect(self.threshold_line_moved)
        self.dc_histogram_widget.upper_threshold_changed.connect(self.upper_threshold_line_moved)

        self.dc_histogram_dock = QDockWidget("DC Histogram", self)
        self.dc_histogram_dock.setObjectName("dc_histogram_dock")
        self.dc_histogram_dock.setWidget(self.dc_histogram_widget)
        self.dc_histogram_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.dc_histogram_dock,
        )
        self.dc_histogram_dock.setFloating(True)
        self.dc_histogram_dock.resize(760, 420)

    def _build_central_widget(self) -> None:
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.add_selection_button)
        button_layout.addWidget(self.remove_selection_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.export_statistics_button)
        button_layout.addWidget(self.export_phasor_button)
        button_layout.addStretch()

        self.status_label = QLabel("Open a supported multichannel microscopy image.")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.phasor_widget, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open image", self.open_image)
        file_menu.addSeparator()
        file_menu.addAction("Save session", self.save_session)
        file_menu.addAction("Load session", self.load_session)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.processing_dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.dc_image_dock.toggleViewAction())
        view_menu.addAction(self.dc_histogram_dock.toggleViewAction())

    def _connect_processing_signals(self) -> None:
        self.processing_widget.mode_changed.connect(self.update_display_settings)
        self.processing_widget.histogram_bins_changed.connect(self.update_display_settings)
        self.processing_widget.colormap_changed.connect(self.update_colormap)

        self.processing_widget.filter_requested.connect(self.apply_selected_filter)
        self.processing_widget.filter_reset_requested.connect(self.reset_filter)

        self.processing_widget.threshold_value_changed.connect(self.threshold_spinbox_changed)
        self.processing_widget.upper_threshold_enabled_changed.connect(
            self.upper_threshold_enabled_changed
        )
        self.processing_widget.upper_threshold_value_changed.connect(
            self.upper_threshold_spinbox_changed
        )
        self.processing_widget.threshold_requested.connect(self.apply_threshold)
        self.processing_widget.threshold_reset_requested.connect(self.reset_threshold)

    def open_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open microscopy image",
            "",
            (
                "Microscopy images (*.tif *.tiff *.lsm *.czi);;"
                "TIFF images (*.tif *.tiff);;"
                "Zeiss LSM images (*.lsm);;"
                "Zeiss CZI images (*.czi);;"
                "All files (*)"
            ),
        )

        if not file_name:
            return

        self._load_image_path(
            Path(file_name),
            add_default_cursor=True,
        )

    def _load_image_path(
        self,
        path: Path,
        *,
        add_default_cursor: bool,
    ) -> bool:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            image = load_image(path)
            dc, g, s = compute_phasor(image)
        except (
            FileNotFoundError,
            ImportError,
            ValueError,
            OSError,
            RuntimeError,
        ) as error:
            QMessageBox.critical(
                self,
                "Unable to open image",
                str(error),
            )
            return False
        finally:
            QApplication.restoreOverrideCursor()

        self.current_path = path
        self.dc = np.asarray(dc, dtype=np.float64)
        self.original_g = np.asarray(g, dtype=np.float64).copy()
        self.original_s = np.asarray(s, dtype=np.float64).copy()
        self.filtered_g = self.original_g.copy()
        self.filtered_s = self.original_s.copy()
        self.g = self.filtered_g.copy()
        self.s = self.filtered_s.copy()

        self.active_filter_description = "No filter"
        self.active_threshold = None
        self.active_upper_threshold = None

        self.processing_widget.set_image_loaded(True)
        self.export_phasor_button.setEnabled(True)
        self.configure_threshold_controls()

        self.update_dc_view()
        self.update_phasor_view()
        self.update_dc_histogram()

        self.clear_selections()

        if add_default_cursor:
            self.add_selection()

        return True

    def configure_threshold_controls(self) -> None:
        if self.dc is None:
            return

        finite_dc = self.dc[np.isfinite(self.dc) & (self.dc >= 0)]
        if finite_dc.size == 0:
            maximum = 1.0
            initial_threshold = 0.0
        else:
            maximum = float(np.max(finite_dc))
            positive_dc = finite_dc[finite_dc > 0]
            initial_threshold = (
                float(np.percentile(positive_dc, 5)) if positive_dc.size > 0 else 0.0
            )

        self.processing_widget.configure_threshold(
            maximum=maximum,
            value=initial_threshold,
        )
        self.dc_histogram_widget.set_threshold(
            initial_threshold,
            visible=True,
        )
        self.dc_histogram_widget.set_upper_threshold(
            maximum,
            visible=False,
        )

    def apply_selected_filter(
        self,
        filter_name: str,
        kernel_size: int,
        repetitions: int,
    ) -> None:
        if self.dc is None or self.original_g is None or self.original_s is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Open an image before applying a filter.",
            )
            return

        if filter_name == "None":
            self.reset_filter()
            return

        if filter_name != "Median":
            QMessageBox.warning(
                self,
                "Unsupported filter",
                f"Filter '{filter_name}' is not implemented.",
            )
            return

        if kernel_size % 2 == 0:
            QMessageBox.warning(
                self,
                "Invalid kernel size",
                "Median-filter kernel size must be odd.",
            )
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            _, filtered_g, filtered_s = phasor_filter_median(
                self.dc,
                self.original_g,
                self.original_s,
                size=kernel_size,
                repeat=repetitions,
            )
        except (ValueError, TypeError, RuntimeError) as error:
            QMessageBox.critical(self, "Filtering failed", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.filtered_g = np.asarray(filtered_g, dtype=np.float64)
        self.filtered_s = np.asarray(filtered_s, dtype=np.float64)
        self.active_filter_description = (
            f"Median {kernel_size}×{kernel_size}, {repetitions} repetition(s)"
        )
        self.apply_current_threshold()

    def reset_filter(self) -> None:
        if self.original_g is None or self.original_s is None:
            return

        self.filtered_g = self.original_g.copy()
        self.filtered_s = self.original_s.copy()
        self.active_filter_description = "No filter"
        self.apply_current_threshold()

    def apply_threshold(
        self,
        minimum: float,
        upper_enabled: bool,
        maximum: float,
    ) -> None:
        if self.dc is None or self.filtered_g is None or self.filtered_s is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Open an image before applying thresholds.",
            )
            return

        minimum = float(minimum)
        maximum = float(maximum)

        if upper_enabled and maximum <= minimum:
            QMessageBox.warning(
                self,
                "Invalid thresholds",
                "Maximum DC must be greater than Minimum DC.",
            )
            return

        self.active_threshold = minimum
        self.active_upper_threshold = maximum if upper_enabled else None

        self.apply_current_threshold()

    def apply_current_threshold(self) -> None:
        if self.dc is None or self.filtered_g is None or self.filtered_s is None:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self.active_threshold is None:
                current_g = self.filtered_g.copy()
                current_s = self.filtered_s.copy()
            else:
                _, thresholded_g, thresholded_s = phasor_threshold(
                    self.dc,
                    self.filtered_g,
                    self.filtered_s,
                    mean_min=self.active_threshold,
                )
                current_g = np.asarray(
                    thresholded_g,
                    dtype=np.float64,
                )
                current_s = np.asarray(
                    thresholded_s,
                    dtype=np.float64,
                )

            if self.active_upper_threshold is not None:
                saturated = ~np.isfinite(self.dc) | (self.dc >= self.active_upper_threshold)
                current_g = current_g.copy()
                current_s = current_s.copy()
                current_g[saturated] = np.nan
                current_s[saturated] = np.nan

            self.g = current_g
            self.s = current_s
        except (ValueError, TypeError, RuntimeError) as error:
            QMessageBox.critical(self, "Thresholding failed", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.update_phasor_view()
        self.update_selection()
        self.update_dc_histogram()

    def reset_threshold(self) -> None:
        if self.filtered_g is None or self.filtered_s is None:
            return

        self.active_threshold = None
        self.active_upper_threshold = None
        self.processing_widget.set_upper_threshold_enabled(False)
        self.g = self.filtered_g.copy()
        self.s = self.filtered_s.copy()

        self.update_phasor_view()
        self.update_selection()
        self.update_dc_histogram()

    def threshold_spinbox_changed(self, value: float) -> None:
        if self.dc is not None:
            self.dc_histogram_widget.set_threshold(
                float(value),
                visible=True,
            )

    def upper_threshold_enabled_changed(self, enabled: bool) -> None:
        if self.dc is None:
            return

        self.dc_histogram_widget.set_upper_threshold(
            self.processing_widget.upper_threshold_value(),
            visible=bool(enabled),
        )

    def upper_threshold_spinbox_changed(self, value: float) -> None:
        if self.dc is None:
            return

        self.dc_histogram_widget.set_upper_threshold(
            float(value),
            visible=self.processing_widget.upper_threshold_enabled(),
        )

    def threshold_line_moved(self, value: float) -> None:
        if self.dc is None:
            return

        threshold = max(0.0, float(value))
        threshold = min(
            threshold,
            self.processing_widget.threshold_maximum(),
        )
        self.processing_widget.set_threshold_value(threshold)

    def upper_threshold_line_moved(self, value: float) -> None:
        if self.dc is None:
            return

        threshold = max(0.0, float(value))
        threshold = min(
            threshold,
            self.processing_widget.threshold_maximum(),
        )
        self.processing_widget.set_upper_threshold_value(threshold)

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
        mode = self.processing_widget.mode()

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

        self._add_selection_roi(
            position=position,
            size=size,
        )

    def _add_selection_roi(
        self,
        *,
        position: tuple[float, float],
        size: tuple[float, float],
    ) -> None:
        if len(self.phasor_rois) >= self.MAX_SELECTIONS:
            return

        selection_index = len(self.phasor_rois)
        color = self.selection_colors[selection_index]

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

        self.phasor_widget.add_roi(roi)
        self.phasor_rois.append(roi)

        self.update_selection()

    def remove_last_selection(self) -> None:
        if not self.phasor_rois:
            return

        roi = self.phasor_rois.pop()
        self.phasor_widget.remove_roi(roi)

        has_selections = bool(self.phasor_rois)
        self.remove_selection_button.setEnabled(has_selections)
        self.export_button.setEnabled(has_selections)
        self.export_statistics_button.setEnabled(has_selections)
        self.add_selection_button.setEnabled(True)

        self.update_selection()

    def clear_selections(self) -> None:
        for roi in self.phasor_rois:
            self.phasor_widget.remove_roi(roi)

        self.phasor_rois.clear()
        self.remove_selection_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_statistics_button.setEnabled(False)
        self.add_selection_button.setEnabled(self.g is not None and self.s is not None)

        if self.dc is not None:
            self.dc_image_widget.clear_overlay(self.dc.shape)

    def save_session(self) -> None:
        if self.current_path is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Open an image before saving a session.",
            )
            return

        default_name = f"{self.current_path.stem}_phasora_session.json"

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Phasora session",
            default_name,
            "Phasora session (*.json)",
        )

        if not file_name:
            return

        output_path = Path(file_name)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")

        cursors = [
            CursorState(
                x=float(roi.pos().x()),
                y=float(roi.pos().y()),
                width=float(roi.size().x()),
                height=float(roi.size().y()),
            )
            for roi in self.phasor_rois
        ]

        x_range, y_range = self.phasor_widget.view_range()

        active_filter_name = (
            "None"
            if self.active_filter_description == "No filter"
            else self.processing_widget.filter_name()
        )

        session = PhasoraSession(
            image_path=str(self.current_path.resolve()),
            mode=self.processing_widget.mode().value,
            histogram_bins=self.processing_widget.histogram_bins(),
            colormap=self.processing_widget.colormap_name(),
            filter_name=active_filter_name,
            filter_size=self.processing_widget.filter_size(),
            filter_repetitions=(self.processing_widget.filter_repetitions()),
            minimum_threshold=self.active_threshold,
            maximum_threshold=self.active_upper_threshold,
            cursors=cursors,
            view_range=ViewRangeState(
                x_min=x_range[0],
                x_max=x_range[1],
                y_min=y_range[0],
                y_max=y_range[1],
            ),
        )

        try:
            session.save(output_path)
        except (ValueError, OSError) as error:
            QMessageBox.critical(
                self,
                "Unable to save session",
                str(error),
            )
            return

        QMessageBox.information(
            self,
            "Session saved",
            f"Session saved to:\n{output_path}",
        )

    def load_session(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Phasora session",
            "",
            "Phasora session (*.json)",
        )

        if not file_name:
            return

        try:
            session = PhasoraSession.load(file_name)
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
        ) as error:
            QMessageBox.critical(
                self,
                "Unable to load session",
                str(error),
            )
            return

        image_path = Path(session.image_path)

        if not image_path.exists():
            QMessageBox.critical(
                self,
                "Image not found",
                (f"The image referenced by this session no longer exists:\n{image_path}"),
            )
            return

        if not self._load_image_path(
            image_path,
            add_default_cursor=False,
        ):
            return

        try:
            self.processing_widget.set_mode(session.mode)
            self.processing_widget.set_histogram_bins(session.histogram_bins)
            self.processing_widget.set_colormap_name(session.colormap)
            self.processing_widget.set_filter_settings(
                name=session.filter_name,
                size=session.filter_size,
                repetitions=session.filter_repetitions,
            )
        except ValueError as error:
            QMessageBox.critical(
                self,
                "Invalid session settings",
                str(error),
            )
            return

        self.update_colormap()
        self.update_display_settings()

        if session.filter_name == "Median":
            self.apply_selected_filter(
                session.filter_name,
                session.filter_size,
                session.filter_repetitions,
            )
        else:
            self.reset_filter()

        if session.minimum_threshold is None:
            self.active_threshold = None
        else:
            self.processing_widget.set_threshold_value(session.minimum_threshold)
            self.active_threshold = session.minimum_threshold

        if session.maximum_threshold is None:
            self.processing_widget.set_upper_threshold_enabled(False)
            self.active_upper_threshold = None
        else:
            self.processing_widget.set_upper_threshold_value(session.maximum_threshold)
            self.processing_widget.set_upper_threshold_enabled(True)
            self.active_upper_threshold = session.maximum_threshold

        self.apply_current_threshold()

        self.clear_selections()

        for cursor in session.cursors[: self.MAX_SELECTIONS]:
            self._add_selection_roi(
                position=(cursor.x, cursor.y),
                size=(cursor.width, cursor.height),
            )

        if session.view_range is not None:
            self.phasor_widget.set_view_range(
                x_range=(
                    session.view_range.x_min,
                    session.view_range.x_max,
                ),
                y_range=(
                    session.view_range.y_min,
                    session.view_range.y_max,
                ),
            )

        self.update_selection()

        QMessageBox.information(
            self,
            "Session loaded",
            f"Session loaded from:\n{file_name}",
        )

    def _cursor_mask(
        self,
        roi: pg.CircleROI,
        *,
        valid_mask: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        if self.g is None or self.s is None:
            raise RuntimeError("Phasor coordinates are not available.")

        roi_position = roi.pos()
        roi_size = roi.size()

        center_g = float(roi_position.x() + roi_size.x() / 2.0)
        center_s = float(roi_position.y() + roi_size.y() / 2.0)
        radius = float(
            min(
                roi_size.x(),
                roi_size.y(),
            )
            / 2.0
        )

        mask = circular_phasor_mask(
            g=self.g,
            s=self.s,
            center_g=center_g,
            center_s=center_s,
            radius=radius,
            valid_mask=valid_mask,
        )

        return center_g, center_s, radius, mask

    def export_current_cursor_statistics(self) -> None:
        if self.dc is None or self.g is None or self.s is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Open an image before exporting cursor statistics.",
            )
            return

        if not self.phasor_rois:
            QMessageBox.warning(
                self,
                "No cursors",
                "Add at least one phasor cursor before exporting statistics.",
            )
            return

        default_name = (
            f"{self.current_path.stem}_cursor_statistics.csv"
            if self.current_path is not None
            else "cursor_statistics.csv"
        )

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export cursor statistics",
            default_name,
            "CSV file (*.csv)",
        )

        if not file_name:
            return

        output_path = Path(file_name)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")

        valid = np.isfinite(self.dc) & np.isfinite(self.g) & np.isfinite(self.s)
        valid_pixels = int(np.count_nonzero(valid))

        statistics: list[CursorStatistics] = []

        for index, roi in enumerate(
            self.phasor_rois,
            start=1,
        ):
            center_g, center_s, radius, mask = self._cursor_mask(
                roi,
                valid_mask=valid,
            )

            selected_pixels = int(np.count_nonzero(mask))
            selected_percentage = (
                100.0 * selected_pixels / valid_pixels if valid_pixels > 0 else 0.0
            )

            statistics.append(
                CursorStatistics.from_arrays(
                    cursor_index=index,
                    center_g=center_g,
                    center_s=center_s,
                    radius=radius,
                    selected_pixels=selected_pixels,
                    selected_percentage=selected_percentage,
                    g=self.g[mask],
                    s=self.s[mask],
                    dc=self.dc[mask],
                )
            )

        try:
            export_cursor_statistics(
                output_path=output_path,
                statistics=statistics,
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
            f"Cursor statistics exported to:\n{output_path}",
        )

    def export_phasor_image(self) -> None:
        if self.g is None or self.s is None:
            QMessageBox.warning(
                self,
                "No phasor available",
                "Open an image before exporting the phasor plot.",
            )
            return

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export phasor image",
            "phasor_plot.png",
            ("PNG image (*.png);;TIFF image (*.tif *.tiff)"),
        )

        if not file_name:
            return

        output_path = Path(file_name)

        if output_path.suffix.lower() not in {
            ".png",
            ".tif",
            ".tiff",
        }:
            if "TIFF" in selected_filter:
                output_path = output_path.with_suffix(".tif")
            else:
                output_path = output_path.with_suffix(".png")

        resolution_labels = [
            "Normal — 100 DPI",
            "High — 300 DPI",
            "Super high — 600 DPI",
        ]

        selected_resolution, accepted = QInputDialog.getItem(
            self,
            "Export resolution",
            "Resolution:",
            resolution_labels,
            1,
            False,
        )

        if not accepted:
            return

        dpi_by_label = {
            "Normal — 100 DPI": 100,
            "High — 300 DPI": 300,
            "Super high — 600 DPI": 600,
        }
        dpi = dpi_by_label[selected_resolution]

        background_labels = [
            "White",
            "Black",
            "Transparent",
        ]

        selected_background, accepted = QInputDialog.getItem(
            self,
            "Export background",
            "Background:",
            background_labels,
            0,
            False,
        )

        if not accepted:
            return

        background_by_label = {
            "White": "white",
            "Black": "black",
            "Transparent": "transparent",
        }
        background = background_by_label[selected_background]

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            self.phasor_widget.export_image(
                output_path,
                dpi=dpi,
                background=background,
            )
        except (
            ValueError,
            OSError,
            RuntimeError,
        ) as error:
            QMessageBox.critical(
                self,
                "Export failed",
                str(error),
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self,
            "Export complete",
            (f"Phasor image exported at {dpi} DPI with {background} background to:\n{output_path}"),
        )

    def export_current_selections(self) -> None:
        if self.dc is None or self.g is None or self.s is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Open an image before exporting selections.",
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

        valid = np.isfinite(self.dc) & np.isfinite(self.g) & np.isfinite(self.s)
        selections: list[SelectionResult] = []

        for index, roi in enumerate(self.phasor_rois, start=1):
            roi_position = roi.pos()
            roi_size = roi.size()

            center_g = float(roi_position.x() + roi_size.x() / 2.0)
            center_s = float(roi_position.y() + roi_size.y() / 2.0)
            radius = float(min(roi_size.x(), roi_size.y()) / 2.0)

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
            QMessageBox.critical(self, "Export failed", str(error))
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Selections were exported to:\n{output_directory}",
        )

    def update_dc_view(self) -> None:
        if self.dc is None:
            return

        self.dc_image_widget.set_image(self.dc)

    def update_dc_histogram(self) -> None:
        if self.dc is None:
            self.dc_histogram_widget.clear()
            return

        finite_dc = self.dc[np.isfinite(self.dc) & (self.dc >= 0)]
        if finite_dc.size == 0:
            self.dc_histogram_widget.clear()
            return

        positive_dc = finite_dc[finite_dc > 0]
        upper = (
            float(np.percentile(positive_dc, 99.9))
            if positive_dc.size > 0
            else float(np.max(finite_dc))
        )
        if not np.isfinite(upper) or upper <= 0:
            upper = 1.0

        counts, edges = np.histogram(
            finite_dc,
            bins=self.DC_HISTOGRAM_BINS,
            range=(0.0, upper),
        )
        centers = (edges[:-1] + edges[1:]) / 2.0

        line_value = (
            self.active_threshold
            if self.active_threshold is not None
            else self.processing_widget.threshold_value()
        )

        self.dc_histogram_widget.set_histogram(
            centers=centers,
            counts=counts,
            x_max=upper,
        )
        self.dc_histogram_widget.set_threshold(
            float(line_value),
            visible=True,
        )

        upper_line_value = (
            self.active_upper_threshold
            if self.active_upper_threshold is not None
            else self.processing_widget.upper_threshold_value()
        )
        self.dc_histogram_widget.set_upper_threshold(
            float(upper_line_value),
            visible=(
                self.active_upper_threshold is not None
                or self.processing_widget.upper_threshold_enabled()
            ),
        )

    def update_phasor_view(self) -> None:
        if self.dc is None or self.g is None or self.s is None:
            return

        mode = self.processing_widget.mode()
        x_range, y_range = phasor_plot_ranges(mode)
        valid = np.isfinite(self.dc) & np.isfinite(self.g) & np.isfinite(self.s)

        histogram = compute_phasor_histogram(
            self.g,
            self.s,
            bins=self.processing_widget.histogram_bins(),
            x_range=x_range,
            y_range=y_range,
            valid_mask=valid,
            log_scale=True,
        )

        self.phasor_widget.set_histogram(
            histogram=histogram,
            x_range=x_range,
            y_range=y_range,
            title=f"{mode.value} phasor histogram",
        )

        self.update_reference_curve()

    def update_reference_curve(self) -> None:
        mode = self.processing_widget.mode()
        if mode == PhasorMode.LIFETIME:
            curve_g, curve_s = universal_semicircle()
        else:
            theta = np.linspace(0.0, 2.0 * np.pi, 512)
            curve_g = np.cos(theta)
            curve_s = np.sin(theta)

        self.phasor_widget.set_reference_curve(curve_g, curve_s)
        self.phasor_widget.set_angular_reference_grid(visible=(mode == PhasorMode.SPECTRAL))

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
                [(0, 0, 4), (87, 15, 109), (249, 142, 8), (252, 255, 164)],
            ),
            "Viridis": pg.ColorMap(
                [0.0, 0.35, 0.70, 1.0],
                [(68, 1, 84), (49, 104, 142), (53, 183, 121), (253, 231, 37)],
            ),
            "Magma": pg.ColorMap(
                [0.0, 0.35, 0.70, 1.0],
                [(0, 0, 4), (114, 31, 129), (241, 96, 93), (252, 253, 191)],
            ),
            "Cyan": pg.ColorMap(
                [0.0, 0.50, 1.0],
                [(0, 0, 0), (0, 100, 130), (180, 255, 255)],
            ),
            "Grayscale": pg.ColorMap(
                [0.0, 1.0],
                [(0, 0, 0), (255, 255, 255)],
            ),
            "Spectral": pg.ColorMap(
                [0.0, 0.16, 0.33, 0.50, 0.66, 0.83, 1.0],
                [
                    (0, 0, 0),
                    (75, 0, 130),
                    (0, 80, 255),
                    (0, 220, 220),
                    (80, 220, 80),
                    (255, 220, 0),
                    (255, 40, 40),
                ],
            ),
            "Red-Green": pg.ColorMap(
                [0.0, 0.35, 0.65, 1.0],
                [(0, 0, 0), (180, 0, 0), (255, 220, 0), (0, 255, 80)],
            ),
        }

        selected_name = self.processing_widget.colormap_name()
        color_map = colormaps.get(selected_name, colormaps["Inferno"])
        self.phasor_widget.set_lookup_table(color_map.getLookupTable(start=0.0, stop=1.0, nPts=256))

    def update_selection(
        self,
        _roi: object | None = None,
    ) -> None:
        if self.dc is None or self.g is None or self.s is None:
            return

        valid = np.isfinite(self.dc) & np.isfinite(self.g) & np.isfinite(self.s)
        overlay = np.zeros((*self.dc.shape, 4), dtype=np.uint8)
        selection_summaries: list[str] = []
        valid_pixels = int(np.count_nonzero(valid))

        for index, roi in enumerate(self.phasor_rois):
            roi_position = roi.pos()
            roi_size = roi.size()

            center_g = float(roi_position.x() + roi_size.x() / 2.0)
            center_s = float(roi_position.y() + roi_size.y() / 2.0)
            radius = float(min(roi_size.x(), roi_size.y()) / 2.0)

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
            selected_percentage = (
                100.0 * selected_pixels / valid_pixels if valid_pixels > 0 else 0.0
            )
            selection_summaries.append(
                f"C{index + 1}: "
                f"g={center_g:.3f}, "
                f"s={center_s:.3f}, "
                f"r={radius:.3f}, "
                f"{selected_pixels:,} px "
                f"({selected_percentage:.2f}%)"
            )

        self.dc_image_widget.set_overlay(overlay)

        file_name = self.current_path.name if self.current_path is not None else "Unknown"
        minimum_threshold_text = (
            "None" if self.active_threshold is None else f"{self.active_threshold:.2f}"
        )
        maximum_threshold_text = (
            "None" if self.active_upper_threshold is None else f"{self.active_upper_threshold:.2f}"
        )
        selection_text = (
            " | ".join(selection_summaries) if selection_summaries else "No active cursors"
        )

        has_selections = bool(self.phasor_rois)
        self.remove_selection_button.setEnabled(has_selections)
        self.export_button.setEnabled(has_selections)
        self.export_statistics_button.setEnabled(has_selections)
        self.add_selection_button.setEnabled(
            self.g is not None
            and self.s is not None
            and len(self.phasor_rois) < self.MAX_SELECTIONS
        )

        self.status_label.setText(
            f"File: {file_name} | "
            f"Mode: {self.processing_widget.mode().value} | "
            f"Histogram bins: {self.processing_widget.histogram_bins()} | "
            f"Filter: {self.active_filter_description} | "
            f"DC minimum: {minimum_threshold_text} | "
            f"DC maximum: {maximum_threshold_text} | "
            f"Valid pixels: {valid_pixels:,} | "
            f"{selection_text}"
        )
