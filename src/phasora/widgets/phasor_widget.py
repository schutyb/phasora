from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import tifffile
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PhasorWidget(QWidget):
    """2D phasor histogram with reference curve and interactive ROIs."""

    ANGULAR_STEP_DEGREES = 30
    SPECTRAL_MIN = -1.2
    SPECTRAL_MAX = 1.2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.graphics = pg.GraphicsLayoutWidget()

        self.plot = self.graphics.addPlot(
            title="Spectral phasor histogram",
        )
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "Real")
        self.plot.setLabel("left", "Imaginary")

        bottom_axis = self.plot.getAxis("bottom")
        left_axis = self.plot.getAxis("left")
        bottom_axis.setStyle(
            tickLength=-6,
            tickTextOffset=4,
            autoExpandTextSpace=False,
        )
        left_axis.setStyle(
            tickLength=-6,
            tickTextOffset=4,
            autoExpandTextSpace=False,
        )
        bottom_axis.setHeight(42)
        left_axis.setWidth(58)

        self.plot.setXRange(
            self.SPECTRAL_MIN,
            self.SPECTRAL_MAX,
            padding=0.0,
        )
        self.plot.setYRange(
            self.SPECTRAL_MIN,
            self.SPECTRAL_MAX,
            padding=0.0,
        )

        self.image_item = pg.ImageItem()
        self.image_item.setZValue(0)
        self.plot.addItem(self.image_item)

        self._lookup_table: np.ndarray | None = None

        self.reference_curve_item = pg.PlotDataItem(
            pen=pg.mkPen(
                color=(220, 220, 220),
                width=2,
            )
        )
        self.reference_curve_item.setZValue(5)
        self.plot.addItem(self.reference_curve_item)

        self._angular_grid_items: list[pg.GraphicsObject] = []
        self._angular_label_items: list[pg.TextItem] = []
        self._build_angular_reference_grid()
        self.set_angular_reference_grid(visible=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

    def _build_angular_reference_grid(self) -> None:
        standard_radial_pen = pg.mkPen(
            color=(185, 185, 185, 170),
            width=1.2,
            style=Qt.PenStyle.DotLine,
        )
        cardinal_radial_pen = pg.mkPen(
            color=(205, 205, 205, 190),
            width=1.5,
            style=Qt.PenStyle.SolidLine,
        )
        ring_pen = pg.mkPen(
            color=(185, 185, 185, 145),
            width=1.2,
            style=Qt.PenStyle.DotLine,
        )

        for angle_degrees in range(
            0,
            360,
            self.ANGULAR_STEP_DEGREES,
        ):
            angle_radians = np.deg2rad(angle_degrees)
            x = float(np.cos(angle_radians))
            y = float(np.sin(angle_radians))

            pen = cardinal_radial_pen if angle_degrees % 90 == 0 else standard_radial_pen

            radial_line = pg.PlotDataItem(
                [0.0, x],
                [0.0, y],
                pen=pen,
            )
            radial_line.setZValue(1)
            self.plot.addItem(radial_line)
            self._angular_grid_items.append(radial_line)

        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            361,
        )

        for radius in (1.0 / 3.0, 2.0 / 3.0):
            ring = pg.PlotDataItem(
                radius * np.cos(theta),
                radius * np.sin(theta),
                pen=ring_pen,
            )
            ring.setZValue(1)
            self.plot.addItem(ring)
            self._angular_grid_items.append(ring)

        cardinal_labels = {
            0: (1.06, 0.0),
            90: (0.0, 1.06),
            180: (-1.06, 0.0),
            270: (0.0, -1.06),
        }

        label_anchors = {
            0: (0.0, 0.5),
            90: (0.5, 1.0),
            180: (1.0, 0.5),
            270: (0.5, 0.0),
        }

        for angle_degrees, position in cardinal_labels.items():
            label = pg.TextItem(
                text=f"{angle_degrees}°",
                color=(225, 225, 225),
                anchor=label_anchors[angle_degrees],
            )
            label.setPos(*position)
            label.setZValue(10)
            self.plot.addItem(label)
            self._angular_label_items.append(label)

    def set_angular_reference_grid(self, *, visible: bool) -> None:
        """Show or hide the spectral angular reference grid."""

        for item in self._angular_grid_items:
            item.setVisible(bool(visible))

        for label in self._angular_label_items:
            label.setVisible(bool(visible))

    def set_histogram(
        self,
        *,
        histogram: np.ndarray,
        x_range: Sequence[float],
        y_range: Sequence[float],
        title: str,
    ) -> None:
        """Display a 2D histogram in phasor coordinates."""

        array = np.asarray(histogram)

        if array.ndim != 2:
            raise ValueError(f"Phasor histogram must be a 2D array, received shape {array.shape}.")

        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("Phasor plot ranges must contain exactly two values.")

        x_min = float(x_range[0])
        x_max = float(x_range[1])
        y_min = float(y_range[0])
        y_max = float(y_range[1])

        self.image_item.setImage(
            array,
            autoLevels=True,
        )
        self.image_item.setRect(
            QRectF(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min,
            )
        )

        spectral_circle_visible = x_min <= -1.0 and x_max >= 1.0 and y_min <= -1.0 and y_max >= 1.0

        if spectral_circle_visible:
            display_x_min = self.SPECTRAL_MIN
            display_x_max = self.SPECTRAL_MAX
            display_y_min = self.SPECTRAL_MIN
            display_y_max = self.SPECTRAL_MAX
        else:
            display_x_min = x_min
            display_x_max = x_max
            display_y_min = y_min
            display_y_max = y_max

        self.plot.enableAutoRange(
            x=False,
            y=False,
        )

        self.plot.setXRange(
            display_x_min,
            display_x_max,
            padding=0.0,
        )
        self.plot.setYRange(
            display_y_min,
            display_y_max,
            padding=0.0,
        )

        self.plot.getViewBox().setMouseEnabled(
            x=True,
            y=True,
        )

        self.plot.setTitle(title)

    def set_reference_curve(
        self,
        g: np.ndarray,
        s: np.ndarray,
    ) -> None:
        """Update the spectral unit circle or lifetime semicircle."""

        self.reference_curve_item.setData(
            np.asarray(g, dtype=np.float64),
            np.asarray(s, dtype=np.float64),
        )

    def set_lookup_table(self, lookup_table: np.ndarray) -> None:
        """Set the histogram colormap lookup table."""

        table = np.asarray(lookup_table, dtype=np.uint8)
        self._lookup_table = table.copy()
        self.image_item.setLookupTable(self._lookup_table)

    def _apply_export_theme(self, background: str) -> None:
        if background == "black":
            background_color = QColor(0, 0, 0, 255)
            foreground_color = QColor(230, 230, 230, 255)
            grid_color = QColor(195, 195, 195, 180)
            ring_color = QColor(185, 185, 185, 150)
        elif background == "white":
            background_color = QColor(255, 255, 255, 255)
            foreground_color = QColor(20, 20, 20, 255)
            grid_color = QColor(55, 55, 55, 185)
            ring_color = QColor(75, 75, 75, 155)
        else:
            background_color = QColor(255, 255, 255, 0)
            foreground_color = QColor(20, 20, 20, 255)
            grid_color = QColor(55, 55, 55, 185)
            ring_color = QColor(75, 75, 75, 155)

        self.graphics.setBackground(QBrush(background_color))
        self.plot.getViewBox().setBackgroundColor(background_color)

        for axis_name in ("left", "bottom"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(foreground_color))
            axis.setTextPen(pg.mkPen(foreground_color))

        self.plot.titleLabel.setAttr(
            "color",
            foreground_color.name(),
        )

        self.reference_curve_item.setPen(
            pg.mkPen(
                foreground_color,
                width=2,
            )
        )

        radial_count = 360 // self.ANGULAR_STEP_DEGREES

        for index, item in enumerate(self._angular_grid_items):
            if index < radial_count:
                angle_degrees = index * self.ANGULAR_STEP_DEGREES
                cardinal = angle_degrees % 90 == 0

                item.setPen(
                    pg.mkPen(
                        grid_color,
                        width=1.5 if cardinal else 1.2,
                        style=(Qt.PenStyle.SolidLine if cardinal else Qt.PenStyle.DotLine),
                    )
                )
            else:
                item.setPen(
                    pg.mkPen(
                        ring_color,
                        width=1.2,
                        style=Qt.PenStyle.DotLine,
                    )
                )

        for label in self._angular_label_items:
            label.setColor(foreground_color)

        self._apply_transparent_zero_lookup_table()

    def _apply_transparent_zero_lookup_table(self) -> None:
        """Make zero-count histogram pixels reveal the export background."""

        if self._lookup_table is None or self._lookup_table.size == 0:
            return

        table = self._lookup_table

        if table.ndim != 2 or table.shape[1] not in {3, 4}:
            return

        if table.shape[1] == 3:
            alpha = np.full(
                (table.shape[0], 1),
                255,
                dtype=np.uint8,
            )
            export_table = np.concatenate(
                [table, alpha],
                axis=1,
            )
        else:
            export_table = table.copy()

        export_table[0, 3] = 0
        self.image_item.setLookupTable(export_table)

    def _restore_screen_theme(self) -> None:
        background_color = QColor(0, 0, 0, 255)
        foreground_color = QColor(230, 230, 230, 255)
        grid_color = QColor(195, 195, 195, 180)
        ring_color = QColor(185, 185, 185, 150)

        self.graphics.setBackground(QBrush(background_color))
        self.plot.getViewBox().setBackgroundColor(background_color)

        for axis_name in ("left", "bottom"):
            axis = self.plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(foreground_color))
            axis.setTextPen(pg.mkPen(foreground_color))

        self.plot.titleLabel.setAttr(
            "color",
            foreground_color.name(),
        )

        self.reference_curve_item.setPen(
            pg.mkPen(
                foreground_color,
                width=2,
            )
        )

        radial_count = 360 // self.ANGULAR_STEP_DEGREES

        for index, item in enumerate(self._angular_grid_items):
            if index < radial_count:
                angle_degrees = index * self.ANGULAR_STEP_DEGREES
                cardinal = angle_degrees % 90 == 0

                item.setPen(
                    pg.mkPen(
                        grid_color,
                        width=1.5 if cardinal else 1.2,
                        style=(Qt.PenStyle.SolidLine if cardinal else Qt.PenStyle.DotLine),
                    )
                )
            else:
                item.setPen(
                    pg.mkPen(
                        ring_color,
                        width=1.2,
                        style=Qt.PenStyle.DotLine,
                    )
                )

        for label in self._angular_label_items:
            label.setColor(foreground_color)

        if self._lookup_table is not None:
            self.image_item.setLookupTable(self._lookup_table)

    def export_image(
        self,
        output_path: str | Path,
        *,
        dpi: int,
        background: str = "white",
        width_inches: float = 7.0,
        height_inches: float = 7.0,
    ) -> None:
        """Export the complete phasor plot as PNG or TIFF."""

        path = Path(output_path)
        suffix = path.suffix.lower()

        if suffix not in {".png", ".tif", ".tiff"}:
            raise ValueError("Export format must be PNG or TIFF.")

        if dpi <= 0:
            raise ValueError("DPI must be greater than zero.")

        normalized_background = background.strip().lower()
        if normalized_background not in {
            "white",
            "black",
            "transparent",
        }:
            raise ValueError("Background must be white, black, or transparent.")

        requested_width = max(
            1,
            int(round(width_inches * dpi)),
        )

        if normalized_background == "white":
            export_background = QColor(255, 255, 255, 255)
        elif normalized_background == "black":
            export_background = QColor(0, 0, 0, 255)
        else:
            export_background = QColor(255, 255, 255, 0)

        self._apply_export_theme(normalized_background)

        try:
            exporter = ImageExporter(self.plot)
            parameters = exporter.parameters()

            parameters["width"] = requested_width
            parameters["antialias"] = True
            parameters["background"] = export_background

            image = exporter.export(toBytes=True)
        finally:
            self._restore_screen_theme()

        if image is None or image.isNull():
            raise RuntimeError("PyQtGraph was unable to render the phasor plot.")

        image = image.convertToFormat(QImage.Format.Format_RGBA8888)

        dots_per_meter = int(round(dpi / 0.0254))
        image.setDotsPerMeterX(dots_per_meter)
        image.setDotsPerMeterY(dots_per_meter)

        if suffix == ".png":
            if not image.save(str(path), "PNG"):
                raise OSError(f"Unable to save PNG image to {path}.")
            return

        width_pixels = image.width()
        height_pixels = image.height()
        bytes_per_line = image.bytesPerLine()
        buffer = image.bits()

        array = np.frombuffer(
            buffer,
            dtype=np.uint8,
            count=bytes_per_line * height_pixels,
        ).reshape(
            height_pixels,
            bytes_per_line,
        )

        rgba = (
            array[:, : width_pixels * 4]
            .reshape(
                height_pixels,
                width_pixels,
                4,
            )
            .copy()
        )

        tifffile.imwrite(
            path,
            rgba,
            photometric="rgb",
            resolution=(dpi, dpi),
            resolutionunit="INCH",
            metadata={
                "axes": "YXS",
                "dpi": dpi,
                "background": normalized_background,
            },
        )

    def view_range(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        ranges = self.plot.getViewBox().viewRange()

        return (
            (
                float(ranges[0][0]),
                float(ranges[0][1]),
            ),
            (
                float(ranges[1][0]),
                float(ranges[1][1]),
            ),
        )

    def set_view_range(
        self,
        *,
        x_range: Sequence[float],
        y_range: Sequence[float],
    ) -> None:
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("View ranges must contain exactly two values.")

        self.plot.setXRange(
            float(x_range[0]),
            float(x_range[1]),
            padding=0.0,
        )
        self.plot.setYRange(
            float(y_range[0]),
            float(y_range[1]),
            padding=0.0,
        )

    def add_roi(self, roi: pg.ROI) -> None:
        """Add an interactive ROI to the phasor plot."""

        roi.setZValue(20)
        self.plot.addItem(roi)

    def remove_roi(self, roi: pg.ROI) -> None:
        """Remove an interactive ROI from the phasor plot."""

        self.plot.removeItem(roi)
