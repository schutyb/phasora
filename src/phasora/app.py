from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Phasora")
        self.resize(1000, 700)

        self.title_label = QLabel("Phasora v0.1")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            """
            QLabel {
                font-size: 32px;
                font-weight: bold;
            }
            """
        )

        self.open_button = QPushButton("Open TIFF")
        self.open_button.setMinimumHeight(45)
        self.open_button.clicked.connect(self.open_tiff)

        self.status_label = QLabel(
            "Open a multichannel TIFF image to calculate its phasor."
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(self.title_label)
        layout.addSpacing(20)
        layout.addWidget(self.open_button)
        layout.addSpacing(20)
        layout.addWidget(self.status_label)
        layout.addStretch()

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

        self.show_image_summary(
            path=Path(file_name),
            image=image,
            dc=dc,
            g=g,
            s=s,
        )

    def show_image_summary(
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
            f"File: {path.name}\n"
            f"Shape: {height} × {width} × {channels}\n"
            f"Data type: {image.dtype}\n"
            f"Mean DC: {mean_dc:.4f}\n"
            f"Mean g: {mean_g:.4f}\n"
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