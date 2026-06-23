import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Phasora")
        self.resize(1000, 700)

        title = QLabel("Phasora v0.1")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            """
            QLabel {
                font-size: 32px;
                font-weight: bold;
            }
            """
        )

        self.setCentralWidget(title)


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())