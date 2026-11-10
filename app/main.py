from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = QApplication(sys.argv)
    application.setApplicationName("NVIDIA AI Video Studio")
    application.setOrganizationName("NVIDIA AI Video Studio")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
