from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from . import __version__
from .logging_config import configure_logging
from .ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Create or reuse the QApplication configured for the desktop app."""

    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("YJ Studio")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("YJ Research")
    app.setFont(QFont("Microsoft YaHei", 10))
    return app


def run(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    app = create_application(argv)
    window = MainWindow()
    window.show()
    logger.info("YJ Studio %s started", __version__)
    return int(app.exec())

