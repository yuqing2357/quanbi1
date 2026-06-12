from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console and rotating file logging once."""

    root = logging.getLogger()
    if root.handlers:
        return

    log_dir = Path.home() / ".yj_studio" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_dir / "yj_studio.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

