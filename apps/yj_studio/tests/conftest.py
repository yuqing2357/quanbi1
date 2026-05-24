"""Shared pytest fixtures for the yj_studio test suite.

The project deliberately disables pytest-qt (see pyproject.toml). A few tests
need a ``QApplication`` to be alive (e.g. anything that touches QUndoStack or
widgets); we create one on demand as a session-scoped fixture so individual
tests can opt in.
"""

from __future__ import annotations

import os
import sys

import pytest

# Force the offscreen platform so GUI fixtures work in headless CI / shells.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app
