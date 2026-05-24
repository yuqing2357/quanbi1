"""Progress dialog that loads a Petrel reservoir grid on a worker thread.

Loading a real reservoir grid takes anywhere from ~15s (ZCORN cache
hit, properties only) to several minutes (first-time ZCORN cache
build for a multi-GB ASCII file). We can't block the UI thread for
that long, so the loader runs on a ``QThread`` and reports progress
back via signals into a ``QProgressDialog``.

Usage:

    grid = run_reservoir_load_dialog(parent, master_path)
    if grid is None:
        return  # user cancelled or load failed (already reported)

The returned grid is a fully-initialised ``ReservoirGrid`` ready to
register with ``ReservoirRegistry``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from yj_studio.reservoir import ReservoirGrid

logger = logging.getLogger(__name__)


class _LoaderWorker(QObject):
    """QObject wrapper around ``ReservoirGrid.load_from_master``."""

    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)    # ReservoirGrid
    failed = pyqtSignal(str)

    def __init__(self, master_path: Path) -> None:
        super().__init__()
        self._master_path = master_path
        # Cancellation is checked at chunk boundaries — the heavy
        # ZCORN cache build can take minutes, so we expose a flag the
        # progress callback raises into a CancellationError.
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            # Reserve the [0, 0.6] band for load_from_master, [0.6, 1.0]
            # for the downsample pass. Both run on this worker thread so
            # the dialog stays responsive throughout.
            def _load_cb(frac: float, msg: str) -> None:
                if self._cancelled:
                    raise _Cancelled()
                self.progress.emit(float(frac) * 0.6, str(msg))

            def _ds_cb(frac: float, msg: str) -> None:
                if self._cancelled:
                    raise _Cancelled()
                self.progress.emit(0.6 + float(frac) * 0.4, str(msg))

            grid = ReservoirGrid.load_from_master(
                self._master_path, progress_cb=_load_cb
            )
            # Build the 3D overview now — otherwise the first render
            # would block the UI thread for several minutes.
            self.progress.emit(0.6, "Building 3D overview (downsampling)")
            grid.downsampled(progress_cb=_ds_cb)
            self.succeeded.emit(grid)
        except _Cancelled:
            self.failed.emit("用户取消")
        except Exception as exc:  # noqa: BLE001 — worker boundary
            logger.exception("Reservoir grid load failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _Cancelled(Exception):
    pass


def run_reservoir_load_dialog(
    parent: QWidget | None,
    master_path: Path,
) -> Optional[ReservoirGrid]:
    """Show a modal progress dialog while a worker loads the grid.

    Returns the loaded ``ReservoirGrid`` on success, ``None`` if the
    user cancelled or an error occurred (an error message box is
    shown to the user automatically).
    """

    dialog = QProgressDialog(
        f"加载储层模型\n{master_path.name}",
        "取消",
        0,
        100,
        parent,
    )
    dialog.setWindowTitle("加载储层模型")
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)

    thread = QThread()
    worker = _LoaderWorker(master_path)
    worker.moveToThread(thread)

    result: dict[str, object] = {"grid": None, "error": None}

    def _on_progress(frac: float, msg: str) -> None:
        dialog.setValue(int(round(frac * 100)))
        dialog.setLabelText(f"加载储层模型\n{master_path.name}\n\n{msg}")

    def _on_succeeded(grid_obj: object) -> None:
        result["grid"] = grid_obj
        thread.quit()

    def _on_failed(message: str) -> None:
        result["error"] = message
        thread.quit()

    worker.progress.connect(_on_progress)
    worker.succeeded.connect(_on_succeeded)
    worker.failed.connect(_on_failed)
    thread.started.connect(worker.run)
    dialog.canceled.connect(worker.cancel)

    thread.start()
    # Spin the local event loop via exec_() — QProgressDialog blocks
    # the caller until the user cancels or we explicitly close it.
    # We close it manually when the thread finishes.
    def _close_dialog() -> None:
        dialog.setValue(100)
        dialog.close()

    thread.finished.connect(_close_dialog)
    dialog.exec()
    # Wait for the worker thread to actually finish before tearing it down,
    # so we don't outrun its last signal.
    thread.wait(60_000)

    if result["error"] is not None:
        QMessageBox.warning(parent, "加载储层模型", str(result["error"]))
        return None
    return result["grid"]    # type: ignore[return-value]
