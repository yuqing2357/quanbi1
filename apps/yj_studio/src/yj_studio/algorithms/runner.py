"""Main-process front-end for running algorithms.

The runner spawns a ``multiprocessing.Process`` that imports the algorithm by
its ``module:Class`` path and streams messages back through a ``Queue``. A
``QTimer`` polls the queue without blocking the UI event loop and re-emits
each message as a Qt signal that ``AlgorithmDock`` consumes.

The same runner is intended to host the SAM3 subprocess in Phase 9: that
worker will live in a different conda environment, but the protocol is the
same — only the spawn mechanism differs.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import traceback
from typing import Any, Protocol

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from yj_studio.scene.layer import Layer
from yj_studio.scene.layer_store import LayerStore

from .algorithm import Algorithm
from .context import AlgorithmContext
from .protocol import CancellationError
from .result import AlgorithmResult
from .serialization import layer_to_payload, payload_to_layer
from .worker import run_worker

logger = logging.getLogger(__name__)


class _TaskSignals(Protocol):
    """Common shape every task handle exposes — algorithm dock relies on
    duck-typing the four pyqtSignal attributes below.
    """

    progress: pyqtSignal
    finished: pyqtSignal
    errored: pyqtSignal
    cancelled: pyqtSignal

    def cancel(self) -> None: ...


class AlgorithmTask(QObject):
    """Handle to an in-flight algorithm run.

    Emits ``progress`` / ``finished`` / ``errored`` / ``cancelled``. Owners
    should call :meth:`cancel` to soft-cancel the worker.
    """

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list, str)  # output layers, summary
    errored = pyqtSignal(str, str)  # message, traceback
    cancelled = pyqtSignal()

    def __init__(
        self,
        algorithm_cls: type[Algorithm],
        params: dict[str, Any],
        input_layers: dict[str, Layer],
        *,
        layer_store: LayerStore | None = None,
        parent: QObject | None = None,
        auto_attach_outputs: bool = False,
    ) -> None:
        super().__init__(parent)
        self._algorithm_cls = algorithm_cls
        self._params = params
        self._input_layers = input_layers
        self._layer_store = layer_store
        self._process: mp.Process | None = None
        self._inbox: mp.Queue | None = None
        self._outbox: mp.Queue | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._drain_outbox)
        # MainWindow attaches output layers through QUndoStack so the user can
        # undo a whole algorithm run; tests opt back into the auto path.
        self._auto_attach_outputs = auto_attach_outputs
        self._finished_emitted = False

    # ---------------------------------------------------------- lifecycle

    def start(self) -> None:
        ctx_inputs = {role: layer_to_payload(layer) for role, layer in self._input_layers.items()}
        message = {
            "kind": "run",
            "algorithm": self._algorithm_cls.import_path(),
            "params": self._params,
            "input_layers": ctx_inputs,
        }
        self._inbox = mp.Queue()
        self._outbox = mp.Queue()
        self._inbox.put(message)
        self._process = mp.Process(
            target=run_worker,
            args=(self._inbox, self._outbox),
            name=f"yj-algo-{self._algorithm_cls.id}",
            daemon=True,
        )
        self._process.start()
        self._timer.start()

    def cancel(self) -> None:
        if self._inbox is not None:
            try:
                self._inbox.put({"kind": "cancel"})
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------- internals

    def _drain_outbox(self) -> None:
        if self._outbox is None:
            return
        try:
            while True:
                message = self._outbox.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
        # If the worker process died without emitting a terminal message,
        # surface that as an error so the UI doesn't hang on a spinner.
        if (
            not self._finished_emitted
            and self._process is not None
            and not self._process.is_alive()
        ):
            self._stop_timer()
            self._finished_emitted = True
            self.errored.emit("Algorithm worker exited unexpectedly", "")

    def _handle_message(self, message: dict[str, Any]) -> None:
        kind = message.get("kind")
        if kind == "progress":
            self.progress.emit(float(message.get("fraction", 0.0)), str(message.get("message", "")))
        elif kind == "log":
            logger.log(
                _level_for(message.get("level", "info")),
                "[%s] %s",
                self._algorithm_cls.id,
                message.get("message", ""),
            )
        elif kind == "done":
            self._stop_timer()
            self._finished_emitted = True
            output_layers = [payload_to_layer(p) for p in message.get("output_layers", [])]
            if self._auto_attach_outputs and self._layer_store is not None:
                for layer in output_layers:
                    self._layer_store.add(layer)
            self.finished.emit(output_layers, str(message.get("summary", "")))
        elif kind == "error":
            self._stop_timer()
            self._finished_emitted = True
            self.errored.emit(str(message.get("message", "")), str(message.get("traceback", "")))
        elif kind == "cancelled":
            self._stop_timer()
            self._finished_emitted = True
            self.cancelled.emit()

    def _stop_timer(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self._process is not None:
            self._process.join(timeout=0.5)


class _InProcessWorker(QObject):
    """Runs ``algorithm.run`` on a worker thread. Lives until ``finished``."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list, str)
    errored = pyqtSignal(str, str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        algorithm_cls: type[Algorithm],
        params: dict[str, Any],
        input_layers: dict[str, Layer],
        services: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._algorithm_cls = algorithm_cls
        self._params = params
        self._input_layers = input_layers
        self._services = dict(services or {})
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            algorithm = self._algorithm_cls()
            params_model = self._algorithm_cls.input_schema.model_validate(self._params)
            ctx = AlgorithmContext(
                input_layers=dict(self._input_layers),
                params=params_model,
                progress_callback=lambda f, m: self.progress.emit(float(f), str(m)),
                cancel_checker=lambda: self._cancel_requested,
                services=dict(self._services),
            )
            result = algorithm.run(ctx)
            self.finished.emit(list(result.output_layers), result.summary or "")
        except CancellationError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.errored.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())


class InProcessAlgorithmTask(QObject):
    """``AlgorithmTask``-shaped handle for algorithms that must run in the main
    process (e.g. SAM3, which holds GPU model state). Uses ``QThread`` so the
    UI event loop keeps spinning while the algorithm runs.
    """

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list, str)
    errored = pyqtSignal(str, str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        algorithm_cls: type[Algorithm],
        params: dict[str, Any],
        input_layers: dict[str, Layer],
        *,
        layer_store: LayerStore | None = None,
        parent: QObject | None = None,
        auto_attach_outputs: bool = False,
        services: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._layer_store = layer_store
        self._auto_attach_outputs = auto_attach_outputs
        self._thread = QThread(self)
        self._worker = _InProcessWorker(algorithm_cls, params, input_layers, services)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.errored.connect(self._on_errored)
        self._worker.cancelled.connect(self._on_cancelled)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._worker.request_cancel()

    def _on_finished(self, layers: list[Layer], summary: str) -> None:
        if self._auto_attach_outputs and self._layer_store is not None:
            for layer in layers:
                self._layer_store.add(layer)
        self.finished.emit(layers, summary)
        self._cleanup_thread()

    def _on_errored(self, message: str, tb: str) -> None:
        self.errored.emit(message, tb)
        self._cleanup_thread()

    def _on_cancelled(self) -> None:
        self.cancelled.emit()
        self._cleanup_thread()

    def _cleanup_thread(self) -> None:
        self._thread.quit()
        self._thread.wait(2000)


def _level_for(level: str) -> int:
    return {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }.get(level.lower(), logging.INFO)


class AlgorithmRunner(QObject):
    """Factory for :class:`AlgorithmTask` plus an in-process fallback path."""

    def __init__(
        self,
        layer_store: LayerStore | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._layer_store = layer_store
        self._services: dict[str, Any] = {}

    def register_service(self, name: str, service: Any) -> None:
        """Make ``service`` available to in-process algorithms via
        ``ctx.services[name]``. Subprocess algorithms never see it."""

        self._services[name] = service

    def unregister_service(self, name: str) -> None:
        self._services.pop(name, None)

    def submit(
        self,
        algorithm_cls: type[Algorithm],
        params: dict[str, Any],
        input_layers: dict[str, Layer],
        *,
        auto_attach_outputs: bool = False,
    ) -> "AlgorithmTask | InProcessAlgorithmTask":
        """Start ``algorithm_cls`` and return a task handle.

        Routes to a worker process or a worker thread based on
        ``algorithm_cls.runs_in_subprocess``. Both task types expose the same
        Qt signals (``progress / finished / errored / cancelled``), so the
        AlgorithmDock can treat them uniformly.

        ``auto_attach_outputs=False`` (default) means the caller listens to
        ``task.finished`` and is responsible for adding output layers to the
        store — typically through an undo command. Tests can set ``True`` to
        skip the wrapping.
        """

        if algorithm_cls.runs_in_subprocess:
            task: AlgorithmTask | InProcessAlgorithmTask = AlgorithmTask(
                algorithm_cls,
                params,
                input_layers,
                layer_store=self._layer_store,
                parent=self,
                auto_attach_outputs=auto_attach_outputs,
            )
        else:
            task = InProcessAlgorithmTask(
                algorithm_cls,
                params,
                input_layers,
                layer_store=self._layer_store,
                parent=self,
                auto_attach_outputs=auto_attach_outputs,
                services=self._services,
            )
        task.start()
        return task

    def run_sync(
        self,
        algorithm_cls: type[Algorithm],
        params: dict[str, Any],
        input_layers: dict[str, Layer],
        *,
        services: dict[str, Any] | None = None,
    ) -> AlgorithmResult:
        """Run synchronously in the current process. Used by tests and stubs.

        Skips IPC entirely and does NOT auto-attach output layers (so tests
        can assert directly on the returned result). ``services`` defaults to
        the runner's registered services so tests can simply re-use the
        registered ones, or pass a stub for mocking.
        """

        algorithm = algorithm_cls()
        params_model = algorithm_cls.input_schema.model_validate(params)
        merged_services = dict(self._services)
        if services:
            merged_services.update(services)
        ctx = AlgorithmContext(
            input_layers=dict(input_layers),
            params=params_model,
            services=merged_services,
        )
        try:
            return algorithm.run(ctx)
        except CancellationError as exc:
            return AlgorithmResult.failure(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Algorithm %s failed", algorithm_cls.id)
            return AlgorithmResult.failure(f"{type(exc).__name__}: {exc}")
