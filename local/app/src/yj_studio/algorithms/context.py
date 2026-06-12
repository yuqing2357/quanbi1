from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from yj_studio.scene.layer import Layer

from .protocol import CancellationError

ProgressCallback = Callable[[float, str], None]
CancelChecker = Callable[[], bool]


@dataclass(slots=True)
class AlgorithmContext:
    """Runtime context handed to an ``Algorithm.run``.

    Lives entirely inside whichever process the algorithm runs in. The worker
    constructs one from the IPC ``run`` message; in-process synchronous calls
    can build one directly. ``progress_callback`` and ``cancel_checker`` are
    closures the worker provides — they bridge to the IPC queue without the
    algorithm having to know about IPC.

    ``services`` is a free-form bag of in-process resources that only
    ``runs_in_subprocess=False`` algorithms (e.g. SAM3) can rely on. Cross-
    process algorithms must keep ``services`` empty — pickling a GPU model
    into a subprocess does not work and would defeat the point of lazy
    loading it in the main process.
    """

    input_layers: dict[str, Layer] = field(default_factory=dict)
    params: Any = None
    progress_callback: ProgressCallback | None = None
    cancel_checker: CancelChecker | None = None
    services: dict[str, Any] = field(default_factory=dict)

    def report_progress(self, fraction: float, message: str = "") -> None:
        if self.progress_callback is not None:
            self.progress_callback(float(fraction), message)
        # Cancel is checked at the same heartbeat as progress reports to keep
        # algorithm code simple: just call report_progress() in the inner loop.
        self.check_cancel()

    def check_cancel(self) -> None:
        if self.cancel_checker is not None and self.cancel_checker():
            raise CancellationError("Algorithm cancelled by user")
