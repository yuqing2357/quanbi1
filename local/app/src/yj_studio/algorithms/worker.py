"""Subprocess entry point and helpers for running algorithms out-of-process.

The main UI process drops a ``run`` message in ``inbox``; this worker decodes
it, instantiates the algorithm by ``module:Class`` path, ships progress and
log messages back through ``outbox``, and finally writes a ``done`` /
``error`` / ``cancelled`` message.

Designed to be Windows-safe: there is no module-level state and ``run_worker``
is called by ``multiprocessing.Process`` after a ``spawn`` import, so the
algorithm classes must be importable from the worker's ``sys.path``.
"""

from __future__ import annotations

import importlib
import queue
import traceback
from typing import Any

from .context import AlgorithmContext
from .protocol import CancellationError
from .result import AlgorithmResult
from .serialization import layer_to_payload, payloads_to_layers


def _resolve_algorithm(import_path: str):
    module_name, _, class_name = import_path.partition(":")
    if not class_name:
        raise ValueError(f"Bad algorithm import path: {import_path!r}")
    module = importlib.import_module(module_name)
    cls = module
    for part in class_name.split("."):
        cls = getattr(cls, part)
    return cls


def _drain_cancel_flag(inbox) -> bool:
    """Non-blocking check whether the parent has sent a ``cancel`` message."""

    try:
        while True:
            msg = inbox.get_nowait()
            if isinstance(msg, dict) and msg.get("kind") == "cancel":
                return True
    except queue.Empty:
        return False


def run_worker(inbox, outbox) -> None:
    """Worker entry point. Lives until the run finishes or errors out.

    Both queues are ``multiprocessing.Queue`` instances passed by the parent.
    """

    cancelled = [False]

    try:
        message = inbox.get()  # blocks until parent sends `run`
    except (EOFError, OSError) as exc:
        outbox.put({"kind": "error", "message": f"IPC closed: {exc}", "traceback": ""})
        return

    if not isinstance(message, dict) or message.get("kind") != "run":
        outbox.put(
            {
                "kind": "error",
                "message": f"Expected run message, got {message!r}",
                "traceback": "",
            }
        )
        return

    try:
        algorithm_cls = _resolve_algorithm(message["algorithm"])
        params_model = algorithm_cls.input_schema
        params = params_model.model_validate(message.get("params", {}))
        layers = payloads_to_layers(message.get("input_layers", {}))

        def report_progress(fraction: float, msg: str) -> None:
            outbox.put(
                {"kind": "progress", "fraction": float(fraction), "message": str(msg)}
            )

        def cancel_checker() -> bool:
            if not cancelled[0] and _drain_cancel_flag(inbox):
                cancelled[0] = True
            return cancelled[0]

        ctx = AlgorithmContext(
            input_layers=layers,
            params=params,
            progress_callback=report_progress,
            cancel_checker=cancel_checker,
        )

        algorithm = algorithm_cls()
        result: AlgorithmResult = algorithm.run(ctx)
        outbox.put(
            {
                "kind": "done",
                "ok": bool(result.ok),
                "output_layers": [layer_to_payload(layer) for layer in result.output_layers],
                "summary": result.summary,
            }
        )
    except CancellationError:
        outbox.put({"kind": "cancelled"})
    except Exception as exc:  # noqa: BLE001 — worker boundary, convert to message
        outbox.put(
            {
                "kind": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _ensure_dict_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Layer payload must be dict, got {type(payload)!r}")
    return payload
