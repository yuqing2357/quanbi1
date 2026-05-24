"""IPC message types between the main UI process and an algorithm worker.

The protocol is intentionally plain ``dict`` so it survives
``multiprocessing.Queue`` (which uses pickle internally) and, in the future,
``subprocess`` JSON streaming for the SAM3 worker that lives in a different
Python environment.

Messages are tagged with a ``kind`` field; payload schemas live below.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class RunMessage(TypedDict):
    kind: Literal["run"]
    algorithm: str  # "module.path:ClassName"
    params: dict[str, Any]
    input_layers: dict[str, dict[str, Any]]  # role -> layer payload


class CancelMessage(TypedDict):
    kind: Literal["cancel"]


class ProgressMessage(TypedDict):
    kind: Literal["progress"]
    fraction: float
    message: str


class LogMessage(TypedDict):
    kind: Literal["log"]
    level: str
    message: str


class DoneMessage(TypedDict):
    kind: Literal["done"]
    ok: bool
    output_layers: list[dict[str, Any]]
    summary: str


class ErrorMessage(TypedDict):
    kind: Literal["error"]
    message: str
    traceback: str


class CancelledMessage(TypedDict):
    kind: Literal["cancelled"]


# Public alias for any incoming/outgoing message
ParentToChild = RunMessage | CancelMessage
ChildToParent = ProgressMessage | LogMessage | DoneMessage | ErrorMessage | CancelledMessage


class CancellationError(Exception):
    """Raised by ``AlgorithmContext.check_cancel`` when the user pressed cancel."""
