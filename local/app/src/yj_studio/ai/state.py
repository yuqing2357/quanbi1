from __future__ import annotations

from enum import Enum


class AIServiceState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
