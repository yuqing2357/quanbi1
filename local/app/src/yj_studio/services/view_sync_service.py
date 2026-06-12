from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

SyncCallback = Callable[[str, Any, object | None], None]


class ViewSyncService(QObject):
    topic_published = pyqtSignal(str, object, object)

    def __init__(self) -> None:
        super().__init__()
        self._subscribers: dict[str, list[SyncCallback]] = defaultdict(list)

    def subscribe(self, topic: str, callback: SyncCallback) -> None:
        self._subscribers[topic].append(callback)

    def publish(self, topic: str, value: Any, origin: object | None = None) -> None:
        self.topic_published.emit(topic, value, origin)
        for callback in list(self._subscribers.get(topic, [])):
            callback(topic, value, origin)

