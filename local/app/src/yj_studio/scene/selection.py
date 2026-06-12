from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Selection:
    layer_ids: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.layer_ids.clear()

