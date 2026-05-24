from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class InteractionTool:
    id: str
    label: str
    icon: str = ""
    cursor: str = "arrow"
    enabled: bool = True

    def activate(self, view: Any) -> None:
        return None

    def deactivate(self, view: Any) -> None:
        return None

    def on_mouse_press(self, view: Any, event: Any) -> bool:
        return False

    def on_mouse_move(self, view: Any, event: Any) -> bool:
        return False

    def on_mouse_release(self, view: Any, event: Any) -> bool:
        return False

    def on_mouse_double_click(self, view: Any, event: Any) -> bool:
        return False

    def on_key_press(self, view: Any, event: Any) -> bool:
        return False

    def on_key_release(self, view: Any, event: Any) -> bool:
        return False

    def on_pick_result(
        self,
        world_xyz: tuple[float, float, float] | None,
        picked_layer_id: str | None,
        picked_cell_id: int | None,
    ) -> None:
        return None
