from __future__ import annotations

from .tool import InteractionTool


class NavigationTool(InteractionTool):
    def __init__(self) -> None:
        super().__init__(id="navigation", label="Navigation", icon="navigation", cursor="arrow")

