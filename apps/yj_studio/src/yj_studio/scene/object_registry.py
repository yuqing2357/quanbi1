from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ObjectRegistry:
    relationships: dict[str, set[str]] = field(default_factory=dict)

    def link(self, parent_id: str, child_id: str) -> None:
        self.relationships.setdefault(parent_id, set()).add(child_id)

    def children_of(self, parent_id: str) -> tuple[str, ...]:
        return tuple(sorted(self.relationships.get(parent_id, set())))

