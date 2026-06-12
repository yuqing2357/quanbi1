from __future__ import annotations

from dataclasses import dataclass, field

from yj_studio.scene.layer import Layer


@dataclass(slots=True)
class AlgorithmResult:
    ok: bool
    output_layers: list[Layer] = field(default_factory=list)
    summary: str = ""
    error: str | None = None

    @classmethod
    def success(cls, output_layers: list[Layer] | None = None, summary: str = "") -> "AlgorithmResult":
        return cls(ok=True, output_layers=output_layers or [], summary=summary)

    @classmethod
    def failure(cls, error: str) -> "AlgorithmResult":
        return cls(ok=False, error=error)

