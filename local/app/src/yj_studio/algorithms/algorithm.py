from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from .context import AlgorithmContext
from .result import AlgorithmResult


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    pass


class Algorithm(ABC):
    id: ClassVar[str]
    category: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str] = ""
    input_schema: ClassVar[type[BaseModel]] = EmptyInput
    output_schema: ClassVar[type[BaseModel]] = EmptyOutput
    # Input role names that point to a Layer (handled separately from params).
    layer_inputs: ClassVar[dict[str, str]] = {}
    runs_in_subprocess: ClassVar[bool] = True
    supports_cancel: ClassVar[bool] = True

    @classmethod
    def import_path(cls) -> str:
        """Return a ``"module.path:ClassName"`` reference used by the worker."""

        return f"{cls.__module__}:{cls.__qualname__}"

    @abstractmethod
    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        raise NotImplementedError
