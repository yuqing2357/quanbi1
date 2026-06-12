from __future__ import annotations

from collections.abc import Iterable

from .algorithm import Algorithm


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._algorithms: dict[str, type[Algorithm]] = {}

    def register(self, algorithm_cls: type[Algorithm]) -> type[Algorithm]:
        algorithm_id = algorithm_cls.id
        if algorithm_id in self._algorithms:
            raise ValueError(f"Algorithm already registered: {algorithm_id}")
        self._algorithms[algorithm_id] = algorithm_cls
        return algorithm_cls

    def get(self, algorithm_id: str) -> type[Algorithm]:
        return self._algorithms[algorithm_id]

    def iter_algorithms(self) -> Iterable[type[Algorithm]]:
        return self._algorithms.values()


registry = AlgorithmRegistry()


def register_algorithm(algorithm_cls: type[Algorithm]) -> type[Algorithm]:
    return registry.register(algorithm_cls)

