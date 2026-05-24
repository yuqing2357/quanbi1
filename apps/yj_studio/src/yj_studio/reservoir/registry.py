"""In-process registry of loaded ``ReservoirGrid`` objects.

ReservoirGrid holds memmaps and LRU caches — it can't be pickled or
embedded into the (slotted, serialisable) Layer types. Instead the
Layer carries a stable string ``grid_id``, and renderers/algorithms
look the live grid up here.

The registry is meant to be registered into the app's tool/service
container at startup; tests can instantiate one directly.

Loading a grid is *expensive* (the first-time ZCORN cache build alone
is several minutes). Callers should keep the same ``grid_id`` across
sessions when possible — we derive a default from the source file
path so re-loading the same file twice returns the same id.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator

from .grid import ReservoirGrid

logger = logging.getLogger(__name__)


class ReservoirRegistry:
    """Maps ``grid_id`` strings → ``ReservoirGrid`` instances."""

    def __init__(self) -> None:
        self._grids: dict[str, ReservoirGrid] = {}

    # ------------------------------------------------------------------ ids

    @staticmethod
    def id_for_path(master_path: Path) -> str:
        """Stable id derived from the master GRDECL path.

        We use the absolute path's sha1 prefix so paths with non-ASCII
        characters (Chinese filenames) stay safe inside id strings,
        layer JSON, etc.
        """
        p = str(Path(master_path).resolve())
        digest = hashlib.sha1(p.encode("utf-8")).hexdigest()
        return f"reservoir-{digest[:12]}"

    # ------------------------------------------------------------------ CRUD

    def register(self, grid: ReservoirGrid, grid_id: str | None = None) -> str:
        gid = grid_id or self.id_for_path(grid.master_path)
        self._grids[gid] = grid
        logger.info("Registered reservoir grid %s from %s", gid, grid.master_path)
        return gid

    def get(self, grid_id: str) -> ReservoirGrid | None:
        return self._grids.get(grid_id)

    def require(self, grid_id: str) -> ReservoirGrid:
        grid = self._grids.get(grid_id)
        if grid is None:
            raise KeyError(f"No reservoir grid with id={grid_id!r}")
        return grid

    def unregister(self, grid_id: str) -> ReservoirGrid | None:
        grid = self._grids.pop(grid_id, None)
        if grid is not None:
            grid.clear_geometry_cache()
            logger.info("Unregistered reservoir grid %s", grid_id)
        return grid

    def ids(self) -> Iterator[str]:
        return iter(self._grids.keys())

    def __contains__(self, grid_id: str) -> bool:
        return grid_id in self._grids

    def __len__(self) -> int:
        return len(self._grids)
