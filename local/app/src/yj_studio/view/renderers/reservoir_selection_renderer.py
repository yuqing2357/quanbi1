"""Render ``ReservoirSelectionLayer`` as a highlighted subset of cells.

Selections are the output of SAM3 mask reverse-lookup or video
propagation: a set of ``(i, j, k)`` triples on a known reservoir
grid. We piggy-back on the existing downsampled overview mesh from
``ReservoirGridRenderer`` — the same ``__super_I/J/K`` cell_data it
attaches lets us pick a sub-mesh of super-cells covering the chosen
cells. That keeps the 3D footprint identical to the rest of the
overview (no shape mismatch) and avoids re-rendering several million
hexes per selection.

Trade-off: a super-cell is 2x2x4 = 16 underlying cells, so the
highlight is coarser than the actual selection. The pixel-level
truth lives on the 2D workbench; this view answers "where is the
AI-extracted body in 3D".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from yj_studio.scene.layers import ReservoirSelectionLayer
from yj_studio.view.highlight import highlight_color, highlight_opacity

if TYPE_CHECKING:
    from yj_studio.reservoir import ReservoirGrid, ReservoirRegistry
    from yj_studio.view.renderers.reservoir_grid_renderer import (
        ReservoirGridRenderer,
    )

logger = logging.getLogger(__name__)


class ReservoirSelectionRenderer:
    """Render selection layers by extracting matching super-cells from
    the corresponding grid's overview mesh.

    Requires references to the registry (to find the live grid) and
    the grid renderer (to access its cached overview mesh + the
    ``__super_I/J/K`` cell_data).
    """

    def __init__(
        self,
        plotter,
        registry: "ReservoirRegistry",
        grid_renderer: "ReservoirGridRenderer",
    ) -> None:
        self._plotter = plotter
        self._registry = registry
        self._grid_renderer = grid_renderer
        self._actor_names: dict[str, str] = {}

    def render(
        self,
        layer: ReservoirSelectionLayer,
        *,
        highlighted: bool = False,
        z_count: int | None = None,
    ) -> None:
        actor_name = self._actor_name(layer)
        if not layer.visible or layer.n_cells == 0:
            self.clear(layer.id)
            return

        grid = self._registry.get(layer.grid_id)
        if grid is None:
            logger.warning(
                "Selection layer %s references unknown grid %s; skipping",
                layer.id, layer.grid_id,
            )
            self.clear(layer.id)
            return

        overview_mesh = self._grid_renderer._build_or_get_mesh(    # noqa: SLF001
            layer.grid_id, grid, z_count=z_count,
        )
        if overview_mesh.n_cells == 0:
            self.clear(layer.id)
            return

        sub_mesh = self._extract_super_cells(overview_mesh, layer.cell_ids, grid)
        if sub_mesh.n_cells == 0:
            logger.info(
                "Selection layer %s: no overlap with overview mesh", layer.id
            )
            self.clear(layer.id)
            return

        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        # Selection renders as the actual super-cell hex sub-mesh, so
        # the body's shape and extent in 3D are honest. This relies on
        # the reservoir grid layer being translucent (default 0.45) so
        # the selection inside the model stays visible; if a user
        # raises the grid opacity back to 1.0 the selection will be
        # occluded — that's a deliberate trade-off, not a bug.
        self._plotter.add_mesh(
            sub_mesh,
            name=actor_name,
            color=highlight_color(layer.color, highlighted)[:3],
            opacity=highlight_opacity(layer.opacity, highlighted),
            show_edges=True,
            edge_color=(0.05, 0.05, 0.05),
            line_width=0.4,
            pickable=False,
        )
        self._plotter.render()

    def clear(self, layer_id: str) -> None:
        actor_name = self._actor_names.get(layer_id, f"reservoir-selection-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()

    # ------------------------------------------------------------------ helpers

    def _actor_name(self, layer: ReservoirSelectionLayer) -> str:
        actor_name = f"reservoir-selection-{layer.id}"
        self._actor_names[layer.id] = actor_name
        return actor_name

    def _extract_super_cells(
        self,
        overview_mesh: pv.UnstructuredGrid,
        cell_ids: np.ndarray,
        grid: "ReservoirGrid",
    ) -> pv.UnstructuredGrid:
        """Map cell IJK → super-cell IJK → indices into ``overview_mesh``.

        The downsample block (typically 2x2x4) divides cell IJK into
        super-cell IJK by floor division; super-cells are stored on
        the mesh's cell_data as ``__super_I/J/K``. We build a set of
        target super-cell triples then mask the mesh cells whose triple
        is in the set.
        """
        ds = grid.downsampled()
        block = ds.block    # e.g. (2, 2, 4)
        super_ijk_targets = (
            cell_ids // np.asarray(block, dtype=np.int32)
        )
        # Dedup to keep the set lookup small.
        super_ijk_targets = np.unique(super_ijk_targets, axis=0)

        # Encode each super-cell triple as a single int for fast lookup.
        Nx, Ny, _Nz = ds.shape
        flat_targets = (
            super_ijk_targets[:, 0].astype(np.int64) * (Ny * 10000)
            + super_ijk_targets[:, 1].astype(np.int64) * 10000
            + super_ijk_targets[:, 2].astype(np.int64)
        )
        target_set = set(flat_targets.tolist())

        I = np.asarray(overview_mesh.cell_data["__super_I"])
        J = np.asarray(overview_mesh.cell_data["__super_J"])
        K = np.asarray(overview_mesh.cell_data["__super_K"])
        flat_mesh = (
            I.astype(np.int64) * (Ny * 10000)
            + J.astype(np.int64) * 10000
            + K.astype(np.int64)
        )
        mask = np.isin(flat_mesh, list(target_set))
        cell_indices = np.where(mask)[0]
        if cell_indices.size == 0:
            return pv.UnstructuredGrid()
        return overview_mesh.extract_cells(cell_indices)
