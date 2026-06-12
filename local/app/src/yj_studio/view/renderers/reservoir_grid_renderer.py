"""Render ``ReservoirGridLayer`` + ``ReservoirPropertyLayer`` in 3D.

The renderer pulls the live ``ReservoirGrid`` from the
``ReservoirRegistry`` (the layer only holds a string id) and lazily
builds the 2x2x4 downsampled overview the first time the grid is
rendered. The downsample takes a few minutes on the reference model
and is cached on the grid itself, so subsequent renders are instant.

Output: one VTK ``UnstructuredGrid`` of hexahedra, one per active
super-cell. On the reference grid that's ~2.9 M hex cells, which
PyVista handles smoothly with smooth_shading off.

Coloring:
  - If a ``ReservoirPropertyLayer`` referencing this grid is visible,
    its property values drive the cell scalars + colormap.
  - Otherwise the grid is shaded with the layer's flat color.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from yj_studio.reservoir.palettes import palette_for
from yj_studio.reservoir.seismic_mapping import SeismicIndexTransform
from yj_studio.scene.layers import ReservoirGridLayer, ReservoirPropertyLayer
from yj_studio.view.display_coordinates import display_z
from yj_studio.view.highlight import highlight_color, highlight_opacity


# Super-cells with z extent above this many metres are dropped as
# malformed — they're typically pinched at one IJK corner and stretched
# to the bottom of the model, producing the "long spike" artefacts
# along the edges of Petrel exports. The threshold is conservative;
# real storey thickness rarely exceeds a few × downsample_dk × per-cell
# dz (= ~50m on the reference grid), so 250m kills the spikes without
# clipping any reasonable thick package.
_MAX_SUPERCELL_DZ_METRES = 250.0

if TYPE_CHECKING:
    from yj_studio.reservoir import ReservoirGrid, ReservoirRegistry
    from yj_studio.reservoir.downsample import DownsampledGrid

logger = logging.getLogger(__name__)


class ReservoirGridRenderer:
    """Draw reservoir grids + property overlays."""

    def __init__(self, plotter, registry: "ReservoirRegistry") -> None:
        self._plotter = plotter
        self._registry = registry
        self._actor_names: dict[str, str] = {}
        # Cache the assembled mesh per grid_id so re-rendering the
        # same grid with a different property doesn't re-build VTK
        # connectivity. Mesh + color array share the same active-cell
        # ordering, so we keep them in sync via _mesh_cache.
        self._mesh_cache: dict[tuple[str, int | None], pv.UnstructuredGrid] = {}
        self._transform = SeismicIndexTransform()

    def render(
        self,
        grid_layer: ReservoirGridLayer,
        property_layers: list[ReservoirPropertyLayer] | None = None,
        *,
        highlighted: bool = False,
        z_count: int | None = None,
    ) -> None:
        """Render the grid; if property layers are passed, the first visible
        one drives the coloring.
        """

        actor_name = self._actor_name(grid_layer)
        if not grid_layer.visible:
            self.clear(grid_layer.id)
            return

        grid = self._registry.get(grid_layer.grid_id)
        if grid is None:
            logger.warning(
                "ReservoirGridLayer %s references unknown grid_id=%r",
                grid_layer.id, grid_layer.grid_id,
            )
            return

        # Pick the property to color by, if any.
        active_prop: ReservoirPropertyLayer | None = None
        if property_layers:
            for pl in property_layers:
                if pl.visible and pl.grid_layer_id == grid_layer.id:
                    active_prop = pl
                    break

        mesh = self._build_or_get_mesh(grid_layer.grid_id, grid, z_count=z_count)
        if mesh.n_cells == 0:
            logger.warning("Reservoir grid has zero renderable cells")
            return

        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)

        if active_prop is not None:
            self._apply_property(mesh, grid, active_prop)
            # Prefer a project-specific categorical palette (e.g. the
            # LITHOLOGIES one in reservoir.palettes) over the layer's
            # named cmap string. This keeps 3D and 2D section colours
            # in lockstep — switch the palette there, both views update.
            cmap = palette_for(active_prop.property_name) or active_prop.cmap
            self._plotter.add_mesh(
                mesh,
                name=actor_name,
                scalars=active_prop.property_name,
                cmap=cmap,
                clim=active_prop.clim,
                show_edges=grid_layer.show_wireframe,
                edge_color="gray",
                line_width=0.5,
                opacity=highlight_opacity(grid_layer.opacity, highlighted),
                pickable=True,
                scalar_bar_args={"title": active_prop.property_name},
            )
        else:
            # Flat color fallback — no property layer visible.
            color = highlight_color(grid_layer.color, highlighted)
            self._plotter.add_mesh(
                mesh,
                name=actor_name,
                color=color[:3],
                show_edges=grid_layer.show_wireframe,
                edge_color="gray",
                line_width=0.5,
                opacity=highlight_opacity(grid_layer.opacity, highlighted),
                pickable=True,
            )
        self._plotter.render()

    def clear(self, layer_id: str) -> None:
        actor_name = self._actor_names.get(layer_id, f"reservoir-grid-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()

    def invalidate_grid_mesh(self, grid_id: str) -> None:
        """Drop all cached meshes for a grid (e.g. after registry unregister)."""
        for key in list(self._mesh_cache):
            if key[0] == grid_id:
                self._mesh_cache.pop(key, None)

    # ------------------------------------------------------------------ internals

    def _actor_name(self, layer: ReservoirGridLayer) -> str:
        name = f"reservoir-grid-{layer.id}"
        self._actor_names[layer.id] = name
        return name

    def _build_or_get_mesh(
        self, grid_id: str, grid: "ReservoirGrid", *, z_count: int | None
    ) -> pv.UnstructuredGrid:
        # Cache key includes z_count because the same mesh under a
        # different active seismic volume needs a different display-z
        # transform.
        cache_key = (grid_id, z_count)
        cached = self._mesh_cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info("Building reservoir overview mesh for %s (z_count=%s)",
                    grid_id, z_count)
        ds = grid.downsampled()    # 2x2x4 by default; cached on the grid
        mesh = self._assemble_unstructured(ds, z_count=z_count)
        self._mesh_cache[cache_key] = mesh
        return mesh

    def _assemble_unstructured(
        self, ds: "DownsampledGrid", *, z_count: int | None
    ) -> pv.UnstructuredGrid:
        """Turn an active-super-cell soup into a PyVista UnstructuredGrid.

        Each active super-cell becomes one VTK hexahedron. Point
        duplication is deliberate (one independent 8-vertex hex per
        cell) — VTK's hex topology assumes a fixed corner ordering
        and any attempt to share vertices between cells would need to
        match cells that share faces (which corner-point grids only
        do within an IJK column, not across faults / pinches).
        """

        active = ds.active
        if not active.any():
            return pv.UnstructuredGrid()

        # Collect active super-cells' corners in flat order.
        active_flat = active.ravel(order="C")
        all_corners = ds.corners.reshape(-1, 8, 3)
        active_corners = all_corners[active_flat]    # (n_active, 8, 3)

        # Drop malformed super-cells (z extent > threshold). These are
        # the "long spike" artefacts at Petrel model edges where a
        # pinched cell stretches to the bottom of the volume.
        z_per_cell = active_corners[..., 2]    # (n_active, 8)
        z_extent = z_per_cell.max(axis=1) - z_per_cell.min(axis=1)
        valid = z_extent < _MAX_SUPERCELL_DZ_METRES
        dropped = int((~valid).sum())
        if dropped > 0:
            logger.info(
                "Dropped %d malformed super-cells (z extent > %.0f m)",
                dropped, _MAX_SUPERCELL_DZ_METRES,
            )
        active_corners = active_corners[valid]
        n_active = active_corners.shape[0]
        if n_active == 0:
            return pv.UnstructuredGrid()

        # Apply world→sample transform to every point.
        pts = active_corners.reshape(-1, 3)
        pts_sample = self._transform.world_to_sample(pts)
        # Z direction: align with however the seismic scene renders depth.
        # If an active seismic volume is loaded its z_count drives
        # ``display_z`` (sample=0 at the top, deeper sample drawn lower
        # via z_count - sample). When no seismic is loaded we fall back
        # to a plain negation so deeper cells still sit lower.
        if pts_sample.size > 0:
            if z_count is not None:
                pts_sample[:, 2] = display_z(pts_sample[:, 2], z_count)
            else:
                pts_sample[:, 2] = -pts_sample[:, 2]

        # VTK hex corner order is (bottom 4, top 4) with bottom-face
        # going counter-clockwise viewed from above. Our cell_corners
        # already produces (lowK-SW, lowK-SE, lowK-NW, lowK-NE,
        # hiK-SW, hiK-SE, hiK-NW, hiK-NE) — we need to reorder to the
        # VTK ordering (SW, SE, NE, NW, ...) inside each face.
        vtk_corner_order = [0, 1, 3, 2, 4, 5, 7, 6]
        # Build the cell-connectivity array: each cell is `8, p0, p1, ...`
        base = np.arange(n_active, dtype=np.int64) * 8
        conn = np.empty((n_active, 9), dtype=np.int64)
        conn[:, 0] = 8
        for slot_out, slot_in in enumerate(vtk_corner_order):
            conn[:, 1 + slot_out] = base + slot_in
        conn = conn.ravel()

        from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonDataModel import VTK_HEXAHEDRON

        cell_types = np.full(n_active, VTK_HEXAHEDRON, dtype=np.uint8)
        grid_pv = pv.UnstructuredGrid(conn, cell_types, pts_sample)

        # Record source super-cell IJK so coloring can index back.
        # ``valid`` was applied to active_corners above; apply the same
        # mask to the IJK index list so the two stay in sync.
        ijk_idx = np.argwhere(active)[valid]    # (n_active, 3) in (I, J, K)
        grid_pv.cell_data["__super_I"] = ijk_idx[:, 0].astype(np.int32)
        grid_pv.cell_data["__super_J"] = ijk_idx[:, 1].astype(np.int32)
        grid_pv.cell_data["__super_K"] = ijk_idx[:, 2].astype(np.int32)
        return grid_pv

    def _apply_property(
        self,
        mesh: pv.UnstructuredGrid,
        grid: "ReservoirGrid",
        prop_layer: ReservoirPropertyLayer,
    ) -> None:
        """Attach the property layer's values as cell scalars on the mesh."""

        ds = grid.downsampled()
        name = prop_layer.property_name
        if prop_layer.is_integer:
            arr = ds.int_properties.get(name)
        else:
            arr = ds.float_properties.get(name)
        if arr is None:
            logger.warning(
                "Reservoir property %r not found on downsampled grid", name
            )
            return

        I = mesh.cell_data["__super_I"]
        J = mesh.cell_data["__super_J"]
        K = mesh.cell_data["__super_K"]
        values = arr[I, J, K]
        mesh.cell_data[name] = values
