"""ReservoirGrid: corner-point grid loaded from a Petrel GRDECL export.

The grid wraps four parts:

- ``spec``: SPECGRID nx/ny/nz (so we know the index space).
- ``active``: ACTNUM ``(nx, ny, nz) int8`` array — small enough (140
  MB on our 372x343x1076 reference) to keep fully in RAM.
- ``coord``: pillar array ``(nx+1, ny+1, 6) float32`` — 3 MB, trivial.
- ``zcorn``: corner-z memmap ``(2nx, 2ny, 2nz) float32`` — sized in GB
  but never read whole; pages in on demand.

Plus a ``properties`` dict for LITHOLOGIES / PORO / ... — these are
each (nx, ny, nz), typically a few hundred MB.

Cell geometry (cell_corners) is computed on demand for K chunks and
LRU-cached. The default chunk size is 32 K-layers, which is ~94 MB
per chunk on the reference grid; we keep 4 chunks resident.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from yj_studio.io.grdecl import find_includes, summarize_grdecl
from yj_studio.io.grdecl.parser import (
    read_actnum,
    read_coord,
    read_float_property,
    read_int_property,
)
from yj_studio.io.grdecl.spec import SpecGrid
from yj_studio.io.grdecl.zcorn_cache import (
    build_cache,
    cell_corners,
    open_zcorn,
    zcorn_for_k_range,
)

logger = logging.getLogger(__name__)

DEFAULT_K_CHUNK = 32
DEFAULT_CACHED_CHUNKS = 4


# Property keywords we look for. Order matters only for the dock UI:
# the first one present becomes the default colored property.
_INT_PROPERTY_KEYWORDS = ("LITHOLOGIES", "FACIES", "FLUXNUM", "SATNUM", "PVTNUM")
_FLOAT_PROPERTY_KEYWORDS = ("PORO", "PERMX", "PERMY", "PERMZ", "NTG", "SO", "SW", "SG")


@dataclass(slots=True)
class _ChunkCache:
    """Tiny LRU keyed by (k0, k1) → cell_corners array."""

    capacity: int
    _store: "OrderedDict[tuple[int, int], np.ndarray]" = field(default_factory=OrderedDict)

    def get(self, key: tuple[int, int]) -> np.ndarray | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: tuple[int, int], value: np.ndarray) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


@dataclass(slots=True)
class ReservoirGrid:
    """Loaded corner-point grid + cached chunk geometry."""

    spec: SpecGrid
    master_path: Path
    coord: np.ndarray          # (nx+1, ny+1, 6) float32
    active: np.ndarray         # (nx, ny, nz) int8
    zcorn: np.memmap           # (2nx, 2ny, 2nz) float32
    properties: dict[str, np.ndarray] = field(default_factory=dict)
    k_chunk: int = DEFAULT_K_CHUNK

    _chunks: _ChunkCache = field(default_factory=lambda: _ChunkCache(DEFAULT_CACHED_CHUNKS))
    # Lazy-built 3D overview. Stored here (not on the layer) so a
    # single grid shared by multiple ReservoirGridLayers doesn't
    # re-downsample. Built on first access via ``downsampled()``.
    _downsampled: object | None = None
    # Cached full-grid xy bounds (computed from COORD on first call).
    _xy_bounds: tuple[float, float, float, float] | None = None
    # Cached per-axis active-only bbox (k / i / j → bbox).
    _active_axis_bounds: dict[str, tuple[float, float, float, float]] | None = None
    # Cached active-cell sample range (depth in sample-index units).
    _active_sample_range: tuple[float, float] | None = None
    # Cache for roi.roi_z_bounds(): {ROI tuple: (z_min, z_max)}.
    # Same ROI is reused across many frames during SAM3 propagation;
    # the lookup walks ZCORN slabs which is ~1 GB of paged-in data,
    # not free to redo.
    _roi_z_cache: dict[tuple[int, int, int, int, int, int], tuple[float, float]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------ loaders

    @classmethod
    def load_from_master(
        cls,
        master_path: Path,
        properties: list[str] | None = None,
        progress_cb=None,
    ) -> "ReservoirGrid":
        """Load a complete reservoir grid from a master GRDECL file.

        Looks up INCLUDEs to find COORD / ZCORN / ACTNUM. Properties
        are loaded from the master file directly (Petrel writes
        LITHOLOGIES and PORO inline). ``properties`` filters which
        property keywords to actually load — defaults to all known
        keywords that appear in the master's top-level inventory.
        """

        from .cache import cache_path_for, load_grid_cache

        master_path = Path(master_path).resolve()
        if not master_path.exists():
            raise FileNotFoundError(master_path)

        summary = summarize_grdecl(master_path)
        if summary.specgrid is None:
            raise ValueError(f"No SPECGRID in {master_path}")
        spec = summary.specgrid
        includes = summary.includes
        if progress_cb:
            progress_cb(0.05, f"SPECGRID {spec.nx}x{spec.ny}x{spec.nz}")

        coord_path = _pick_include(includes, "_COORD")
        zcorn_path = _pick_include(includes, "_ZCORN")
        actnum_path = _pick_include(includes, "_ACTNUM")
        for label, path in [("COORD", coord_path), ("ZCORN", zcorn_path), ("ACTNUM", actnum_path)]:
            if path is None:
                raise FileNotFoundError(
                    f"{label} INCLUDE not found in {master_path}"
                )

        # Try the .npz fast path first. The cache key embeds the GRDECL
        # mtime so editing the source file naturally misses cache.
        spec_fp = f"{spec.nx}x{spec.ny}x{spec.nz}"
        grid_cache_path = cache_path_for(master_path, spec_fp)
        cached = None
        if grid_cache_path.exists():
            try:
                if progress_cb:
                    progress_cb(0.10, "Loading cached grid (fast path)")
                cached = load_grid_cache(grid_cache_path)
            except Exception:
                logger.exception("Failed to load grid cache %s; rebuilding",
                                 grid_cache_path)
                cached = None

        if cached is not None:
            coord, active, loaded_props, ds_payload = cached
            if progress_cb:
                progress_cb(0.80, "Opening ZCORN memmap")
            cache_path_zcorn = build_cache(zcorn_path, spec)
            zcorn = open_zcorn(cache_path_zcorn, spec)
            if progress_cb:
                progress_cb(1.0, "ReservoirGrid loaded (cache)")
            instance = cls(
                spec=spec,
                master_path=master_path,
                coord=coord,
                active=active,
                zcorn=zcorn,
                properties=loaded_props,
            )
            if ds_payload is not None:
                from .downsample import DownsampledGrid
                instance._downsampled = DownsampledGrid(**ds_payload)
            return instance

        # ----- slow path: parse ASCII from scratch
        if progress_cb:
            progress_cb(0.10, "Reading ACTNUM")
        active = read_actnum(actnum_path, spec)

        if progress_cb:
            progress_cb(0.20, "Reading COORD")
        coord = read_coord(coord_path, spec)

        # ZCORN: build (or reuse) the binary cache, then memmap. The
        # build is the slow step on first run — surface progress.
        if progress_cb:
            progress_cb(0.25, "Building ZCORN cache (first run is slow)")

        def _zcorn_cb(written: int, total: int) -> None:
            if progress_cb is None or total <= 0:
                return
            # Map ZCORN's [0, 1] into our [0.25, 0.80] band.
            progress_cb(0.25 + 0.55 * (written / total), "Building ZCORN cache")

        cache_path = build_cache(zcorn_path, spec, progress_cb=_zcorn_cb)
        zcorn = open_zcorn(cache_path, spec)

        # Properties
        if progress_cb:
            progress_cb(0.85, "Reading properties")
        keywords_seen = set(summary.keywords_seen)
        loaded: dict[str, np.ndarray] = {}
        want = properties
        for kw in _INT_PROPERTY_KEYWORDS:
            if kw not in keywords_seen:
                continue
            if want is not None and kw not in want:
                continue
            try:
                loaded[kw] = read_int_property(master_path, kw, spec)
            except Exception:
                logger.exception("Failed to read int property %s", kw)
        for kw in _FLOAT_PROPERTY_KEYWORDS:
            if kw not in keywords_seen:
                continue
            if want is not None and kw not in want:
                continue
            try:
                loaded[kw] = read_float_property(master_path, kw, spec)
            except Exception:
                logger.exception("Failed to read float property %s", kw)

        if progress_cb:
            progress_cb(1.0, "ReservoirGrid loaded")

        return cls(
            spec=spec,
            master_path=master_path,
            coord=coord,
            active=active,
            zcorn=zcorn,
            properties=loaded,
        )

    # ------------------------------------------------------------------ accessors

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.spec.nx, self.spec.ny, self.spec.nz)

    def property(self, name: str) -> np.ndarray:
        return self.properties[name]

    def has_property(self, name: str) -> bool:
        return name in self.properties

    def property_names(self) -> list[str]:
        return list(self.properties.keys())

    # ------------------------------------------------------------------ geometry

    def corners_for_k_chunk(
        self, k0: int, k1: int
    ) -> np.ndarray:
        """Return ``(nx, ny, k1-k0, 8, 3) float32`` corner array for a K range.

        The result is LRU-cached on (k0, k1) so repeated calls (e.g.
        when the user scrubs the I or J slice within the same K
        window) don't re-touch ZCORN.
        """

        if not (0 <= k0 < k1 <= self.spec.nz):
            raise ValueError(f"bad k range [{k0}, {k1}) for nz={self.spec.nz}")

        key = (k0, k1)
        cached = self._chunks.get(key)
        if cached is not None:
            return cached

        slab = zcorn_for_k_range(self.zcorn, self.spec, k0, k1)
        corners = cell_corners(slab, self.coord, self.spec, k_offset=k0)
        self._chunks.put(key, corners)
        return corners

    def chunk_for_k(self, k: int) -> tuple[int, int]:
        """Return the (k0, k1) chunk window that contains layer ``k``.

        Chunks are aligned to ``self.k_chunk`` so iterating K
        sequentially reuses the cache.
        """

        if not 0 <= k < self.spec.nz:
            raise IndexError(f"k={k} out of range for nz={self.spec.nz}")
        k0 = (k // self.k_chunk) * self.k_chunk
        k1 = min(k0 + self.k_chunk, self.spec.nz)
        return k0, k1

    def corners_for_k_layer(self, k: int) -> np.ndarray:
        """Return ``(nx, ny, 8, 3)`` corners for a single K layer."""

        k0, k1 = self.chunk_for_k(k)
        chunk = self.corners_for_k_chunk(k0, k1)
        return chunk[:, :, k - k0]

    def clear_geometry_cache(self) -> None:
        self._chunks.clear()
        self._downsampled = None
        self._xy_bounds = None
        self._active_axis_bounds = None
        self._active_sample_range = None

    def local_xy_bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) of all pillar xy in local frame.

        Computed from COORD (top + bottom pillar endpoints), so the
        envelope covers everywhere a cell could occupy.
        """

        if self._xy_bounds is not None:
            return self._xy_bounds
        coord = self.coord
        xs = np.concatenate([coord[..., 0].ravel(), coord[..., 3].ravel()])
        ys = np.concatenate([coord[..., 1].ravel(), coord[..., 4].ravel()])
        self._xy_bounds = (float(xs.min()), float(xs.max()),
                           float(ys.min()), float(ys.max()))
        return self._xy_bounds

    def active_xy_bounds_per_axis(
        self,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Per-section-axis tight bbox over pillars touched by active cells.

        Returns a dict ``{"k": (x_min, x_max, y_min, y_max),
        "i": (y_min, y_max, _, _), "j": (x_min, x_max, _, _)}``.

        - K bbox encloses every (i, j) with any active cell over all K,
          so the same window works for any chosen K layer.
        - I bbox horizontal = local-y across the j range that has any
          active cell anywhere (so any chosen i lands in the same
          window).
        - J bbox horizontal = local-x across the i range that has any
          active cell.

        Used by the 2D section view to lock its axes — SAM3 needs the
        pixel grid to mean the same thing across index choices on a
        given axis. Different *axes* may have different windows, which
        is intentional: a prompt box on a K layer can't be reused on
        an I section anyway.
        """

        if self._active_axis_bounds is not None:
            return self._active_axis_bounds

        nx, ny, nz = self.spec.nx, self.spec.ny, self.spec.nz
        active = self.active != 0
        has_active_ij = active.any(axis=2)       # (nx, ny) — over all K
        has_active_i = active.any(axis=(1, 2))   # (nx,)   — over all (j, k)
        has_active_j = active.any(axis=(0, 2))   # (ny,)   — over all (i, k)

        # K bbox: pillars touched by any active cell. Each active
        # (i, j) marks pillars (i, j), (i+1, j), (i, j+1), (i+1, j+1).
        pillar_k = np.zeros((nx + 1, ny + 1), dtype=bool)
        pillar_k[:-1, :-1] |= has_active_ij
        pillar_k[1:,  :-1] |= has_active_ij
        pillar_k[:-1, 1:]  |= has_active_ij
        pillar_k[1:,  1:]  |= has_active_ij
        xs_k = np.concatenate([self.coord[..., 0][pillar_k],
                               self.coord[..., 3][pillar_k]])
        ys_k = np.concatenate([self.coord[..., 1][pillar_k],
                               self.coord[..., 4][pillar_k]])

        # I bbox: pillars whose j is active anywhere. Take all i (we
        # don't constrain i — every i can be chosen as the section).
        j_mask = np.zeros(ny + 1, dtype=bool)
        j_mask[:-1] |= has_active_j
        j_mask[1:]  |= has_active_j
        ys_i = np.concatenate([self.coord[:, j_mask, 1].ravel(),
                               self.coord[:, j_mask, 4].ravel()])

        # J bbox: pillars whose i is active anywhere.
        i_mask = np.zeros(nx + 1, dtype=bool)
        i_mask[:-1] |= has_active_i
        i_mask[1:]  |= has_active_i
        xs_j = np.concatenate([self.coord[i_mask, :, 0].ravel(),
                               self.coord[i_mask, :, 3].ravel()])

        self._active_axis_bounds = {
            "k": (float(xs_k.min()), float(xs_k.max()),
                  float(ys_k.min()), float(ys_k.max())),
            "i": (float(ys_i.min()), float(ys_i.max()), 0.0, 0.0),
            "j": (float(xs_j.min()), float(xs_j.max()), 0.0, 0.0),
        }
        return self._active_axis_bounds

    def sample_range(self, z_step: float) -> tuple[float, float]:
        """Return (sample_min, sample_max) for the grid's full depth span.

        Uses every pillar endpoint, so this is the maximum possible
        sample window. Most active cells occupy only a fraction of
        this range — see :meth:`active_sample_range` for the tight
        version.
        """

        coord = self.coord
        z_min = float(min(coord[..., 2].min(), coord[..., 5].min()))
        z_max = float(max(coord[..., 2].max(), coord[..., 5].max()))
        return z_min / z_step, z_max / z_step

    def active_sample_range(self, z_step: float) -> tuple[float, float]:
        """Return (sample_min, sample_max) restricted to active cells.

        We approximate cell depths from the per-cell z_center (computed
        on the fly from ZCORN) so this is a tight envelope of where
        active data actually lives. Cached because computing it touches
        every K-chunk.
        """

        if self._active_sample_range is not None:
            return self._active_sample_range

        active = self.active != 0
        # Walk K-chunks; record min/max z among active cells in each chunk.
        z_min = float("inf")
        z_max = float("-inf")
        for k0 in range(0, self.spec.nz, self.k_chunk):
            k1 = min(k0 + self.k_chunk, self.spec.nz)
            corners = self.corners_for_k_chunk(k0, k1)    # (nx, ny, nk, 8, 3)
            mask = active[:, :, k0:k1]
            if not mask.any():
                continue
            cell_zs = corners[..., 2]    # (nx, ny, nk, 8)
            # Reduce per-cell z to min/max then mask.
            cell_min = cell_zs.min(axis=-1)
            cell_max = cell_zs.max(axis=-1)
            active_min = cell_min[mask]
            active_max = cell_max[mask]
            if active_min.size:
                z_min = min(z_min, float(active_min.min()))
                z_max = max(z_max, float(active_max.max()))

        if not np.isfinite(z_min):    # no active cells at all
            return self.sample_range(z_step)
        self._active_sample_range = (z_min / z_step, z_max / z_step)
        return self._active_sample_range

    def downsampled(
        self,
        block: tuple[int, int, int] = (2, 2, 4),
        progress_cb=None,
    ) -> "object":
        """Return (and lazily build) the downsampled overview.

        Returns a ``DownsampledGrid`` — see ``reservoir.downsample``.
        Type-hinted as ``object`` here to avoid an import cycle between
        ``grid`` and ``downsample``.
        """

        if self._downsampled is not None:
            return self._downsampled
        from .downsample import downsample as _downsample
        self._downsampled = _downsample(self, block=block, progress_cb=progress_cb)
        # First-time downsample → refresh the on-disk grid cache so
        # subsequent launches skip the 170 s pass entirely.
        try:
            self._save_cache()
        except Exception:
            logger.exception("Failed to update grid cache with downsample")
        return self._downsampled

    def _save_cache(self) -> None:
        """Write the full grid + downsample to the .npz cache."""

        from .cache import cache_path_for, save_grid_cache

        spec_fp = f"{self.spec.nx}x{self.spec.ny}x{self.spec.nz}"
        path = cache_path_for(self.master_path, spec_fp)
        save_grid_cache(
            path,
            coord=self.coord,
            active=self.active,
            properties=self.properties,
            downsampled=self._downsampled,
        )


def _pick_include(includes: list[Path], suffix: str) -> Path | None:
    """Find the first include whose stem ends with ``suffix`` (case-insensitive)."""

    suffix_upper = suffix.upper()
    for inc in includes:
        if suffix_upper in inc.stem.upper():
            return inc
    return None
