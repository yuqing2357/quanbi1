"""ZCORN binary cache + memory-mapped access.

ZCORN is the corner-z array (``2*nx * 2*ny * 2*nz`` floats). On our
reference Petrel export it occupies 5.38 GB of ASCII text. Parsing
that on every app launch is prohibitive, and materialising it as a
single numpy array would cost ~4 GB of RAM. This module solves both:

- **First load**: stream the ASCII payload through the GRDECL
  tokeniser, writing batches of float32 straight to a binary cache
  file. Peak memory stays under a megabyte regardless of grid size.
- **Subsequent loads**: ``np.memmap`` the cache. The OS pages in
  exactly the slabs the renderer asks for; the Python process'
  resident size stays small.

Cache invalidation is by source ``(size, mtime)`` fingerprint embedded
in the cache filename, so editing the source file (or replacing it
with a different export) automatically forces a rebuild.

Layout in memory:

    zcorn[i_corner, j_corner, k_corner]

where ``i_corner`` runs 0..2*nx-1, etc. ECLIPSE stores them in
Fortran order with i fastest. Cell ``(i, j, k)``'s eight corners are
located at corner indices ``{2i, 2i+1} × {2j, 2j+1} × {2k, 2k+1}``.
"""

from __future__ import annotations

import logging
import struct
import time
from pathlib import Path

import numpy as np

from .spec import SpecGrid
from .tokens import iter_keyword_floats

logger = logging.getLogger(__name__)

_BATCH_SIZE = 65536


def cache_path_for(source: Path, spec: SpecGrid) -> Path:
    """Compute the deterministic cache path for a ZCORN source file.

    The fingerprint is `(size, mtime_ns)` of the source — small and
    fast, but changes if the user re-exports.
    """

    stat = source.stat()
    cache_dir = source.parent / ".yj_cache"
    name = (
        f"{source.stem}"
        f".{spec.nx}x{spec.ny}x{spec.nz}"
        f".{stat.st_size}.{stat.st_mtime_ns}"
        f".zcorn.f32"
    )
    return cache_dir / name


def build_cache(
    source: Path,
    spec: SpecGrid,
    progress_cb=None,
) -> Path:
    """Stream-parse the ZCORN source into a binary float32 cache.

    Returns the cache path. If the cache already exists with a
    matching fingerprint, it's reused without re-parsing.

    ``progress_cb`` if given is called as ``cb(values_written,
    total_expected)`` every few batches — pass a Qt-friendly closure
    here if you want a progress bar.
    """

    cache = cache_path_for(source, spec)
    expected = spec.zcorn_count
    expected_bytes = expected * 4

    if cache.exists() and cache.stat().st_size == expected_bytes:
        logger.info("ZCORN cache hit: %s", cache)
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename, so a partial write doesn't
    # leave behind a corrupt cache that future runs would trust.
    tmp = cache.with_suffix(cache.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    logger.info(
        "ZCORN cache miss — parsing %s into %s (%d floats, %.2f GB)",
        source, cache, expected, expected_bytes / (1024 ** 3),
    )
    t0 = time.time()
    written = 0
    buf = np.empty(_BATCH_SIZE, dtype=np.float32)
    fill = 0

    with open(tmp, "wb", buffering=1024 * 1024) as out:
        for value in iter_keyword_floats(source, "ZCORN"):
            buf[fill] = value
            fill += 1
            if fill == _BATCH_SIZE:
                out.write(buf.tobytes())
                written += fill
                fill = 0
                if progress_cb is not None and (written % (_BATCH_SIZE * 64) == 0):
                    progress_cb(written, expected)
        if fill:
            out.write(buf[:fill].tobytes())
            written += fill

    if written != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"ZCORN cache build failed: wrote {written} floats, "
            f"expected {expected} ({source})"
        )
    tmp.replace(cache)
    logger.info(
        "ZCORN cache built in %.1fs (%.2f GB)",
        time.time() - t0, expected_bytes / (1024 ** 3),
    )
    if progress_cb is not None:
        progress_cb(expected, expected)
    return cache


def open_zcorn(cache_path: Path, spec: SpecGrid) -> np.memmap:
    """Memory-map a built ZCORN cache as ``(2*nx, 2*ny, 2*nz) float32``.

    The returned memmap is in Fortran order (i_corner fastest), which
    is how ECLIPSE serialised it, so we don't have to copy.
    """

    expected_bytes = spec.zcorn_count * 4
    actual = cache_path.stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"ZCORN cache size mismatch: {actual} bytes vs expected "
            f"{expected_bytes} ({cache_path})"
        )
    mm = np.memmap(
        cache_path,
        dtype=np.float32,
        mode="r",
        shape=(2 * spec.nx, 2 * spec.ny, 2 * spec.nz),
        order="F",
    )
    return mm


def zcorn_for_k_range(
    zcorn: np.memmap, spec: SpecGrid, k0: int, k1: int
) -> np.ndarray:
    """Return a copy of ZCORN for cells with k in ``[k0, k1)``.

    Shape is ``(2*nx, 2*ny, 2*(k1-k0))`` in C-order. We copy because
    the renderer typically wants contiguous arrays — and the slab is
    small enough (a few MB for ~32 K-layers) that the copy is cheap.

    The memmap is left mapped; the OS will reclaim the pages once
    they fall out of the page cache.
    """

    if not (0 <= k0 < k1 <= spec.nz):
        raise ValueError(
            f"bad k range [{k0}, {k1}) for nz={spec.nz}"
        )
    slab = zcorn[:, :, 2 * k0 : 2 * k1]
    # F-order memmap → ascontiguousarray gives us a C-order copy.
    return np.ascontiguousarray(slab)


def cell_corners(
    zcorn_slab: np.ndarray,
    coord: np.ndarray,
    spec: SpecGrid,
    k_offset: int,
    i_range: tuple[int, int] | None = None,
    j_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Materialise ``(ni, nj, nk, 8, 3) float32`` cell corners for a region.

    Inputs:
      zcorn_slab: ``(2*nx, 2*ny, 2*K) float32`` ZCORN slab returned by
        :func:`zcorn_for_k_range`. K is the local layer count.
      coord: full ``(nx+1, ny+1, 6) float32`` COORD pillar array.
      spec: grid spec (only nx/ny used here).
      k_offset: the ``k0`` originally passed to
        :func:`zcorn_for_k_range`. Unused inside; kept for caller-side
        bookkeeping when this output is stitched back into the grid.
      i_range / j_range: optional ``(lo, hi)`` half-open tuples to trim
        the IJ patch. Default is the full IJ plane.

    Output corner ordering (the 8-slot axis):
        slot 0..3 = lower-K face: (SW, SE, NW, NE)
        slot 4..7 = upper-K face: (SW, SE, NW, NE)
    "Lower-K" / "upper-K" refers to the K index direction, not depth.
    Petrel z usually increases downward, so lower-K is geometrically
    above upper-K — but check ``z[slot=0] vs z[slot=4]`` from your data
    if you need to be sure.

    Algorithm: for each of the 4 pillars (i_off, j_off) that touch a
    cell, and each of the 2 K-faces, interpolate the corner xyz along
    the pillar using the ZCORN z-value's fractional position between
    the pillar's top and bottom xyz.
    """

    nx, ny = spec.nx, spec.ny
    del k_offset    # only for caller-side bookkeeping; not used here
    i0, i1 = i_range or (0, nx)
    j0, j1 = j_range or (0, ny)
    nk = zcorn_slab.shape[2] // 2

    ni = i1 - i0
    nj = j1 - j0
    if ni <= 0 or nj <= 0:
        return np.empty((max(ni, 0), max(nj, 0), nk, 8, 3), dtype=np.float32)

    out = np.empty((ni, nj, nk, 8, 3), dtype=np.float32)

    # Precompute corner-index arrays per pillar offset
    ic_base = 2 * np.arange(i0, i1, dtype=np.int64)    # (ni,)
    jc_base = 2 * np.arange(j0, j1, dtype=np.int64)    # (nj,)

    # COORD pillar layout: 6 floats = (x_top, y_top, z_top, x_bot, y_bot, z_bot).
    # The pillar interpolation collapses to z-fraction:
    #     frac = (z_corner - z_top) / (z_bot - z_top)
    #     xy_corner = xy_top + frac * (xy_bot - xy_top)
    # We vectorise over (ni, nj, nk) for one (i_off, j_off, k_off) at a time.

    pillar_offsets = [
        (0, 0, 0),    # SW
        (1, 0, 1),    # SE
        (0, 1, 2),    # NW
        (1, 1, 3),    # NE
    ]
    for di, dj, ij_slot in pillar_offsets:
        pillars = coord[i0 + di : i0 + di + ni, j0 + dj : j0 + dj + nj, :]
        xy_top = pillars[..., 0:2]            # (ni, nj, 2)
        z_top = pillars[..., 2:3]             # (ni, nj, 1)
        xy_bot = pillars[..., 3:5]
        z_bot = pillars[..., 5:6]
        dz = z_bot - z_top
        # Vertical pillars (zero dz) collapse to a single point — use 1
        # to avoid div-by-zero, frac will be 0 so xy stays at top xy.
        dz_safe = np.where(np.abs(dz) < 1e-9, np.float32(1.0), dz)
        xy_delta = xy_bot - xy_top            # (ni, nj, 2)

        # Pillar's contribution to the (ni, nj, nk) cube's 2 K-faces.
        # ZCORN indices on this pillar: (2*i + di, 2*j + dj, 2*k + k_off)
        ic = ic_base + di    # (ni,)
        jc = jc_base + dj    # (nj,)

        for k_face_off in (0, 1):    # lower-K face, then upper-K face
            kc = 2 * np.arange(nk, dtype=np.int64) + k_face_off    # (nk,)
            # Gather (ni, nj, nk) z values for this pillar/face.
            cz = zcorn_slab[np.ix_(ic, jc, kc)]    # (ni, nj, nk) float32
            frac = (cz[..., None] - z_top[..., None, :]) / dz_safe[..., None, :]
            # frac shape: (ni, nj, nk, 1)
            xy = xy_top[..., None, :] + frac * xy_delta[..., None, :]
            # xy shape: (ni, nj, nk, 2). Assemble (ni, nj, nk, 3).
            xyz = np.concatenate([xy, cz[..., None]], axis=-1)
            slot = ij_slot + 4 * k_face_off    # 0..3 lower-K, 4..7 upper-K
            out[:, :, :, slot, :] = xyz

    return out

