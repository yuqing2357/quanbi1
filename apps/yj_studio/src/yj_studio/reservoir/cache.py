"""On-disk cache for ReservoirGrid load output.

Loading a ~5 GB Petrel GRDECL the slow way takes ~3.5 minutes:

    ACTNUM parse        15 s
    COORD parse          1 s
    LITHOLOGIES parse   ~15 s
    PORO parse          ~15 s
    Downsampling (2×2×4) 170 s     ← biggest

ZCORN already has its own binary cache; everything else above was
re-derived from ASCII every launch. This module captures the lot
into a single ``.npz`` so subsequent loads are dominated by disk
I/O — typically 5–10 s for the YJ reference grid.

Layout::

    <grdecl-folder>\\.yj_cache\\<stem>.<grdecl-mtime>.<spec-fingerprint>.grid.npz

    spec.npz:
        - actnum            int8  (nx, ny, nz)
        - coord             float32 (nx+1, ny+1, 6)
        - prop_names        list[str]    (saved as object array)
        - prop_<name>       ndarray per property
        - ds_corners        float32 (Nx, Ny, Nz, 8, 3)
        - ds_active         bool    (Nx, Ny, Nz)
        - ds_block          int32   (3,)
        - ds_int_prop_names list[str]
        - ds_int_<name>     int32 array
        - ds_float_prop_names
        - ds_float_<name>   float32 array

Cache invalidation: filename embeds the GRDECL mtime, so editing the
source file naturally produces a new key — the old cache is left
behind for cleanup later.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def cache_path_for(master_path: Path, spec_fingerprint: str) -> Path:
    """Return the .npz cache path for a given GRDECL file + spec fingerprint.

    Lives alongside the ZCORN binary cache in ``.yj_cache`` next to the
    source GRDECL.
    """

    root = master_path.parent / ".yj_cache"
    root.mkdir(parents=True, exist_ok=True)
    try:
        mtime_ns = master_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    stem = master_path.stem
    return root / f"{stem}.{mtime_ns}.{spec_fingerprint}.grid.npz"


def save_grid_cache(
    cache_path: Path,
    *,
    coord: np.ndarray,
    active: np.ndarray,
    properties: dict[str, np.ndarray],
    downsampled,    # DownsampledGrid | None
) -> None:
    """Serialise everything ReservoirGrid needs (except the ZCORN memmap)."""

    payload: dict[str, np.ndarray] = {
        "coord": np.ascontiguousarray(coord),
        "actnum": np.ascontiguousarray(active),
        "prop_names": np.array(list(properties.keys()), dtype=object),
    }
    for name, arr in properties.items():
        payload[f"prop_{name}"] = np.ascontiguousarray(arr)

    if downsampled is not None:
        payload["ds_corners"] = downsampled.corners
        payload["ds_active"] = downsampled.active
        payload["ds_block"] = np.array(downsampled.block, dtype=np.int32)
        payload["ds_shape"] = np.array(downsampled.shape, dtype=np.int32)
        payload["ds_source_shape"] = np.array(downsampled.source_shape, dtype=np.int32)
        payload["ds_int_prop_names"] = np.array(
            list(downsampled.int_properties.keys()), dtype=object,
        )
        for name, arr in downsampled.int_properties.items():
            payload[f"ds_int_{name}"] = arr
        payload["ds_float_prop_names"] = np.array(
            list(downsampled.float_properties.keys()), dtype=object,
        )
        for name, arr in downsampled.float_properties.items():
            payload[f"ds_float_{name}"] = arr

    # Write to a sibling .partial file then atomically replace.
    # np.savez's "auto-append .npz to the filename" behaviour is bypassed
    # by handing it an opened binary file handle — that way the path
    # we close ends up being the exact name we want to rename from.
    tmp_path = cache_path.with_name(cache_path.name + ".partial")
    with open(tmp_path, "wb") as f:
        np.savez(f, **payload)
    tmp_path.replace(cache_path)
    size_mb = cache_path.stat().st_size / 1024**2
    logger.info(
        "Wrote ReservoirGrid cache (%.1f MB) → %s. "
        "Next launch will load this grid in ~5-10 s instead of ~3 min.",
        size_mb, cache_path,
    )


def load_grid_cache(cache_path: Path):
    """Read a previously-saved .npz back into the pieces ReservoirGrid expects.

    Returns ``(coord, active, properties_dict, downsampled_payload_or_None)``.
    ``downsampled_payload`` is a dict the caller can splat into a
    DownsampledGrid constructor.

    Uses ``mmap_mode='r'`` so the huge ds_corners array isn't fully
    loaded into RSS; it's paged in on demand. allow_pickle=True is
    needed for the object-dtype name lists.
    """

    with np.load(cache_path, mmap_mode="r", allow_pickle=True) as zf:
        coord = np.asarray(zf["coord"])
        active = np.asarray(zf["actnum"])
        prop_names = list(zf["prop_names"])
        properties: dict[str, np.ndarray] = {}
        for name in prop_names:
            properties[name] = np.asarray(zf[f"prop_{name}"])

        downsampled_payload = None
        if "ds_corners" in zf.files:
            int_names = list(zf["ds_int_prop_names"])
            float_names = list(zf["ds_float_prop_names"])
            downsampled_payload = {
                "corners": np.asarray(zf["ds_corners"]),
                "active": np.asarray(zf["ds_active"]),
                "block": tuple(int(x) for x in zf["ds_block"]),
                "shape": tuple(int(x) for x in zf["ds_shape"]),
                "source_shape": tuple(int(x) for x in zf["ds_source_shape"]),
                "int_properties": {n: np.asarray(zf[f"ds_int_{n}"]) for n in int_names},
                "float_properties": {n: np.asarray(zf[f"ds_float_{n}"]) for n in float_names},
            }

    logger.info("Loaded ReservoirGrid cache from %s", cache_path)
    return coord, active, properties, downsampled_payload
