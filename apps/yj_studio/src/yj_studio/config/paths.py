from __future__ import annotations

from pathlib import Path


def _find_workspace_root() -> Path:
    """Locate the repo root by markers instead of a fragile parent depth.

    Walks up from this file looking for the project root. Using markers keeps
    this correct even if the package is moved to a different depth (e.g. the
    planned apps/yj_studio -> local/ migration).
    """

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
        if (parent / "libs").is_dir() and (parent / "config").is_dir():
            return parent
    return here.parents[5]


WORKSPACE_ROOT = _find_workspace_root()
DATA_ROOT = WORKSPACE_ROOT / "data"
CACHE_ROOT = WORKSPACE_ROOT / "cache"

SEISMIC_DATA_ROOT = DATA_ROOT / "seismic"
RESERVOIR_DATA_ROOT = DATA_ROOT / "reservoir"

DEFAULT_SEISMIC_NPY = SEISMIC_DATA_ROOT / "YJ-ALL-SEISMIC_depth_0_653.npy"
DEFAULT_PROCESSED_ROOT = SEISMIC_DATA_ROOT / "processed"
DEFAULT_LITH_POR_MODEL_ROOT = RESERVOIR_DATA_ROOT / "numpy_3x"

LOCAL_PROCESSED_ROOT = WORKSPACE_ROOT / "processed"
LOCAL_LIBS_ROOT = WORKSPACE_ROOT / "libs"
LOCAL_CIGVIS_ROOT = LOCAL_LIBS_ROOT / "cigvis"
LOCAL_WELL_SECTION_ROOT = LOCAL_LIBS_ROOT / "well_section"


def existing_processed_root() -> Path:
    """Return the processed-data root inside the portable project."""

    return DEFAULT_PROCESSED_ROOT
