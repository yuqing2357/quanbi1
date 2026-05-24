from __future__ import annotations

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_SEISMIC_NPY = Path(r"F:\YJ-ALL-SEISMIC_depth_0_653.npy")
DEFAULT_PROCESSED_ROOT = Path(r"F:\YJ-ALL-SEISMIC_depth_0_653_processed")
DEFAULT_LITH_POR_MODEL_ROOT = Path(r"F:\YJ-LITH-POR_model_numpy")

LOCAL_PROCESSED_ROOT = WORKSPACE_ROOT / "processed"
LOCAL_LIBS_ROOT = WORKSPACE_ROOT / "libs"
LOCAL_CIGVIS_ROOT = LOCAL_LIBS_ROOT / "cigvis"
LOCAL_WELL_SECTION_ROOT = LOCAL_LIBS_ROOT / "well_section"


def existing_processed_root() -> Path:
    """Return the first processed-data root that exists for this workstation."""

    if LOCAL_PROCESSED_ROOT.exists():
        return LOCAL_PROCESSED_ROOT
    return DEFAULT_PROCESSED_ROOT

