from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "apps" / "yj_studio" / "src"
DATA_ROOT = PROJECT_ROOT / "data"
SEISMIC_ROOT = DATA_ROOT / "seismic"
RESERVOIR_ROOT = DATA_ROOT / "reservoir"
RESERVOIR_GRDECL_ROOT = RESERVOIR_ROOT / "grdecl"
RESERVOIR_NUMPY_ROOT = RESERVOIR_ROOT / "numpy"
DEFAULT_RESERVOIR_MASTER = RESERVOIR_GRDECL_ROOT / "１２３４.GRDECL"


def add_app_src_to_path() -> None:
    src_text = str(SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def reservoir_master_from_arg(arg: str | None) -> Path:
    if arg is None:
        return DEFAULT_RESERVOIR_MASTER
    path = Path(arg)
    if path.is_dir():
        cands = [
            p
            for p in path.glob("*.GRDECL")
            if "_COORD" not in p.name.upper()
            and "_ZCORN" not in p.name.upper()
            and "_ACTNUM" not in p.name.upper()
        ]
        if not cands:
            raise FileNotFoundError(f"No master GRDECL found in {path}")
        return cands[0]
    return path
