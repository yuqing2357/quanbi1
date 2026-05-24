"""Quick check: for i=186, what K range is actually active?"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "apps" / "yj_studio" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from yj_studio.reservoir import ReservoirGrid


def main() -> None:
    grid = ReservoirGrid.load_from_master(Path(r"F:\１２３４.GRDECL"))
    i = 186
    strip = grid.active[i, :, :] != 0   # (ny, nz)
    print(f"i={i}: total active cells = {int(strip.sum())}")
    active_k = np.argwhere(strip.any(axis=0)).ravel()
    if active_k.size:
        print(f"  active K range: [{active_k.min()}, {active_k.max()}]  (len={active_k.size})")
    # z_center for this strip
    zc = np.load(r"F:\YJ-LITH-POR_model_numpy\z_center_native_i_j_k.npy")
    zc_strip = zc[i, :, :]
    zc_active = zc_strip[strip]
    if zc_active.size:
        print(f"  z_center range on active: [{zc_active.min():.2f}, {zc_active.max():.2f}] m")
        print(f"  → sample range: [{zc_active.min()/10:.2f}, {zc_active.max()/10:.2f}]")


if __name__ == "__main__":
    main()
