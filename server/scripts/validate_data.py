from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CRITICAL_FILES = {
    "lithology_3x": "data/reservoir/numpy_3x/lithology_binary_3x_uint8.npy",
    "porosity_3x": "data/reservoir/numpy_3x/porosity_3x_float16.npy",
    "seismic": "data/seismic/YJ-ALL-SEISMIC.npy",
    "sam3_checkpoint": "weights/sam3.pt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate server-side YJ Studio data files.")
    parser.add_argument("--root", type=Path, default=Path("/root/quanbi"))
    args = parser.parse_args()

    ok = True
    for name, rel in CRITICAL_FILES.items():
        path = args.root / rel
        if not path.exists():
            print(f"[FAIL] {name}: missing {path}")
            ok = False
            continue
        size_gb = path.stat().st_size / 1024**3
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            print(f"[OK]   {name}: {arr.shape} {arr.dtype} ({size_gb:.2f} GB)")
        else:
            print(f"[OK]   {name}: {size_gb:.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
