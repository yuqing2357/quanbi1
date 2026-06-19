from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CRITICAL_FILES = {
    "model_lithology": "data/reservoir/npy_625x625x2_v3/lithology_binary_uint8.npy",
    "model_porosity": "data/reservoir/npy_625x625x2_v3/porosity_float16.npy",
    "reservoir_metadata": "data/reservoir/npy_625x625x2_v3/metadata.json",
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
