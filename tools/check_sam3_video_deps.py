"""Probe every import the SAM3 video predictor reaches at runtime.

Walks all sam3.model.* modules that build_sam3_video_model and its
descendants touch, catches ModuleNotFoundError for each, and prints
the missing third-party module names. Lets us batch-install with one
``pip install`` instead of restarting YJ Studio per missing dep.

Run via:
    set KMP_DUPLICATE_LIB_OK=TRUE
    E:\\miniconda\\envs\\py312\\python.exe tools\\check_sam3_video_deps.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

LIBS = Path(__file__).resolve().parent.parent / "libs"
if str(LIBS) not in sys.path:
    sys.path.insert(0, str(LIBS))

# Modules to try in order. Each is what build_sam3_video_model + its
# children reach when constructed. We import top-down; the first
# failure on each path tells us a missing 3rd-party dep.
TARGETS = [
    "sam3.model_builder",
    "sam3.model.sam3_video_predictor",
    "sam3.model.sam3_video_base",
    "sam3.model.sam3_video_inference",
    "sam3.model.sam3_tracking_predictor",
    "sam3.model.sam3_tracker_base",
    "sam3.model.sam3_tracker_utils",
    "sam3.model.memory",
    "sam3.model.edt",
]


def main() -> None:
    missing: set[str] = set()
    print(f"Probing SAM3 video imports under {LIBS}\n")
    for mod_name in TARGETS:
        try:
            importlib.import_module(mod_name)
            print(f"  OK    {mod_name}")
        except ModuleNotFoundError as exc:
            print(f"  MISS  {mod_name}: {exc.msg}")
            # exc.name is the unloadable module — collect for batch install
            if exc.name:
                # Take the top-level package (e.g. decord.x.y → decord)
                missing.add(exc.name.split(".")[0])
        except Exception as exc:
            print(f"  ERR   {mod_name}: {type(exc).__name__}: {exc}")

    print()
    if missing:
        print("Missing top-level modules:", sorted(missing))
        print()
        cmd = " ".join(sorted(missing))
        print(f"Suggested install:")
        print(f"  E:\\miniconda\\envs\\py312\\python.exe -m pip install {cmd}")
    else:
        print("All targeted modules imported cleanly — SAM3 video deps OK.")


if __name__ == "__main__":
    main()
