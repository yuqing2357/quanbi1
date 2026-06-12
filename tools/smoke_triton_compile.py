"""Smoke-test triton-windows can JIT-compile a kernel on this box.

The full SAM3 model takes ~30 s to load and lives behind a click-
through workflow in YJ Studio. That's too slow for iterating on the
Triton-on-Windows workarounds (cache path, \\\\?\\ UNC prefix, OMP
library conflicts), so this script does the *minimum* that exercises
the same compile path:

  1. Apply the same env + monkey-patches run_yj_studio.py applies.
  2. Run a one-element kernel through triton.

A successful first run takes ~10 s (gcc + cuda compile). Subsequent
runs are cache hits and finish in under a second.

If this script passes, the SAM3 video predictor's NMS kernel will
compile too — they hit the exact same triton.runtime.build code path.

Run via:
    python tools\\smoke_triton_compile.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Mirror run_yj_studio.py's Triton workarounds exactly ----------

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_TRITON_CACHE = Path("C:/yj_triton_cache")
_TRITON_CACHE.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(_TRITON_CACHE)
os.environ["TRITON_HOME"] = str(_TRITON_CACHE)
os.environ["TMP"] = str(_TRITON_CACHE)
os.environ["TEMP"] = str(_TRITON_CACHE)
import tempfile
tempfile.tempdir = str(_TRITON_CACHE)


def _install_triton_unc_workaround() -> None:
    import subprocess as _sp
    _orig_check_call = _sp.check_call

    def _strip_unc(arg):
        if isinstance(arg, str):
            if arg.startswith("\\\\?\\"):
                return arg[4:]
            if len(arg) >= 6 and arg[0] == "-" and arg[1] in "IL" and arg[2:6] == "\\\\?\\":
                return arg[:2] + arg[6:]
        return arg

    def _patched_check_call(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            cmd = type(cmd)(_strip_unc(a) for a in cmd)
            print(f"  [patched cmd] {cmd[0]} ... {len([a for a in cmd if a.startswith('-')])} flags")
        return _orig_check_call(cmd, *args, **kwargs)

    _sp.check_call = _patched_check_call


_install_triton_unc_workaround()


# --- Now the actual test ------------------------------------------

def main() -> int:
    print(f"TRITON_CACHE_DIR = {os.environ['TRITON_CACHE_DIR']}")
    print(f"TMP              = {os.environ['TMP']}")
    print(f"tempfile.tempdir = {tempfile.tempdir}")
    print()

    print("Importing triton...")
    t0 = time.time()
    import triton
    import triton.language as tl
    print(f"  triton {triton.__version__} imported in {time.time() - t0:.2f}s")

    print("Importing torch + checking CUDA...")
    t0 = time.time()
    import torch
    print(f"  torch {torch.__version__} imported in {time.time() - t0:.2f}s")
    if not torch.cuda.is_available():
        print("  !! CUDA not available — triton needs a GPU. Aborting.")
        return 2
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    # The simplest possible JIT kernel — sum two scalars.
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, x + y)

    print()
    print("Compiling + launching a tiny triton kernel...")
    print("(First run triggers gcc + cuda compile; subsequent runs hit cache.)")
    print()
    x = torch.arange(0, 1024, device="cuda", dtype=torch.float32)
    y = torch.arange(0, 1024, device="cuda", dtype=torch.float32) * 2.0
    out = torch.empty_like(x)
    t0 = time.time()
    add_kernel[(1,)](x, y, out, BLOCK=1024)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  kernel completed in {elapsed:.2f}s")

    expected = x + y
    if torch.allclose(out, expected):
        print(f"  result correct ({out[0].item():.1f}, {out[-1].item():.1f})")
    else:
        print(f"  !! result mismatch")
        return 3

    print()
    print("==== PASS — triton-windows compile works end-to-end. ====")
    print("SAM3 video predictor should now load and run without the")
    print("gcc UNC / NMS kernel errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
