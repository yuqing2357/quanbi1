from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_SRC = ROOT / "apps" / "yj_studio" / "src"

# Allow PyTorch + Triton + MKL to coexist on Windows. All three ship
# their own libiomp5md.dll; without this Windows aborts the process on
# the second load (OMP error #15). Setting it before any heavy import
# is enough — env vars propagate to all downstream native libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# triton-windows hits two distinct Windows-only bugs we need to dodge:
#
# 1. The default temp path %LOCALAPPDATA%\Temp\... gets wrapped with a
#    Windows long-path UNC prefix (\\?\C:\...). MSYS2 gcc shipped with
#    triton-windows can't parse that prefix and dies with
#    "cc1.exe: fatal error: \\cuda_utils.c: No such file or directory".
#
# 2. Triton reads its config env vars assuming UTF-8, but Windows
#    surfaces process env in the active ANSI code page (GBK on Chinese
#    systems). If the cache path contains non-ASCII bytes — including
#    Chinese characters like the ones in this project's name — Triton
#    blows up with "'utf-8' codec can't decode byte 0xb1 ...".
#
# Both go away if we point Triton at a SHORT, ASCII-ONLY path. We pick
# C:\yj_triton_cache, completely outside the project tree, so the
# Chinese folder name doesn't leak into any env var Triton reads.
# --- Triton-on-Windows workarounds --------------------------------
#
# triton-windows requires three independent Windows-only fixes for
# anything past the simplest kernel to compile. Apply all of them
# before any module imports triton (i.e. before yj_studio.app).
#
# 1) Cache + tempfile on a short ASCII path. Chinese characters in
#    the env var trip Triton's UTF-8 decode; the system default
#    %LOCALAPPDATA%\Temp can be very long. Hard-override, not
#    setdefault — Windows already populates TMP/TEMP from the user
#    profile and we need to win.
_TRITON_CACHE = Path("C:/yj_triton_cache")
_TRITON_CACHE.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(_TRITON_CACHE)
os.environ["TRITON_HOME"] = str(_TRITON_CACHE)
os.environ["TMP"] = str(_TRITON_CACHE)
os.environ["TEMP"] = str(_TRITON_CACHE)
import tempfile
tempfile.tempdir = str(_TRITON_CACHE)

# 2) Strip the Windows \\?\ long-path prefix before triton hands the
#    paths to MSYS2 gcc. gcc / cc1.exe shipped with triton-windows
#    can't parse that prefix and dies with
#    "cc1.exe: fatal error: \\cuda_utils.c: No such file or directory"
#    on any compile path, regardless of length. Monkey-patch the
#    subprocess call used by triton.runtime.build._build so it
#    rewrites those args before invoking gcc.
def _install_triton_unc_workaround() -> None:
    import subprocess as _sp
    _orig_check_call = _sp.check_call

    def _strip_unc(arg):
        if isinstance(arg, str):
            # Both "\\?\C:\..." and "-I\\?\C:\..." forms appear in the
            # gcc command line — peel the prefix wherever it sits.
            if arg.startswith("\\\\?\\"):
                return arg[4:]
            if len(arg) >= 6 and arg[0] == "-" and arg[1] in "IL" and arg[2:6] == "\\\\?\\":
                return arg[:2] + arg[6:]
        return arg

    def _patched_check_call(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            cmd = type(cmd)(_strip_unc(a) for a in cmd)
        return _orig_check_call(cmd, *args, **kwargs)

    _sp.check_call = _patched_check_call


_install_triton_unc_workaround()

# 3) Allow PyTorch + Triton + MKL libiomp5md.dll to coexist. Set
#    above for completeness, but unrelated to the gcc UNC bug.
# (Handled by the existing KMP_DUPLICATE_LIB_OK setdefault.)


def main() -> int:
    """Run YJ Studio from the repository root without manual PYTHONPATH setup."""

    sys.dont_write_bytecode = True
    src_text = str(PROJECT_SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

    from yj_studio.app import run

    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

 