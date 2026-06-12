"""Sanity-check that the vendored SAM3 source + weights are reachable.

Run this **after** copying assets in via tools/copy_sam3_assets.ps1.

    python tools/check_sam3_setup.py

Exits 0 if everything is in place, 1 otherwise. Does NOT load the model
(that would take 30s and 1.5 GB GPU); only checks paths and ``import sam3``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ok = True

    sam3_pkg = ROOT / "libs" / "sam3"
    if not (sam3_pkg / "__init__.py").exists():
        print(f"[FAIL] {sam3_pkg / '__init__.py'} not found")
        print("       Run tools/copy_sam3_assets.ps1 with an explicit -SourceRoot")
        ok = False
    else:
        print(f"[OK]   {sam3_pkg / '__init__.py'} exists")

    model_builder = sam3_pkg / "model_builder.py"
    if not model_builder.exists():
        print(f"[FAIL] {model_builder} not found")
        print("       libs/sam3/ looks incomplete — re-copy the full package")
        ok = False
    else:
        print(f"[OK]   {model_builder} exists")

    bpe = sam3_pkg / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.exists():
        print(f"[FAIL] {bpe} not found (text encoder needs the BPE vocab)")
        ok = False
    else:
        print(f"[OK]   {bpe} exists")

    ckpt = ROOT / "weights" / "sam3.pt"
    if not ckpt.exists():
        print(f"[FAIL] {ckpt} not found")
        print("       Run tools/copy_sam3_assets.ps1 with an explicit -SourceRoot")
        ok = False
    else:
        size_gb = ckpt.stat().st_size / 1024**3
        print(f"[OK]   {ckpt} present ({size_gb:.2f} GB)")

    # Try importing sam3 with the vendored package on sys.path.
    libs_dir = str(ROOT / "libs")
    if libs_dir not in sys.path:
        sys.path.insert(0, libs_dir)
    try:
        import sam3  # noqa: F401

        print(f"[OK]   import sam3 -> {sam3.__file__}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import sam3 failed: {exc}")
        ok = False

    print()
    if ok:
        print("All SAM3 assets in place. AI Dock 'Start AI' should work.")
        return 0
    print("Some assets are missing. Fix the FAIL lines above and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
