"""GRDECL probe — figure out how big the reservoir model actually is.

Reads only the keywords needed to size the problem (SPECGRID + ACTNUM),
expands ECLIPSE's RLE form (`56064*0`), and prints an inventory:

  - nx, ny, nz, total cell count
  - active cell count + percentage
  - per-K-layer active distribution (so we know if active cells cluster)
  - rough memory estimate for the various data structures we might use

Run via:
    python tools\\probe_grdecl.py

If the only argument is a directory, looks for the standard Petrel
quartet (`*_COORD.GRDECL`, `*_ZCORN.GRDECL`, `*_ACTNUM.GRDECL` and the
master file). Otherwise treat the argument as the master GRDECL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    from project_paths import RESERVOIR_GRDECL_ROOT
except Exception:  # pragma: no cover - direct use outside this repo
    RESERVOIR_GRDECL_ROOT = None


def _open_text(path: Path):
    """Petrel writes the file header in GBK on Chinese Windows. The data
    payload is pure ASCII numbers so falling back to latin-1 is safe."""
    try:
        return open(path, "r", encoding="gbk", errors="replace")
    except Exception:
        return open(path, "r", encoding="latin-1", errors="replace")


def _strip_comment(line: str) -> str:
    idx = line.find("--")
    return line if idx < 0 else line[:idx]


def _iter_tokens(path: Path):
    with _open_text(path) as f:
        for raw in f:
            line = _strip_comment(raw).strip()
            if not line:
                continue
            for tok in line.split():
                yield tok


def find_specgrid(master: Path) -> tuple[int, int, int]:
    it = _iter_tokens(master)
    for tok in it:
        if tok.upper() == "SPECGRID":
            nx = int(next(it))
            ny = int(next(it))
            nz = int(next(it))
            return nx, ny, nz
    raise RuntimeError(f"SPECGRID not found in {master}")


def count_actnum(actnum_path: Path, expected_total: int) -> tuple[int, int, list[int]]:
    """Stream the ACTNUM file once. Returns (active, total, per_k_active).

    per_k_active is filled if total matches expected_total — otherwise we
    can't slice by k yet."""
    active = 0
    total = 0
    started = False
    per_layer = []
    layer_size = 0
    layer_active = 0

    nx_ny = expected_total
    if expected_total > 0:
        # We don't know nz here, but we know nx*ny. Re-derive after we
        # finish counting.
        nx_ny = expected_total

    for tok in _iter_tokens(actnum_path):
        u = tok.upper()
        if not started:
            if u == "ACTNUM":
                started = True
            continue
        if tok == "/":
            break
        if "*" in tok:
            mult_str, val_str = tok.split("*", 1)
            mult = int(mult_str)
            val = int(val_str)
        else:
            mult = 1
            val = int(tok)
        total += mult
        if val:
            active += mult
    return active, total, per_layer


def per_k_actnum(actnum_path: Path, nx: int, ny: int, nz: int) -> list[int]:
    """Second pass: expand ACTNUM and bucket into k-layers without
    materialising the whole array. Each k-layer has nx*ny entries."""
    layer_size = nx * ny
    per_layer = [0] * nz
    idx = 0
    started = False
    for tok in _iter_tokens(actnum_path):
        u = tok.upper()
        if not started:
            if u == "ACTNUM":
                started = True
            continue
        if tok == "/":
            break
        if "*" in tok:
            mult_str, val_str = tok.split("*", 1)
            mult = int(mult_str)
            val = int(val_str)
        else:
            mult = 1
            val = int(tok)
        if val == 0:
            idx += mult
            continue
        # Spread the active count across the layers it spans.
        remaining = mult
        while remaining > 0:
            k = idx // layer_size
            offset_in_layer = idx % layer_size
            take = min(remaining, layer_size - offset_in_layer)
            if 0 <= k < nz:
                per_layer[k] += take
            idx += take
            remaining -= take
    return per_layer


def humansize(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


def find_quartet(root: Path) -> dict[str, Path]:
    grdecls = list(root.glob("*.GRDECL"))
    quartet = {"master": None, "coord": None, "zcorn": None, "actnum": None}
    for p in grdecls:
        upper = p.name.upper()
        if "_COORD" in upper:
            quartet["coord"] = p
        elif "_ZCORN" in upper:
            quartet["zcorn"] = p
        elif "_ACTNUM" in upper:
            quartet["actnum"] = p
        else:
            quartet["master"] = p
    return quartet


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) >= 2 else RESERVOIR_GRDECL_ROOT
    if target is None:
        print("usage: probe_grdecl.py <dir-or-master-grdecl>")
        sys.exit(2)
    if target.is_dir():
        q = find_quartet(target)
        for key in ("master", "coord", "zcorn", "actnum"):
            if q[key] is None:
                print(f"!! could not find {key} GRDECL in {target}")
                sys.exit(2)
            print(f"{key:7s}: {q[key]}  ({humansize(q[key].stat().st_size)})")
        master = q["master"]
        actnum = q["actnum"]
    else:
        master = target
        # Heuristic: replace stem
        actnum = target.with_name(target.stem + "_ACTNUM.GRDECL")
        if not actnum.exists():
            print(f"!! ACTNUM file not found next to {master} (looked for {actnum})")
            sys.exit(2)

    print()
    print("==== SPECGRID ====")
    t0 = time.time()
    nx, ny, nz = find_specgrid(master)
    print(f"  nx, ny, nz = {nx}, {ny}, {nz}")
    total_cells = nx * ny * nz
    print(f"  total cells = {total_cells:,}")
    print(f"  elapsed     = {time.time() - t0:.2f}s")

    print()
    print("==== ACTNUM (first pass: count) ====")
    t0 = time.time()
    active, total, _ = count_actnum(actnum, total_cells)
    print(f"  active cells = {active:,}")
    print(f"  total cells  = {total:,}")
    if total != total_cells:
        print(f"  !! mismatch with SPECGRID total ({total_cells:,})")
    pct = 100.0 * active / max(total, 1)
    print(f"  active ratio = {pct:.2f}%")
    print(f"  elapsed      = {time.time() - t0:.2f}s")

    print()
    print("==== ACTNUM (second pass: per-K distribution) ====")
    t0 = time.time()
    per_k = per_k_actnum(actnum, nx, ny, nz)
    print(f"  elapsed      = {time.time() - t0:.2f}s")
    # Pretty print: show only K layers with any active cell
    nonzero = [(k, c) for k, c in enumerate(per_k) if c > 0]
    print(f"  K layers with any active cell: {len(nonzero)} / {nz}")
    if nonzero:
        first_k = nonzero[0][0]
        last_k = nonzero[-1][0]
        peak_k, peak_c = max(nonzero, key=lambda kv: kv[1])
        print(f"  active K range = [{first_k}, {last_k}]  ({last_k - first_k + 1} layers)")
        print(f"  peak K layer   = k={peak_k} with {peak_c:,} active cells")
        # Print head + tail samples
        print("  --- sample (first 5, peak, last 5) ---")
        for k, c in nonzero[:5]:
            print(f"    k={k:>5}  active={c:,}")
        print(f"    k={peak_k:>5}  active={peak_c:,}   <-- peak")
        for k, c in nonzero[-5:]:
            print(f"    k={k:>5}  active={c:,}")

    print()
    print("==== Memory estimates ====")
    # Full ZCORN array — naive load
    zcorn_count = 8 * total_cells  # 2nx * 2ny * 2nz == 8 * nx*ny*nz
    print(f"  ZCORN float32 (full)       : {humansize(zcorn_count * 4)}")
    # All-cell 8-corner geometry
    geom_full = total_cells * 8 * 3 * 4
    print(f"  cell-corners full          : {humansize(geom_full)}  ({total_cells:,} cells x 8 x 3 x f32)")
    # Active-only geometry
    geom_active = active * 8 * 3 * 4
    print(f"  cell-corners active-only   : {humansize(geom_active)}  ({active:,} cells x 8 x 3 x f32)")
    # COORD pillars
    coord_count = (nx + 1) * (ny + 1) * 6
    print(f"  COORD float32              : {humansize(coord_count * 4)}  ({coord_count:,} floats)")
    # Per-cell scalar property (e.g. PORO)
    prop_full = total_cells * 4
    prop_active = active * 4
    print(f"  scalar property full       : {humansize(prop_full)}")
    print(f"  scalar property active-only: {humansize(prop_active)}")

    print()
    print("==== Verdict ====")
    if active < 5_000_000:
        print("  OK — active cells fit comfortably in memory. Active-only load is viable.")
    elif active < 20_000_000:
        print("  TIGHT — active cells take several GB. Active-only load possible on 16+ GB machines.")
    else:
        print("  HARD — active cells alone are huge. Need streaming or block loading.")


if __name__ == "__main__":
    main()
