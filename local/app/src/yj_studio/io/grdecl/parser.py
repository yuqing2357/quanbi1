"""High-level GRDECL parsing.

The functions here all build on :mod:`tokens` and never materialise
the whole token stream. They return either small typed records (e.g.
:class:`SpecGrid`) or numpy arrays sized exactly to fit the keyword
payload, so callers don't have to grow a list element-by-element.

Typical flow:

    summary = summarize_grdecl(master_path)
    nx, ny, nz = summary.specgrid.nx, ..., ...
    actnum = read_actnum(actnum_path, summary.specgrid)
    pillars = read_coord(coord_path, summary.specgrid)
    # ZCORN goes through binary cache — see ``zcorn_cache.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spec import GrdeclSummary, SpecGrid
from .tokens import (
    iter_keyword_floats,
    iter_keyword_ints,
    iter_tokens,
    open_text,
    strip_comment,
)


# ----------------------------------------------------------------- discovery


def find_specgrid(master_path: Path) -> SpecGrid:
    """Pull the SPECGRID record out of the master GRDECL.

    Reads only as far as needed (a few hundred lines into the file).
    """

    it = iter_tokens(master_path)
    for tok in it:
        if tok.upper() != "SPECGRID":
            continue
        nx = int(next(it))
        ny = int(next(it))
        nz = int(next(it))
        # numres + grid type are optional — read until '/'.
        numres = 1
        grid_type = "F"
        seen_numres = False
        for tok2 in it:
            if tok2 == "/":
                break
            if not seen_numres:
                try:
                    numres = int(tok2)
                    seen_numres = True
                    continue
                except ValueError:
                    pass
            grid_type = tok2
            break
        return SpecGrid(nx=nx, ny=ny, nz=nz, numres=numres, grid_type=grid_type)
    raise KeyError(f"SPECGRID not found in {master_path}")


def find_includes(master_path: Path) -> list[Path]:
    """Resolve all ``INCLUDE 'file' /`` directives to absolute paths.

    Petrel-exported masters reference the COORD / ZCORN / ACTNUM
    files via INCLUDE. The path is quoted (single or double) and
    interpreted relative to the master file's directory.
    """

    base_dir = master_path.parent
    includes: list[Path] = []
    with open_text(master_path) as f:
        in_include = False
        for raw in f:
            line = strip_comment(raw).strip()
            if not line:
                continue
            if not in_include:
                if line.upper().startswith("INCLUDE"):
                    in_include = True
                    rest = line[len("INCLUDE") :].strip()
                    if rest:
                        # Single-line form: ``INCLUDE 'file' /``
                        path = _extract_quoted(rest)
                        if path is not None:
                            includes.append((base_dir / path).resolve())
                            in_include = False
                continue
            # Continuation of an INCLUDE record
            path = _extract_quoted(line)
            if path is not None:
                includes.append((base_dir / path).resolve())
            if "/" in line:
                in_include = False
    return includes


def _extract_quoted(text: str) -> str | None:
    for quote in ("'", '"'):
        i = text.find(quote)
        if i < 0:
            continue
        j = text.find(quote, i + 1)
        if j < 0:
            continue
        return text[i + 1 : j]
    return None


def summarize_grdecl(master_path: Path) -> GrdeclSummary:
    """Scan once: SPECGRID, INCLUDEs, top-level keyword inventory.

    The keyword list is what we see at top level of the master file;
    it does NOT recurse into INCLUDEs. Use this for a quick "what's in
    here" report before deciding what to read in full.
    """

    summary = GrdeclSummary(master_path=master_path)
    keywords: list[str] = []
    in_payload_of: str | None = None

    with open_text(master_path) as f:
        for raw in f:
            line = strip_comment(raw).strip()
            if not line:
                continue
            for tok in line.split():
                if in_payload_of is not None:
                    if tok == "/":
                        in_payload_of = None
                    continue
                if not tok[:1].isalpha():
                    continue
                upper = tok.upper()
                # INCLUDE is handled by a dedicated pass below — here
                # we just step over its single-record payload.
                if upper == "INCLUDE":
                    in_payload_of = upper
                    continue
                keywords.append(upper)
                in_payload_of = upper

    summary.keywords_seen = keywords
    summary.includes = find_includes(master_path)
    try:
        summary.specgrid = find_specgrid(master_path)
    except KeyError:
        summary.specgrid = None
    return summary


# ----------------------------------------------------------------- array reads


def read_actnum(actnum_path: Path, spec: SpecGrid) -> np.ndarray:
    """Read ACTNUM as a (nx, ny, nz) int8 array in Fortran order.

    ECLIPSE arrays are stored in F-order: ``i`` varies fastest, then
    ``j``, then ``k``. We return a C-contiguous array shaped (nx, ny,
    nz) — callers can index as ``a[i, j, k]`` naturally.
    """

    n = spec.total_cells
    out = np.empty(n, dtype=np.int8)
    idx = 0
    for value in iter_keyword_ints(actnum_path, "ACTNUM"):
        if idx >= n:
            raise ValueError(
                f"ACTNUM has more values than SPECGRID expects "
                f"({idx + 1} > {n})"
            )
        out[idx] = value
        idx += 1
    if idx != n:
        raise ValueError(
            f"ACTNUM short read: got {idx} values, expected {n}"
        )
    # Stored Fortran-order: first index varies fastest.
    return out.reshape((spec.nx, spec.ny, spec.nz), order="F").copy()


def read_coord(coord_path: Path, spec: SpecGrid) -> np.ndarray:
    """Read COORD as a ``(nx+1, ny+1, 6) float32`` array.

    ECLIPSE layout: pillar (i, j) at 1D offset ``((nx+1)*j + i) * 6``,
    with i varying fastest. Each pillar carries 6 floats in order
    (x_top, y_top, z_top, x_bot, y_bot, z_bot) and the 6 is the
    *fastest*-varying axis (pillar-internal offset).

    The naive ``reshape((nx+1, ny+1, 6), order='F')`` is WRONG because
    F-order makes the first axis fastest, but ECLIPSE puts the 6
    (pillar-internal) axis fastest. Correct path: reshape as
    ``(6, nx+1, ny+1)`` F-order (so the 6-axis ends up fastest in
    memory) and then move that axis to the end.
    """

    n = spec.coord_count
    out = np.empty(n, dtype=np.float32)
    idx = 0
    for value in iter_keyword_floats(coord_path, "COORD"):
        if idx >= n:
            raise ValueError(
                f"COORD has more values than SPECGRID expects "
                f"({idx + 1} > {n})"
            )
        out[idx] = value
        idx += 1
    if idx != n:
        raise ValueError(
            f"COORD short read: got {idx} values, expected {n}"
        )
    return np.ascontiguousarray(
        out.reshape((6, spec.nx + 1, spec.ny + 1), order="F").transpose(1, 2, 0)
    )


def read_int_property(
    path: Path, keyword: str, spec: SpecGrid
) -> np.ndarray:
    """Read a single int property keyword to a (nx, ny, nz) int32 array."""

    n = spec.total_cells
    out = np.empty(n, dtype=np.int32)
    idx = 0
    for value in iter_keyword_ints(path, keyword):
        if idx >= n:
            raise ValueError(
                f"{keyword} overflow: {idx + 1} > {n}"
            )
        out[idx] = value
        idx += 1
    if idx != n:
        raise ValueError(
            f"{keyword} short read: {idx} / {n}"
        )
    return out.reshape((spec.nx, spec.ny, spec.nz), order="F").copy()


def read_float_property(
    path: Path, keyword: str, spec: SpecGrid
) -> np.ndarray:
    """Read a single float property keyword to a (nx, ny, nz) float32 array."""

    n = spec.total_cells
    out = np.empty(n, dtype=np.float32)
    idx = 0
    for value in iter_keyword_floats(path, keyword):
        if idx >= n:
            raise ValueError(
                f"{keyword} overflow: {idx + 1} > {n}"
            )
        out[idx] = value
        idx += 1
    if idx != n:
        raise ValueError(
            f"{keyword} short read: {idx} / {n}"
        )
    return out.reshape((spec.nx, spec.ny, spec.nz), order="F").copy()
