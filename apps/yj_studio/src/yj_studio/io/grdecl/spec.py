from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SpecGrid:
    """ECLIPSE SPECGRID record: grid size + numres + grid-type flag.

    A typical Petrel SPECGRID looks like ``372 343 1076 1 F`` — we only
    need the first three numbers for sizing; ``numres`` and the type
    flag are kept for round-trip but unused by the renderer.
    """

    nx: int
    ny: int
    nz: int
    numres: int = 1
    grid_type: str = "F"

    @property
    def total_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def zcorn_count(self) -> int:
        """Number of float values in a complete ZCORN array.

        Each cell contributes 8 corner-z values, but neighbouring cells
        share corners. ECLIPSE stores them duplicated: shape is
        ``(2*nx, 2*ny, 2*nz)`` → ``8 * nx * ny * nz`` floats.
        """
        return 8 * self.total_cells

    @property
    def coord_count(self) -> int:
        """Number of float values in the COORD pillar array.

        COORD has ``(nx+1) * (ny+1)`` pillars; each pillar carries one
        top point + one bottom point in xyz → 6 floats per pillar.
        """
        return (self.nx + 1) * (self.ny + 1) * 6


@dataclass(slots=True)
class GrdeclSummary:
    """What we discovered by scanning a master GRDECL once.

    ``includes`` are resolved to absolute paths when possible. Keyword
    locations are not stored — callers re-iterate the file lazily when
    they want to read array payloads.
    """

    master_path: Path
    specgrid: SpecGrid | None = None
    includes: list[Path] = field(default_factory=list)
    keywords_seen: list[str] = field(default_factory=list)
