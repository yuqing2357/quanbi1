"""Streaming GRDECL reader for Petrel corner-point reservoir grids.

GRDECL is the ECLIPSE ASCII keyword format Petrel exports for grid
geometry (COORD / ZCORN / ACTNUM) and cell properties (LITHOLOGIES,
PORO, ...). Files routinely run to multiple GB on real reservoir models
(our reference file is 5.4 GB), so this module is built around streaming
iteration — never call ``f.read()`` on a GRDECL file.

Public API:

    from yj_studio.io.grdecl import iter_tokens, find_specgrid, read_keyword

The :mod:`parser` submodule has higher-level helpers that combine these
into typed results (``GrdeclSummary``, ``SpecGrid``).
"""

from .spec import GrdeclSummary, SpecGrid
from .tokens import iter_tokens
from .parser import find_specgrid, find_includes, summarize_grdecl

__all__ = [
    "GrdeclSummary",
    "SpecGrid",
    "iter_tokens",
    "find_specgrid",
    "find_includes",
    "summarize_grdecl",
]
