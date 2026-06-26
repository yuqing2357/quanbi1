"""On-demand reservoir slice rendering from native corner-point columns."""

from .native_render import (
    INLINE,
    XLINE,
    NativeColumnRenderer,
    ReservoirGeometry,
    SectionResult,
)
from .rgt_overlay import (
    DEFAULT_PARAMS,
    RgtRenderParams,
    compute_rgt_span,
    extract_rgt_slice,
    render_rgt_section,
)

__all__ = [
    "INLINE",
    "XLINE",
    "NativeColumnRenderer",
    "ReservoirGeometry",
    "SectionResult",
    "DEFAULT_PARAMS",
    "RgtRenderParams",
    "compute_rgt_span",
    "extract_rgt_slice",
    "render_rgt_section",
]
