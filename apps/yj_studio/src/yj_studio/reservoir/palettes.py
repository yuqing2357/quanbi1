"""Shared colour palettes for reservoir property visualisation.

Lithology and other categorical reservoir properties have a fixed
value-to-colour mapping that needs to stay in sync across the 2D
section view, the 3D overview renderer, and any future legend or
mesh export. Define each palette once here.

YJ project notes — these palettes encode the on-site geological
interpretation of value codes, which intentionally overrides
``F:/YJ-LITH-POR_model_numpy/metadata.json``:

    LITHOLOGIES:
        0 → 泥岩 (mud)     grey
        1 → 砂岩 (sand)    yellow
        2 → 砥岩 (gravel)  cyan

If geology re-verifies the assignment against Petrel and finds the
metadata is right after all, flip the entries here and the rest of
the app picks it up automatically.
"""

from __future__ import annotations

from matplotlib.colors import ListedColormap


LITHOLOGY_PALETTE = ListedColormap(
    [
        (150 / 255, 150 / 255, 150 / 255, 1.0),    # 0 — 泥岩 grey
        (245 / 255, 214 / 255,  45 / 255, 1.0),    # 1 — 砂岩 yellow
        (  0 / 255, 220 / 255, 220 / 255, 1.0),    # 2 — 砥岩 cyan
    ],
    name="lithology",
)


def palette_for(property_name: str) -> ListedColormap | None:
    """Return a discrete palette for a known categorical property, or None."""
    if property_name == "LITHOLOGIES":
        return LITHOLOGY_PALETTE
    return None
