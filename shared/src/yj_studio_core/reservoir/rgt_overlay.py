"""Single source of truth for the RGT-stratigraphy reservoir section render.

This is the ONE renderer used everywhere a reservoir section is turned into an
RGB image: local QC montages, the desktop viewer, and — critically — the image
fed to SAM3 on the server (seed frame, per-frame JPEG sequence, text
segmentation). "What the user sees" and "what SAM3 eats" are identical by
construction because both call ``render_rgt_section`` with the same params.

Render spec (finalised 2026-06-23, after the carpet search + before/after review):
  * Three base regions, with mud rendered as a high-contrast black background
    so vivid RGT targets keep crisp mask boundaries while no-data stays distinct:
      - no-data : porosity is NaN          -> pure white (255,255,255)
      - mud      : lith==0 AND porosity finite -> black (0,0,0)
      - sand     : lith>0                   -> yellow (255,221,0) blended a=0.7
                                                with the turbo-mapped RGT field
  * RGT gives the stratigraphic colour inside the sand: the RGT slice (on the
    seismic-crop grid, ~half lateral / one-fifth sample of the model) is
    bilinearly upsampled to model coordinates, stretched to ``rgt_span`` and
    turbo-mapped. The turbo LUT is embedded (below) so output is byte-identical
    regardless of whether matplotlib is installed or its version.
  * De-blocking: the native corner-point columns make the discrete display
    staircase ("block-to-block"). Per-class probability fields are
    Gaussian-smoothed with anisotropic sigma (lateral >> depth, so lateral
    steps round off while thin horizontal sand beds survive) and argmax'd back
    to labels. sigma=(lat 4.5, dep 0.9) is the de-block/detail sweet spot.

Consistency note (RGT span): per-slice percentile stretch makes the SAME
horizon a different colour on adjacent frames, which hurts SAM3 video tracking.
For tracking, pass a FIXED ``rgt_span`` (see ``compute_rgt_span``) shared by every
frame of a run, NOT the per-slice default.

Orientation contract:
  * ``lithology`` / ``porosity`` : (n_long, n_sample)  -- raw volume-slice order
    (long/lateral axis first, sample/depth second). This is ``LITH[i]`` for an
    inline and ``LITH[:, j, :]`` for an xline.
  * ``rgt``                      : (rgt_long, rgt_sample) on the RGT grid.
  * return                       : (n_sample, n_long, 3) uint8 -- display order,
    row 0 = shallow. Matches ``slice_to_rgb_image`` (rows = samples/depth).

Pure numpy + scipy (no matplotlib, no Qt). scipy is already a runtime dependency
of this package (native_render uses cKDTree).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

INLINE = 0
XLINE = 1

# RGT-grid downsampling vs the reservoir model. The RGT/seismic crop is the
# 2x lateral / 5x sample node-aligned coarsening of the model, so model index
# ``k`` maps to RGT index ``k // _RGT_LATERAL_STRIDE`` along the long axis.
_RGT_LATERAL_STRIDE = 2

# Embedded matplotlib "turbo" colormap, 256x3 uint8 (== Google Turbo). Embedding
# keeps the render deterministic and matplotlib-free in shared/.
_TURBO_B85 = (
    "MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUas"
    "QkmxQku1Q066RFG/RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjR27mR3HpR3PrR3buR3jw"
    "R3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6"
    "Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXiIMffH8ndHsvaHM3YG9DVGtLSGtTQ"
    "GdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeynJ+6kKu+hLPCe"
    "L/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5f"
    "df5cef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3"
    "sfk2tPg2t/c1ufY1vPU0vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA3"
    "39834d0349s45dk459c56dU569M57NE67s8678068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5"
    "+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks/pYr/pMq/pAp/Y0n/Yom/Icl"
    "/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ7FMP61AO"
    "6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioE"
    "yCgDxSYDwyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4B"
    "lQ0BkgsBjgoBiwkCiAgChQcCgQYCfgUCegQD"
)


def _turbo_lut() -> np.ndarray:
    raw = base64.b64decode(_TURBO_B85)
    return np.frombuffer(raw, dtype=np.uint8).reshape(256, 3).astype(np.float32) / 255.0


_TURBO = _turbo_lut()


@dataclass(frozen=True)
class RgtRenderParams:
    """All knobs for the render. Pin these in config so local == server."""

    alpha: float = 0.7                       # turbo(RGT) weight vs flat yellow
    sigma_lateral: float = 4.5               # gaussian sigma along the long axis
    sigma_depth: float = 0.9                 # gaussian sigma along sample/depth
    rgt_percentile: tuple[float, float] = (2.0, 98.0)  # per-slice stretch span
    sand_rgb: tuple[int, int, int] = (255, 221, 0)
    mud_rgb: tuple[int, int, int] = (0, 0, 0)
    nodata_rgb: tuple[int, int, int] = (255, 255, 255)
    smooth: bool = True                      # False -> nearest/blocky reference

    @classmethod
    def from_mapping(cls, data) -> "RgtRenderParams":
        """Build from a (possibly partial) config dict; unknown keys ignored."""
        if not data:
            return cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kw = {}
        for k, v in dict(data).items():
            if k not in fields:
                continue
            if k in {"rgt_percentile", "sand_rgb", "mud_rgb", "nodata_rgb"}:
                v = tuple(v)
            kw[k] = v
        return cls(**kw)


# Backwards/forwards-friendly default singleton.
DEFAULT_PARAMS = RgtRenderParams()


def extract_rgt_slice(rgt_volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """RGT slice co-registered with model section ``axis``/``index``.

    Model index maps to RGT index by ``index // 2`` along inline/xline (the RGT
    grid is the model's 2x lateral coarsening). Returns (rgt_long, rgt_sample).
    """
    ri = int(index) // _RGT_LATERAL_STRIDE
    if axis == INLINE:
        ri = min(ri, rgt_volume.shape[0] - 1)
        return np.asarray(rgt_volume[ri])
    if axis == XLINE:
        ri = min(ri, rgt_volume.shape[1] - 1)
        return np.asarray(rgt_volume[:, ri, :])
    raise ValueError(f"axis must be INLINE(0) or XLINE(1), got {axis}")


def compute_rgt_span(
    rgt_values: np.ndarray,
    *,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float]:
    """Fixed (lo, hi) RGT stretch span for a whole tracking run.

    Pass the RGT volume (or any representative subset) so every frame shares one
    colour mapping — required for SAM3 tracking (see module note). Returns a
    finite, strictly-increasing pair.
    """
    vals = np.asarray(rgt_values, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(finite, percentile)
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return (lo, hi)


def _turbo_map(t01: np.ndarray) -> np.ndarray:
    """Map [0,1] field to turbo RGB (...,3) float via the embedded LUT.

    Indexing matches matplotlib's ``Colormap.__call__`` (floor(t * N), N=256,
    with t==1 clamped to the last entry) so output is byte-identical to the
    turbo images the spec was tuned on.
    """
    idx = np.clip(np.floor(t01 * 256.0).astype(np.intp), 0, 255)
    return _TURBO[idx]


def render_rgt_section(
    lithology: np.ndarray,
    porosity: np.ndarray,
    rgt: np.ndarray,
    *,
    params: RgtRenderParams | None = None,
    rgt_span: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render one reservoir section to (n_sample, n_long, 3) uint8 RGB.

    Parameters
    ----------
    lithology, porosity : (n_long, n_sample)
        Raw volume-slice order. body = lithology > 0.5; no-data = NaN porosity.
    rgt : (rgt_long, rgt_sample)
        RGT slice on the RGT grid (see ``extract_rgt_slice``). Upsampled here.
    params : RgtRenderParams
        Render knobs (defaults = finalised spec).
    rgt_span : (lo, hi), optional
        Fixed RGT stretch range. None -> per-slice percentile over the body
        (fine for static QC; pass a fixed span for SAM3 tracking).
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    p = params or DEFAULT_PARAMS
    lith = np.asarray(lithology, dtype=np.float32)
    poro = np.asarray(porosity, dtype=np.float32)
    rgt = np.asarray(rgt, dtype=np.float32)
    n_long, n_sample = lith.shape

    sand = lith > 0.5
    nodata = np.isnan(poro)
    mud = (~sand) & (~nodata)
    body_t = sand.T  # (n_sample, n_long)

    sand_c = np.asarray(p.sand_rgb, np.float32) / 255.0
    mud_c = np.asarray(p.mud_rgb, np.float32) / 255.0
    nodata_c = np.asarray(p.nodata_rgb, np.float32) / 255.0

    # --- RGT turbo ramp, bilinearly upsampled to model coords -> (n_sample,n_long)
    gy = np.linspace(0, rgt.shape[0] - 1, n_long)
    gx = np.linspace(0, rgt.shape[1] - 1, n_sample)
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    rgt_up = map_coordinates(rgt, [GY, GX], order=1, mode="nearest").T
    if rgt_span is not None:
        lo, hi = float(rgt_span[0]), float(rgt_span[1])
    elif body_t.any():
        lo, hi = np.percentile(rgt_up[body_t], p.rgt_percentile)
    else:
        lo, hi = float(rgt_up.min()), float(rgt_up.max())
    r01 = np.clip((rgt_up - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    ramp = _turbo_map(r01)
    sand_rgb = (1.0 - p.alpha) * sand_c[None, None, :] + p.alpha * ramp

    # --- region labels: smoothed argmax (de-block) or raw nearest reference
    if p.smooth:
        sig = (max(p.sigma_lateral, 1e-3), max(p.sigma_depth, 1e-3))
        ps = gaussian_filter(sand.astype(np.float32), sig)
        pm = gaussian_filter(mud.astype(np.float32), sig)
        pn = gaussian_filter(nodata.astype(np.float32), sig)
        lab = np.argmax(np.stack([ps, pm, pn], 0), 0).T  # (n_sample,n_long): 0/1/2
    else:
        lab = np.where(body_t, 0, np.where(nodata.T, 2, 1))

    rgb = np.empty((n_sample, n_long, 3), np.float32)
    rgb[lab == 1] = mud_c
    rgb[lab == 2] = nodata_c
    rgb[lab == 0] = sand_rgb[lab == 0]
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
