"""Map Petrel grid-local coordinates → seismic sample-index coordinates.

YJ Studio's 3D scene lives in the seismic sample-index frame
``(axis0, axis1, sample)`` — that's the frame horizons, faults, wells
and slices already use. Petrel-exported reservoir grids store cell
corners in a *grid-local* xy frame (not UTM), so two steps are needed:

1. **MAPAXES rotation**: local xy → world UTM. The GRDECL MAPAXES
   record gives three points defining a local frame:

       MAPAXES x1 y1 x2 y2 x3 y3
       - (x2, y2) is the local origin in world UTM
       - (x1, y1) lies on the local +Y axis
       - (x3, y3) lies on the local +X axis

   World xy is then::

       world_xy = origin + local_x * x_unit + local_y * y_unit

   where ``x_unit`` and ``y_unit`` are unit vectors along the local
   axes (so the local frame can be rotated arbitrarily vs. UTM).
   For the YJ reference dataset y_unit ≈ (0, -1), which is why
   axis1 ends up mirrored if you skip this step.

2. **Seismic indexing**: world UTM → integer sample indices::

       axis0 = (world_x - X0) / DX
       axis1 = (world_y - Y0) / DY
       sample = z_m / DZ

The defaults here are the YJ project values copied from
``D:\\商书记项目\\tools\\convert_grdecl_lith_por_to_numpy.py`` (the
rasteriser that wrote ``lithology_volume_seismic.npy``), so anything
produced by this transform aligns with what the legacy tool produced.
A different Petrel project would override the dataclass fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Constants from convert_grdecl_lith_por_to_numpy.py at D:/商书记项目/.
DEFAULT_MAPAXES = (
    632661.7332, 4173682.699,   # x1, y1 — point on local +Y axis
    632661.7332, 4174682.699,   # x2, y2 — local origin (in UTM)
    633661.7332, 4174682.699,   # x3, y3 — point on local +X axis
)
DEFAULT_X_ORIGIN = 630200.0
DEFAULT_Y_ORIGIN = 4154988.0
DEFAULT_XY_STEP = 12.5
DEFAULT_Z_STEP = 10.0


@dataclass(frozen=True, slots=True)
class SeismicIndexTransform:
    """Affine mapping local (x_m, y_m, z_m) → seismic (axis0, axis1, sample).

    The local frame is the one Petrel writes into COORD pillars; the
    target frame is the seismic volume's integer sample indices.
    """

    mapaxes: tuple[float, ...] = DEFAULT_MAPAXES
    x_origin: float = DEFAULT_X_ORIGIN
    y_origin: float = DEFAULT_Y_ORIGIN
    xy_step: float = DEFAULT_XY_STEP
    z_step: float = DEFAULT_Z_STEP

    def _axis_units(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(origin, x_unit, y_unit)`` derived from MAPAXES."""
        ma = np.asarray(self.mapaxes, dtype=np.float64)
        point_on_y = ma[0:2]
        origin = ma[2:4]
        point_on_x = ma[4:6]
        x_vec = point_on_x - origin
        y_vec = point_on_y - origin
        x_unit = x_vec / np.linalg.norm(x_vec)
        y_unit = y_vec / np.linalg.norm(y_vec)
        return origin, x_unit, y_unit

    def world_to_sample(self, points_xyz: np.ndarray) -> np.ndarray:
        """Vectorised: ``(..., 3) local xyz → (..., 3) (axis0, axis1, sample)``.

        Despite the name, ``points_xyz`` is in the Petrel grid-local
        frame (what COORD pillars contain), not UTM. The local→UTM
        rotation is folded in here.
        """

        arr = np.asarray(points_xyz, dtype=np.float64)
        if arr.shape[-1] != 3:
            raise ValueError(
                f"points_xyz must have last dim 3, got shape {arr.shape}"
            )
        origin, x_unit, y_unit = self._axis_units()
        local_x = arr[..., 0]
        local_y = arr[..., 1]
        z_m = arr[..., 2]
        world_x = origin[0] + local_x * x_unit[0] + local_y * y_unit[0]
        world_y = origin[1] + local_x * x_unit[1] + local_y * y_unit[1]
        out = np.empty(arr.shape, dtype=np.float32)
        out[..., 0] = (world_x - self.x_origin) / self.xy_step
        out[..., 1] = (world_y - self.y_origin) / self.xy_step
        out[..., 2] = z_m / self.z_step
        return out

    def sample_to_world(self, points_sample: np.ndarray) -> np.ndarray:
        """Inverse: ``(axis0, axis1, sample) → local (x_m, y_m, z_m)``.

        Returns coordinates in the Petrel grid-local frame so the
        result round-trips through ``world_to_sample`` without
        re-applying the MAPAXES rotation upstream.
        """

        arr = np.asarray(points_sample, dtype=np.float64)
        if arr.shape[-1] != 3:
            raise ValueError(
                f"points_sample must have last dim 3, got shape {arr.shape}"
            )
        origin, x_unit, y_unit = self._axis_units()
        world_x = arr[..., 0] * self.xy_step + self.x_origin
        world_y = arr[..., 1] * self.xy_step + self.y_origin
        # Invert the 2x2 rotation [x_unit y_unit] applied to (local_x, local_y).
        wx = world_x - origin[0]
        wy = world_y - origin[1]
        det = x_unit[0] * y_unit[1] - x_unit[1] * y_unit[0]
        local_x = ( y_unit[1] * wx - y_unit[0] * wy) / det
        local_y = (-x_unit[1] * wx + x_unit[0] * wy) / det
        out = np.empty(arr.shape, dtype=np.float32)
        out[..., 0] = local_x
        out[..., 1] = local_y
        out[..., 2] = arr[..., 2] * self.z_step
        return out
