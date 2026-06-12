from __future__ import annotations

from dataclasses import dataclass

from yj_studio.config.defaults import DEFAULT_Z_WINDOW_START, DEPTH_STEP_TO_SAMPLE


@dataclass(frozen=True, slots=True)
class CoordTransform:
    """Coordinate conversion for the fixed YJ seismic grid."""

    z_window_start: float = DEFAULT_Z_WINDOW_START
    depth_step_to_sample: float = DEPTH_STEP_TO_SAMPLE
    inline_origin: float = 0.0
    xline_origin: float = 0.0

    def depth_m_to_sample(self, depth_m: float) -> float:
        return float(depth_m) / self.depth_step_to_sample - self.z_window_start

    def sample_to_depth_m(self, sample: float) -> float:
        return (float(sample) + self.z_window_start) * self.depth_step_to_sample

    def ijk_to_inline_xline(self, i: float, j: float, k: float) -> tuple[float, float, float]:
        return (float(i) + self.inline_origin, float(j) + self.xline_origin, float(k))

    def inline_xline_to_ijk(
        self,
        inline: float,
        xline: float,
        sample: float,
    ) -> tuple[float, float, float]:
        return (float(inline) - self.inline_origin, float(xline) - self.xline_origin, float(sample))

