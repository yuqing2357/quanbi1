from __future__ import annotations

from io import StringIO

import numpy as np

from yj_studio.io.readers.well_logs import load_depth_samples, load_log_samples


class FakeLogPath:
    def exists(self) -> bool:
        return True

    def open(self, *_args, **_kwargs):
        return StringIO(
            "DEPT,por\n"
            "1490.0,0.1\n"
            "1500.0,0.2\n"
            "1510.0,0.3\n"
            "99999.0,0.4\n"
        )


def test_load_depth_samples_converts_depth_to_sample() -> None:
    samples = load_depth_samples(FakeLogPath(), z_count=654, z_window_start=0.0)  # type: ignore[arg-type]

    assert samples.tolist() == [149.0, 150.0, 151.0]


def test_load_log_samples_keeps_position_and_value() -> None:
    log = load_log_samples(
        FakeLogPath(),  # type: ignore[arg-type]
        inline_index=12.0,
        xline_index=34.0,
        value_column="por",
        z_count=654,
        z_window_start=0.0,
    )

    np.testing.assert_allclose(
        log.samples,
        np.asarray(
            [
                [12.0, 34.0, 149.0, 0.1],
                [12.0, 34.0, 150.0, 0.2],
                [12.0, 34.0, 151.0, 0.3],
            ],
            dtype=np.float32,
        ),
    )
