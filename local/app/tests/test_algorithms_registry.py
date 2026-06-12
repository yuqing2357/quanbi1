from __future__ import annotations

from yj_studio.algorithms import builtin as _builtin  # noqa: F401 — register side effect
from yj_studio.algorithms.registry import registry


EXPECTED_IDS = {
    # builtin
    "horizon.thickness",
    "measure.distance",
    "measure.area",
    "volume.connectivity",
    "reservoir.sandbody_extract",
    "horizon.closure_contour",
    "trap.detect_structural",
    "trap.evaluate",
    # stubs
    "horizon.autotrack",
    "horizon.autotrack_3d",
    "fault.autopick",
    "mask.region_grow",
}


def test_registry_lists_all_expected_algorithms() -> None:
    ids = {cls.id for cls in registry.iter_algorithms()}
    missing = EXPECTED_IDS - ids
    assert not missing, f"Missing algorithms: {missing}"


def test_phase_two_stubs_fail_with_friendly_message() -> None:
    from yj_studio.algorithms import AlgorithmRunner

    runner = AlgorithmRunner()
    stub_cls = registry.get("fault.autopick")
    result = runner.run_sync(
        stub_cls,
        params={},
        input_layers={},
    )
    assert not result.ok
    assert "Phase-2 stub" in (result.error or "")
