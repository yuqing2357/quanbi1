from __future__ import annotations

from contextlib import nullcontext
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.sam3.engine import SAM3Engine  # noqa: E402


def test_track_video_reports_disabled_config() -> None:
    engine = SAM3Engine(Path("__missing_sam3.pt__"), load_video=False)
    engine._processor = object()

    with pytest.raises(RuntimeError, match="sam3.load_video=false"):
        list(
            engine.track_video(
                Path("."),
                seeds=[],
                seed_local=0,
                fwd_budget=1,
                back_budget=0,
            )
        )


def test_track_video_reports_video_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SAM3Engine(Path("__missing_sam3.pt__"), load_video=True)
    engine._processor = object()
    sam3_pkg = types.ModuleType("sam3")
    sam3_pkg.__path__ = []
    model_builder = types.ModuleType("sam3.model_builder")
    model_builder.build_sam3_video_model = object()
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)

    def fail_load(_builder) -> None:
        raise ImportError("No module named 'triton'")

    monkeypatch.setattr(engine, "_load_video_predictor", fail_load)

    with pytest.raises(RuntimeError, match="triton"):
        list(
            engine.track_video(
                Path("."),
                seeds=[],
                seed_local=0,
                fwd_budget=1,
                back_budget=0,
            )
        )

    assert engine.status_payload()["video_enabled"] is True
    assert engine.status_payload()["video_loaded"] is False
    assert "triton" in str(engine.status_payload()["video_error"])


def test_video_builder_disables_natural_video_temporal_disambiguation() -> None:
    captured: dict[str, object] = {}

    def builder(**kwargs):
        captured.update(kwargs)
        return object()

    engine = SAM3Engine(Path("checkpoint.pt"))
    engine._load_video_predictor(builder)

    assert captured["apply_temporal_disambiguation"] is False
    assert engine.status_payload()["video_temporal_disambiguation"] is False


def _box_mask(box, shape=(20, 20)) -> np.ndarray:
    """Boolean mask filled inside a normalised top-left ``[x,y,w,h]`` box."""
    height, width = shape
    x, y, bw, bh = box
    x0, y0 = int(round(x * width)), int(round(y * height))
    x1, y1 = int(round((x + bw) * width)), int(round((y + bh) * height))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_box_seed_uses_single_vg_prompt_and_propagates(monkeypatch) -> None:
    """A box seed is added as ONE visual-grounding prompt (no points / no
    fresh-state Tracker seed), and full propagation collects the object on the
    seed frame plus both neighbours."""

    box = [0.1, 0.1, 0.2, 0.2]

    class Predictor:
        def __init__(self) -> None:
            self.add_prompt_calls: list[dict] = []

        def add_prompt(self, **kwargs):  # noqa: ANN003
            self.add_prompt_calls.append(kwargs)
            return int(kwargs["frame_idx"]), {
                "out_obj_ids": [101],
                "out_binary_masks": np.stack([_box_mask(box)]),
            }

        def propagate_in_video(self, *, reverse, **kwargs):  # noqa: ANN003
            frame = 3 if not reverse else 1  # fwd=2 -> frame 3, back=1 -> frame 1
            yield frame, {
                "out_obj_ids": [101],
                "out_binary_masks": np.stack([_box_mask(box)]),
            }

    predictor = Predictor()
    engine = SAM3Engine(Path("__missing_sam3.pt__"), load_video=True)
    engine._track_state = {}
    monkeypatch.setattr(engine, "_ensure_video_predictor", lambda: predictor)
    monkeypatch.setattr(engine, "_autocast_ctx", nullcontext)
    monkeypatch.setattr(engine, "_inference_ctx", nullcontext)

    rows = list(
        engine.track_video(
            Path("."),
            seeds=[{"obj_id": 1, "box_xywh": box, "text": ""}],
            seed_local=2,
            fwd_budget=2,
            back_budget=1,
        )
    )

    # Exactly one VG prompt: box path, no points, no fresh-state obj_id.
    assert len(predictor.add_prompt_calls) == 1
    seed_call = predictor.add_prompt_calls[0]
    assert seed_call["points"] is None
    assert seed_call["boxes_xywh"] == [box]
    assert seed_call["obj_id"] is None

    # Seed frame plus both propagated neighbours are collected for the object.
    collected_frames = sorted(frame for frame, objects in rows if 1 in objects)
    assert collected_frames == [1, 2, 3]


def test_multi_box_seed_uses_one_prompt_and_maps_each_object(monkeypatch) -> None:
    """Multiple boxes go in a SINGLE prompt (sequential prompts would reset the
    state and collapse to one object), and each detection is mapped back to the
    seed whose box it overlaps."""

    box1 = [0.1, 0.1, 0.2, 0.2]
    box2 = [0.6, 0.6, 0.2, 0.2]

    class Predictor:
        def __init__(self) -> None:
            self.add_prompt_calls: list[dict] = []

        def add_prompt(self, **kwargs):  # noqa: ANN003
            self.add_prompt_calls.append(kwargs)
            # detector returns its own ids in arbitrary order
            return int(kwargs["frame_idx"]), {
                "out_obj_ids": [202, 101],
                "out_binary_masks": np.stack([_box_mask(box2), _box_mask(box1)]),
            }

        def propagate_in_video(self, *, reverse, **kwargs):  # noqa: ANN003
            if reverse:
                return
            yield 2, {
                "out_obj_ids": [202, 101],
                "out_binary_masks": np.stack([_box_mask(box2), _box_mask(box1)]),
            }

    predictor = Predictor()
    engine = SAM3Engine(Path("__missing_sam3.pt__"), load_video=True)
    engine._track_state = {}
    monkeypatch.setattr(engine, "_ensure_video_predictor", lambda: predictor)
    monkeypatch.setattr(engine, "_autocast_ctx", nullcontext)
    monkeypatch.setattr(engine, "_inference_ctx", nullcontext)

    rows = list(
        engine.track_video(
            Path("."),
            seeds=[
                {"obj_id": 1, "box_xywh": box1, "text": ""},
                {"obj_id": 2, "box_xywh": box2, "text": ""},
            ],
            seed_local=1,
            fwd_budget=1,
            back_budget=0,
        )
    )

    # one prompt carrying both boxes
    assert len(predictor.add_prompt_calls) == 1
    assert predictor.add_prompt_calls[0]["boxes_xywh"] == [box1, box2]

    # both seeds present on seed frame (1) and propagated frame (2)
    assert [frame for frame, _objects in rows] == [1, 2]
    assert [set(objects) for _frame, objects in rows] == [{1, 2}, {1, 2}]


def test_match_models_to_seeds_assigns_by_overlap() -> None:
    from yj_studio_server.sam3.engine import _match_models_to_seeds

    box1 = [0.1, 0.1, 0.2, 0.2]
    box2 = [0.6, 0.6, 0.2, 0.2]
    objects = {101: _box_mask(box1), 202: _box_mask(box2)}

    mapping = _match_models_to_seeds(objects, [box1, box2], [1, 2])

    assert mapping == {101: 1, 202: 2}


def _make_fake_cc_module(has_cc_torch: bool) -> types.ModuleType:
    mod = types.ModuleType("sam3.perflib.connected_components")
    mod.HAS_CC_TORCH = has_cc_torch

    def triton_cc(_tensor):  # simulates the broken Triton kernel
        raise RuntimeError("Triton Error [CUDA]: invalid argument")

    def cpu_cc(tensor):
        return ("cpu_labels", tensor)

    mod.connected_components = triton_cc
    mod.connected_components_cpu = cpu_cc
    return mod


def test_cc_fallback_forces_cpu_when_cc_torch_missing(monkeypatch) -> None:
    from yj_studio_server.sam3.engine import _install_cpu_connected_components_fallback

    sam3_pkg = types.ModuleType("sam3")
    sam3_pkg.__path__ = []
    perflib_pkg = types.ModuleType("sam3.perflib")
    perflib_pkg.__path__ = []
    cc_mod = _make_fake_cc_module(has_cc_torch=False)
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.perflib", perflib_pkg)
    monkeypatch.setitem(sys.modules, "sam3.perflib.connected_components", cc_mod)

    _install_cpu_connected_components_fallback()

    assert getattr(cc_mod, "_yj_forced_cpu_cc", False) is True

    class _FakeTensor:
        def dim(self) -> int:
            return 4

    fake = _FakeTensor()
    # the patched callable now routes to the CPU implementation instead of triton
    label, passed = cc_mod.connected_components(fake)
    assert label == "cpu_labels"
    assert passed is fake  # already 4D, forwarded unchanged


def test_cc_fallback_keeps_native_kernel_when_cc_torch_present(monkeypatch) -> None:
    from yj_studio_server.sam3.engine import _install_cpu_connected_components_fallback

    sam3_pkg = types.ModuleType("sam3")
    sam3_pkg.__path__ = []
    perflib_pkg = types.ModuleType("sam3.perflib")
    perflib_pkg.__path__ = []
    cc_mod = _make_fake_cc_module(has_cc_torch=True)
    original = cc_mod.connected_components
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.perflib", perflib_pkg)
    monkeypatch.setitem(sys.modules, "sam3.perflib.connected_components", cc_mod)

    _install_cpu_connected_components_fallback()

    assert getattr(cc_mod, "_yj_forced_cpu_cc", False) is False
    assert cc_mod.connected_components is original  # untouched fast path


def test_connected_components_cv2_matches_contract() -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from yj_studio_server.sam3.engine import _connected_components_cv2

    mask = torch.zeros((1, 1, 4, 4), dtype=torch.uint8)
    mask[0, 0, 0, 0] = 1  # singleton component (area 1)
    mask[0, 0, 2:4, 2:4] = 1  # 2x2 component (area 4)

    labels, counts = _connected_components_cv2(mask)
    assert labels.shape == mask.shape
    assert counts.shape == mask.shape
    counts_np = counts[0, 0].cpu().numpy()
    # background pixels have count 0; the two blobs carry their pixel areas
    assert counts_np[0, 0] == 1
    assert counts_np[2, 2] == 4 and counts_np[3, 3] == 4
    assert counts_np[1, 1] == 0  # background


def test_mask_norm_box_and_iou() -> None:
    from yj_studio_server.sam3.engine import _iou_xywh, _mask_norm_box

    box = [0.1, 0.1, 0.2, 0.2]
    derived = _mask_norm_box(_box_mask(box))
    assert _iou_xywh(box, derived) > 0.8  # mask bbox round-trips close to the box
    # disjoint boxes have zero overlap
    assert _iou_xywh([0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]) == 0.0
