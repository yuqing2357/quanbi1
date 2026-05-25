"""End-to-end SAM3 video predictor smoke test.

Reproduces the exact call chain the workbench uses:
  1. Load grid (from cache).
  2. Render N ROI frames offscreen and dump as JPEGs.
  3. Build the SAM3 video predictor.
  4. init_state on the JPEG folder.
  5. add_prompt with a synthetic seed bbox.
  6. propagate_in_video forward + reverse.
  7. Print per-frame score, raise on failure.

The point of this script is to fail FAST when triton, autocast, the
predictor API, or any downstream piece is wrong, without taking the
30 s of GUI / SAM3 load every iteration.

Run via (cmd):
    set KMP_DUPLICATE_LIB_OK=TRUE
    E:\\miniconda\\envs\\py312\\python.exe tools\\smoke_sam3_video.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# --- Mirror run_yj_studio.py's Triton workarounds ----------------

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_TRITON_CACHE = Path("C:/yj_triton_cache")
_TRITON_CACHE.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(_TRITON_CACHE)
os.environ["TRITON_HOME"] = str(_TRITON_CACHE)
os.environ["TMP"] = str(_TRITON_CACHE)
os.environ["TEMP"] = str(_TRITON_CACHE)
tempfile.tempdir = str(_TRITON_CACHE)


def _install_triton_unc_workaround() -> None:
    import subprocess as _sp
    _orig = _sp.check_call

    def _strip(arg):
        if isinstance(arg, str):
            if arg.startswith("\\\\?\\"):
                return arg[4:]
            if len(arg) >= 6 and arg[0] == "-" and arg[1] in "IL" and arg[2:6] == "\\\\?\\":
                return arg[:2] + arg[6:]
        return arg

    def _patched(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)):
            cmd = type(cmd)(_strip(x) for x in cmd)
        return _orig(cmd, *a, **kw)

    _sp.check_call = _patched


_install_triton_unc_workaround()

# --- Paths --------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "apps" / "yj_studio" / "src"
SAM3_SRC = ROOT / "libs"
WEIGHTS = ROOT / "weights" / "sam3.pt"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SAM3_SRC) not in sys.path:
    sys.path.insert(0, str(SAM3_SRC))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log("Importing yj_studio.reservoir + sam3...")
    import numpy as np
    import torch
    from PIL import Image

    from yj_studio.reservoir import ReservoirGrid
    from yj_studio.reservoir.roi import default_roi
    from yj_studio.reservoir.sam3_render import render_roi_section

    log(f"torch {torch.__version__}, CUDA={torch.cuda.is_available()}")

    log("Loading reservoir grid (should be cache hit)...")
    grid = ReservoirGrid.load_from_master(Path(r"F:\１２３４.GRDECL"))
    log(f"  grid shape {grid.shape}, active {int(grid.active.sum()):,}")

    # Use a narrow centred ROI so this finishes quickly.
    big = default_roi(grid)
    il, ih, jl, jh, kl, kh = big
    def shrink(lo, hi, frac=0.3):
        mid = (lo + hi) // 2
        half = max(1, int((hi - lo) * frac / 2))
        return mid - half, mid + half
    j_lo, j_hi = shrink(jl, jh)
    k_lo, k_hi = shrink(kl, kh)
    roi = (il, ih, j_lo, j_hi, k_lo, k_hi)
    log(f"  ROI: {roi}")

    # Render a small range of frames around the centre of i.
    axis = "i"
    seed_idx = (il + ih) // 2
    n_each = 3    # tiny — we only want to verify the call works
    idx_lo = max(il, seed_idx - n_each)
    idx_hi = min(ih - 1, seed_idx + n_each)
    indices = list(range(idx_lo, idx_hi + 1))
    seed_frame_in_window = seed_idx - idx_lo
    log(f"  axis={axis} seed={seed_idx} frames={indices}")

    tempdir = Path(tempfile.mkdtemp(prefix="smoke_sam3_video_"))
    log(f"  rendering {len(indices)} frames to {tempdir}")
    frames = []
    for offset, idx in enumerate(indices):
        f = render_roi_section(grid, axis, idx, roi)
        frames.append(f)
        Image.fromarray(f.image).save(tempdir / f"{offset:05d}.jpg", quality=92)
    H, W = frames[0].image.shape[:2]
    log(f"  frame shape {W}x{H}")

    # Synthesise a seed bbox at the centre of the image. In real use
    # this comes from the user's prompt mask.
    cx_norm = 0.5
    cy_norm = 0.5
    bw_norm = 0.2
    bh_norm = 0.4
    seed_box_xywh = [cx_norm, cy_norm, bw_norm, bh_norm]
    log(f"  seed box (normalised xywh): {seed_box_xywh}")

    log("Checking GPU memory before SAM3 video load...")
    free_b, total_b = torch.cuda.mem_get_info()
    log(f"  GPU: {free_b/1024**3:.2f} GB free / {total_b/1024**3:.2f} GB total")
    if free_b < 5 * 1024**3:
        log("  !! Less than 5 GB free — SAM3 video may OOM. Continuing anyway.")

    log("Building SAM3 video predictor (this takes ~30s the first time)...")
    from sam3.model_builder import build_sam3_video_model
    t0 = time.time()
    try:
        predictor = build_sam3_video_model(
            checkpoint_path=str(WEIGHTS),
            device="cuda",
            strict_state_dict_loading=False,
        )
    except Exception as exc:
        log(f"  !! build_sam3_video_model threw: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 4
    log(f"  built in {time.time() - t0:.1f}s, type={type(predictor).__name__}")
    free_b, _ = torch.cuda.mem_get_info()
    log(f"  GPU free after load: {free_b/1024**3:.2f} GB")

    # Discover the actual propagation API. The class can have either
    # of these shapes depending on the SAM3 release; we probe.
    has_model_attr = hasattr(predictor, "model")
    has_propagate = hasattr(predictor, "propagate_in_video")
    has_model_propagate = has_model_attr and hasattr(predictor.model, "propagate_in_video")
    log(f"  has .model           : {has_model_attr}")
    log(f"  has .propagate_in_video      : {has_propagate}")
    log(f"  has .model.propagate_in_video: {has_model_propagate}")

    if has_propagate:
        prop_obj = predictor
    elif has_model_propagate:
        prop_obj = predictor.model
    else:
        log("  !! Neither path exists — check SAM3 version. Aborting.")
        return 3
    log(f"  using propagate from: {type(prop_obj).__name__}")

    log("Initialising video session...")
    t0 = time.time()
    session_state = predictor.init_state(
        resource_path=str(tempdir),
        async_loading_frames=False,
        video_loader_type="jpg",
    )
    log(f"  init_state OK in {time.time() - t0:.1f}s")

    log(f"Adding seed prompt at frame {seed_frame_in_window} (text='visual')...")
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
        predictor.add_prompt(
            inference_state=session_state,
            frame_idx=seed_frame_in_window,
            text_str="visual",
            points=None,
            point_labels=None,
            boxes_xywh=[seed_box_xywh],
            box_labels=[1],
            obj_id=1,
        )
        log(f"  add_prompt OK in {time.time() - t0:.1f}s")

        log("Propagating forward...")
        t0 = time.time()
        n_fwd = 0
        for frame_idx_local, outputs in prop_obj.propagate_in_video(
            inference_state=session_state,
            start_frame_idx=seed_frame_in_window,
            max_frame_num_to_track=n_each,
            reverse=False,
        ):
            n_fwd += 1
            keys = list(outputs.keys()) if isinstance(outputs, dict) else "<not-dict>"
            log(f"    frame {frame_idx_local}: outputs keys={keys}")
        log(f"  forward done: {n_fwd} frames in {time.time() - t0:.1f}s")

        log("Propagating reverse...")
        t0 = time.time()
        n_rev = 0
        for frame_idx_local, outputs in prop_obj.propagate_in_video(
            inference_state=session_state,
            start_frame_idx=seed_frame_in_window,
            max_frame_num_to_track=n_each,
            reverse=True,
        ):
            n_rev += 1
        log(f"  reverse done: {n_rev} frames in {time.time() - t0:.1f}s")

    log("")
    log(f"==== PASS — {n_fwd} forward + {n_rev} reverse frames tracked. ====")
    log(f"Use these in view_sam3_workbench:")
    log(f"  - call propagate_in_video on '{type(prop_obj).__name__}'")
    log(f"    (so: {'predictor.' + ('propagate_in_video' if has_propagate else 'model.propagate_in_video')})")
    log("  - wrap add_prompt + propagate in torch.autocast(bf16) + inference_mode")

    import shutil
    shutil.rmtree(tempdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
