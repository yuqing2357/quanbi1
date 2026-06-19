"""Verify frame counts and readability of the 12.5 m reservoir videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    summary = json.loads((args.directory / "summary.json").read_text())
    results = {}
    for axis, spec in summary["axes"].items():
        path = args.directory / spec["video"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SystemExit(f"cannot open {path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        first_ok, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
        last_ok, _ = capture.read()
        capture.release()
        expected = int(spec["frame_count"])
        if frame_count != expected or not first_ok or not last_ok:
            raise SystemExit(
                f"{axis}: expected={expected}, actual={frame_count}, "
                f"first={first_ok}, last={last_ok}"
            )
        results[axis] = {
            "expected_frames": expected,
            "actual_frames": frame_count,
            "size": [width, height],
            "fps": fps,
            "first_frame_readable": first_ok,
            "last_frame_readable": last_ok,
            "bytes": path.stat().st_size,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
