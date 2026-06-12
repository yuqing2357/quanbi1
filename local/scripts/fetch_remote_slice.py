from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Small local remote-server probe.")
    parser.add_argument("--server-url", default=_default_server_url())
    parser.add_argument("--volume-id", default="seismic")
    parser.add_argument("--axis", choices=("inline", "xline", "z"), default="z")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()

    base = args.server_url.rstrip("/")
    with urlopen(f"{base}/health", timeout=10) as response:
        health = json.load(response)
    print(json.dumps({"health": health}, ensure_ascii=False, indent=2))

    with urlopen(f"{base}/volumes", timeout=10) as response:
        volumes = json.load(response)
    volume = next((item for item in volumes if item.get("id") == args.volume_id), None)
    if volume is None:
        raise SystemExit(f"Volume not found: {args.volume_id}")
    shape = tuple(int(v) for v in volume["shape"])
    axis_pos = {"inline": 0, "xline": 1, "z": 2}[args.axis]
    index = args.index if args.index is not None else shape[axis_pos] // 2
    query = urlencode({"volume_id": args.volume_id, "axis": args.axis, "index": index})
    with urlopen(f"{base}/slice?{query}", timeout=30) as response:
        data = response.read()
    arr = np.load(BytesIO(data), allow_pickle=False)
    print(
        json.dumps(
            {
                "slice": {
                    "volume_id": args.volume_id,
                    "axis": args.axis,
                    "index": index,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "bytes_received": len(data),
                    "finite_ratio": float(np.isfinite(arr).mean()),
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _default_server_url() -> str:
    for path in (ROOT / "local" / "config" / "local.yaml", ROOT / "local" / "config" / "local.example.yaml"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line.startswith("server_url:"):
                return line.split(":", 1)[1].strip()
    return "http://114.214.170.109:8765"


if __name__ == "__main__":
    raise SystemExit(main())
