#!/usr/bin/env python
"""Stage configured volumes into the RAM-backed stage dir (preload.stage_dir).

One-time copy from the (slow) data disk into a tmpfs RAM disk such as
``/dev/shm``.  After staging, start the server with ``preload.stage_dir`` set
and ``volumes_to_ram: false``: the server mmaps the RAM-disk copies, so startup
is instant and every slice is served at RAM speed — and it survives server
restarts (the RAM disk is only cleared on a full machine reboot).

Usage (on the server, in the yjstudio-server env):

    python server/scripts/stage_data_to_shm.py            # stage all volumes
    python server/scripts/stage_data_to_shm.py seismic     # stage specific ids
    python server/scripts/stage_data_to_shm.py --clear     # remove the stage dir

The destination mirrors each volume's data_root-relative path under stage_dir,
which is exactly what VolumeCache looks for.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("YJ_STUDIO_ROOT", Path(__file__).resolve().parents[2]))
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.config import load_config  # noqa: E402


def _stage_dir(cfg) -> Path:
    return Path(dict(cfg.preload).get("stage_dir") or "/dev/shm/yj_studio")


def main(argv: list[str]) -> int:
    cfg = load_config()
    stage_dir = _stage_dir(cfg)

    if "--clear" in argv:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
            print(f"cleared {stage_dir}")
        else:
            print(f"nothing to clear: {stage_dir} does not exist")
        return 0

    wanted = [a for a in argv if not a.startswith("-")]
    volume_ids = wanted or list(cfg.volumes)

    total_gb = 0.0
    for volume_id in volume_ids:
        spec = cfg.volumes.get(volume_id)
        if spec is None:
            print(f"skip {volume_id}: not in config")
            continue
        rel = str(spec.get("path", ""))
        src = cfg.data_root / rel
        dst = stage_dir / rel
        if not src.exists():
            print(f"skip {volume_id}: missing source {src}")
            continue
        size_gb = src.stat().st_size / 2**30
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"ok   {volume_id}: already staged ({size_gb:.1f} GB) -> {dst}")
            total_gb += size_gb
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"copy {volume_id}: {src} -> {dst} ({size_gb:.1f} GB) ...", flush=True)
        start = time.time()
        shutil.copy2(src, dst)
        elapsed = time.time() - start
        rate = size_gb / elapsed if elapsed else 0.0
        print(f"done {volume_id}: {size_gb:.1f} GB in {elapsed:.0f}s ({rate:.2f} GB/s)", flush=True)
        total_gb += size_gb

    print(f"stage dir: {stage_dir} ({total_gb:.1f} GB staged)")
    print("Now restart the server (preload.stage_dir set, volumes_to_ram: false).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
