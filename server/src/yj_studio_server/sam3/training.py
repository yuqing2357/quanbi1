from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def run_training_backend(
    command: str | list[str],
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    timeout_s: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run an optional external training command and collect its artifacts.

    The command may contain ``{dataset_dir}`` and ``{output_dir}``
    placeholders. The process also receives ``YJ_DATASET_DIR`` and
    ``YJ_TRAIN_OUTPUT_DIR`` environment variables. If it writes
    ``metrics.json`` in the output directory, those metrics are captured.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    args = _command_args(command, dataset_path, output_path)
    env = os.environ.copy()
    env.update(
        {
            "YJ_DATASET_DIR": str(dataset_path),
            "YJ_TRAIN_OUTPUT_DIR": str(output_path),
        }
    )
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})

    completed = subprocess.run(
        args,
        cwd=str(output_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Training command failed with exit code {completed.returncode}: "
            f"{_tail(completed.stderr) or _tail(completed.stdout)}"
        )

    metrics = _read_metrics(output_path)
    checkpoint = _resolve_checkpoint(output_path, metrics)
    return {
        "command": args,
        "output_dir": str(output_path),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "metrics": metrics,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _command_args(command: str | list[str], dataset_dir: Path, output_dir: Path) -> list[str]:
    if isinstance(command, str):
        args = shlex.split(command)
    else:
        args = [str(item) for item in command]
    if not args:
        raise ValueError("training command is empty")
    values = {"dataset_dir": str(dataset_dir), "output_dir": str(output_dir)}
    return [part.format(**values) for part in args]


def _read_metrics(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_checkpoint(output_dir: Path, metrics: dict[str, Any]) -> Path | None:
    raw = metrics.get("checkpoint") or metrics.get("checkpoint_path")
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = output_dir / path
        if path.exists():
            return path
    for name in ("checkpoint.pt", "best.pt", "latest.pt", "model.pt", "checkpoint.pth", "best.ckpt"):
        path = output_dir / name
        if path.exists():
            return path
    candidates = [
        path
        for suffix in ("*.pt", "*.pth", "*.ckpt")
        for path in output_dir.glob(suffix)
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _tail(text: str | None, *, max_chars: int = 2000) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]
