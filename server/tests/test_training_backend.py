from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SRC = _REPO_ROOT / "server" / "src"
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))

from yj_studio_server.sam3.training import run_training_backend  # noqa: E402
from yj_studio_server.sam3.models import ModelRegistry  # noqa: E402


def test_run_training_backend_collects_metrics_and_checkpoint(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "run"
    dataset_dir.mkdir()
    script = tmp_path / "fake_train.py"
    script.write_text(
        "\n".join(
            [
                "import json, os, pathlib",
                "dataset = pathlib.Path(os.environ['YJ_DATASET_DIR'])",
                "out = pathlib.Path(os.environ['YJ_TRAIN_OUTPUT_DIR'])",
                "assert dataset.exists()",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'best.pt').write_text('weights', encoding='utf-8')",
                "(out / 'metrics.json').write_text(json.dumps({'dice': 0.91, 'checkpoint': 'best.pt'}), encoding='utf-8')",
                "print('ok')",
            ]
        ),
        encoding="utf-8",
    )

    result = run_training_backend(
        [sys.executable, "-B", str(script)],
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )

    assert result["checkpoint"] == str(output_dir / "best.pt")
    assert result["metrics"]["dice"] == 0.91
    assert result["stdout_tail"].strip() == "ok"


def test_model_registry_records_parent_model_id() -> None:
    registry = ModelRegistry(_REPO_ROOT / "runtime" / "local" / "tmp" / f"model-registry-{uuid4().hex}")

    base = registry.add_model(checkpoint="base.pt", dataset_version="d0")
    child = registry.add_model(checkpoint="child.pt", dataset_version="d1", metrics={"dice": 0.9})
    payload = registry.load()

    assert payload["active_model"] == base["id"]
    assert base["parent_model_id"] is None
    assert child["parent_model_id"] == base["id"]
    assert payload["models"][1]["metrics"]["dice"] == 0.9
