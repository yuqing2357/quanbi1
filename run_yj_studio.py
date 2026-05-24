from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_SRC = ROOT / "apps" / "yj_studio" / "src"


def main() -> int:
    """Run YJ Studio from the repository root without manual PYTHONPATH setup."""

    sys.dont_write_bytecode = True
    src_text = str(PROJECT_SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

    from yj_studio.app import run

    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

 