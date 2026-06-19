# Diagnostic Helpers

This directory contains ad-hoc, read-only investigation scripts retained for
provenance. They are not part of the desktop or server runtime.

- `check_encoding.py`: inspect historical reservoir class/porosity encoding.
- `inspect_3x_direct.py`: inspect historical direct-3x and repeat-3x outputs.

New maintained diagnostics should use `tools/project_paths.py` and write
reviewable artifacts under `outputs/diagnostics/`.
