# Tool Script Index

The top-level `tools/` folder contains project utilities, not application UI
tools. Application interaction tools live in `local/app/src/yj_studio/tools/`.

Keep these files at their current paths until all references in docs, tests, and
comments are updated.

## Shared Path Helper

- `project_paths.py`: common project-root and data-path resolution for scripts.

## Reservoir Conversion And Checks

- `create_reservoir_3x_numpy.py`: convert reservoir GRDECL-derived data into 3x
  runtime numpy volumes.
- `calibrate_reservoir_transform.py`: calibrate reservoir-to-seismic alignment.
- `probe_grdecl.py`: inspect GRDECL content and dimensions.
- `verify_grdecl_parser.py`: validate GRDECL parser behavior.
- `inspect_active_bbox.py`: inspect active reservoir cell bounds.
- `check_active_depth_at_i.py`: inspect active cell depth at a selected index.
- `smoke_reservoir_grid.py`: quick reservoir grid smoke test.
- `smoke_reservoir_sections.py`: quick reservoir section visualization smoke
  test.
- `smoke_downsample.py`: quick downsampling check.

## SAM3 Setup And Smoke Tests

- `check_sam3_setup.py`: verify SAM3 assets and import readiness.
- `check_sam3_video_deps.py`: verify SAM3 video dependencies.
- `copy_sam3_assets.ps1`: copy SAM3 assets into the expected local locations.
- `smoke_sam3_render.py`: quick render smoke test.
- `smoke_sam3_video.py`: quick SAM3 video predictor smoke test.
- `smoke_triton_compile.py`: quick Triton compile/cache check.

## Generated Output

- `_sam3_smoke_out/`: generated SAM3 smoke-test output; ignored by git.
- `__pycache__/`: Python bytecode cache; ignored by git.
