# Project Cleanup Plan

This file records the intended organization so future cleanup does not break
paths that are already used by the desktop app, server scripts, or docs.

## Stable Top-Level Boundaries

```text
apps/      shared desktop application code
libs/      vendored/shared dependencies
data/      large seismic, reservoir, result, and generated numpy data
weights/   model weights
server/    remote service code, server config examples, deployment scripts
local/     local launch/debug/test helpers
tools/     conversion, validation, and smoke-test utilities
docs/      plans, architecture notes, deployment notes, and figures
runtime/   generated logs/cache/jobs; not committed
cache/     generated caches; not committed
legacy/    old project material kept for reference
```

## What Has Been Organized

- Server runtime files were collected under `server/`.
- Local development and remote-connection helpers were collected under `local/`.
- Runtime output locations were separated into `runtime/` and ignored by git.
- Reservoir runtime data now points to numpy volumes under
  `data/reservoir/numpy_3x/`.
- Documentation figures were moved from the root of `docs/` into
  `docs/figures/reservoir/` and `docs/figures/legacy/`.
- Tool scripts remain in `tools/` and are documented in `tools/README.md`.

## Current Hold Points

Do not move these yet:

- `tools/*.py`: several docs, comments, and smoke-test scripts reference these
  paths directly.
- `apps/yj_studio/src/yj_studio/tools/`: this is application interaction-tool
  code, not the same thing as top-level preprocessing scripts.
- `data/`: large files are already isolated and should not be renamed casually.
- `weights/`: model paths may be used by SAM3 setup checks.

## Later Cleanup, If Needed

When the project is stable, the top-level `tools/` folder can be split into
subfolders with compatibility wrappers:

```text
tools/
  reservoir/
  sam3/
  diagnostics/
  project_paths.py
```

That change should be done together with path updates in docs, server tests, and
any app comments that mention the old script names.

## Server Sync Rule

For normal development:

- Change server service behavior: sync `server/`.
- Change desktop behavior: sync `apps/`.
- Change local debug helpers: keep in `local/`, sync only if needed.
- Change data: sync `data/` deliberately, usually not as part of code updates.
- Never sync `runtime/` as source of truth.
