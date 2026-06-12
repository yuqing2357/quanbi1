# YJ Studio Portable

YJ Studio Portable keeps the desktop application, data, server runtime, and local
debug helpers in one copyable project folder. The current project split is:

```text
YJ_Studio_Portable/
  apps/        # desktop application and shared UI/runtime code
  libs/        # vendored or third-party source dependencies
  data/        # seismic, reservoir, generated numpy volumes, and results
  weights/     # model weights
  server/      # remote server service, config examples, deployment scripts
  local/       # local development, debugging, and remote-connection helpers
  tools/       # one-off conversion, validation, and smoke-test utilities
  docs/        # architecture notes, deployment notes, figures, plans
  runtime/     # generated logs/cache/jobs; ignored by git
  cache/       # generated caches; ignored by git
  legacy/      # historical project material kept for reference
```

## What To Edit

- Desktop app behavior: `apps/yj_studio/`
- Remote API/service behavior: `server/`
- Local helper scripts: `local/`
- Large data files: `data/`
- Model weights: `weights/`
- Historical conversion or smoke-test scripts: `tools/`

Do not put large data under `server/` or `local/`. Those folders should stay
small enough to upload or sync when only the runtime code changes.

## Current Reservoir Data Path

Runtime reservoir volumes are numpy files:

```text
data/reservoir/numpy_3x/lithology_binary_3x_uint8.npy
data/reservoir/numpy_3x/porosity_3x_float16.npy
data/reservoir/numpy_3x/metadata.json
```

The old GRDECL files are kept as source material for regeneration:

```text
data/reservoir/grdecl/
```

## Useful Docs

- Project layout: `docs/portable_project_layout.md`
- Cleanup plan: `docs/project_cleanup_plan.md`
- Remote server architecture: `docs/remote_server_architecture.md`
- Deployment notes: `docs/deployment.md`
- Tool index: `tools/README.md`
- Figure index: `docs/figures/README.md`
