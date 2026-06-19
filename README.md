# YJ Studio

YJ Studio is a PyQt6 desktop application backed by a FastAPI/SAM3 server and a
shared, UI-independent Python core.

## Repository Layout

```text
YJ_Studio_Portable/
  local/       desktop application and local launch/connection helpers
  server/      FastAPI service, SAM3 workers, deployment scripts and tests
  shared/      pure shared package used by both local and server code
  config/      versioned config templates and environment definitions
  data/        large source, intermediate and runtime datasets (not in Git)
  outputs/     reports, visualizations, videos and exported artifacts
  runtime/     disposable logs, caches, jobs, PID files and temporary files
  weights/     model assets (not in Git)
  libs/        vendored third-party source
  tools/       offline conversion, validation and diagnostic utilities
  tests/       repository-level integration tests
  docs/        architecture, operations, data notes and plans
  packaging/   packaging and distribution assets
  legacy/      historical material kept only for reference
```

The stable code boundaries are:

- `local/app/src/yj_studio`: Qt UI, interaction and visualization.
- `server/src/yj_studio_server`: API, inference, jobs and persistence.
- `shared/src/yj_studio_core`: models and scientific helpers without Qt or FastAPI.

## Runtime Reservoir Model

The active reservoir model is the cropped, node-aligned v3 NumPy pair:

```text
data/reservoir/npy_625x625x2_v3/lithology_binary_uint8.npy
data/reservoir/npy_625x625x2_v3/porosity_float16.npy
data/reservoir/npy_625x625x2_v3/metadata.json
```

Its spacing is `6.25 m x 6.25 m x 2 m`. GRDECL and older NumPy volumes are
offline source or historical intermediates, not the active runtime model.

## Start

Desktop:

```powershell
E:\miniconda\envs\py312\python.exe local\run_viewer.py
```

Server:

```bash
cd /root/quanbi
bash server/scripts/start_server.sh
```

See [STRUCTURE.md](STRUCTURE.md) for directory ownership rules and
[docs/deployment.md](docs/deployment.md) for deployment details.
