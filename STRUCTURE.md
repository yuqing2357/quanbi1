# Project Structure

This file is the authoritative directory-ownership reference for YJ Studio.

## Stable Code Boundaries

| Path | Ownership | Git |
|---|---|---|
| `local/app/` | PyQt6 desktop package, desktop tests | yes |
| `local/scripts/` | local connection and inspection helpers | yes |
| `server/` | FastAPI, SAM3, jobs, persistence, deployment and server tests | yes |
| `shared/` | pure shared models/scientific helpers; no Qt or FastAPI | yes |
| `tools/` | offline conversion, validation and diagnostics | yes |
| `tests/` | cross-package integration tests only | yes |

Dependency direction:

```text
local  ─┐
        ├──> shared
server ─┘
```

`local` must not import `server`. `shared` must not import Qt, FastAPI or either
application package.

## Configuration

```text
config/
  local.example.yaml
  server.example.yaml
  local.yaml              # machine-specific, ignored
  server.yaml             # machine-specific, ignored
  env/                    # conda and pip environment definitions
```

Only templates and environment definitions are committed. Secrets, host paths
and live server connection settings stay out of Git.

## Data And Generated Files

| Path | Content | Lifetime |
|---|---|---|
| `data/` | source, intermediate and active scientific datasets | persistent |
| `data/results/` | authoritative server targets/training artifacts | persistent |
| `weights/` | downloaded/pretrained model assets | persistent |
| `outputs/` | reports, visualizations, videos and exports | reproducible delivery |
| `runtime/` | logs, cache, jobs, PID and temporary files | disposable |
| `cache/` | shared tool/library cache such as Triton | disposable |

`outputs/` and `runtime/` are different: an output is intentionally retained
for review or delivery; runtime state may be deleted whenever no process is
using it.

Recommended generated layout:

```text
outputs/
  reports/
  diagnostics/
  visualizations/
  videos/
  exports/

runtime/
  local/
    logs/
    cache/
    tmp/
  server/
    logs/
    cache/
    jobs/
```

## Data Lifecycle

The current data paths remain stable because they contain more than 100 GB and
are referenced by deployed configuration. Their semantic roles are:

```text
data/seismic/                         seismic source and processed volumes
data/reservoir/grdecl/                original offline GRDECL source
data/reservoir/numpy/                 native/intermediate NumPy arrays
data/reservoir/numpy_3x*/             historical derived volumes
data/reservoir/npy_625x625x2_v3/      active runtime reservoir volume
data/results/                         server-owned persistent results
```

Do not move large data merely for cosmetic consistency. A future data migration
must update configuration, verify checksums, test both runtimes and only then
retire old paths.

## Tooling Rule

Top-level `tools/` contains repository utilities, while
`local/app/src/yj_studio/tools/` contains interactive application tools. They
are unrelated. New offline scripts should use `tools/project_paths.py` and
write retained artifacts under `outputs/`, never directly under `runtime/`.

## Root Directory Rule

The root should contain source directories and a small set of project metadata
files only. Do not create root-level pytest directories, reports, screenshots,
downloaded slices or connection secrets. `.gitignore` enforces these rules.
