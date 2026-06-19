# Project Cleanup Plan

## Completed Boundaries

- Desktop code lives under `local/app/`.
- Server code lives under `server/`.
- Shared models and scientific helpers live under `shared/`.
- Live configuration is separated from committed templates in `config/`.
- Large data, weights, retained outputs and disposable runtime files are
  separate top-level concerns.
- The active reservoir runtime model is `npy_625x625x2_v3`.

## Cleanup Rules

1. Establish a tested Git baseline before structural moves.
2. Trace imports, documentation and deployment references before moving code.
3. Keep `data/`, `weights/` and `libs/` stable unless a dedicated migration is
   justified.
4. Put retained visual evidence under `outputs/`.
5. Put logs, caches, jobs and temporary files under `runtime/`.
6. Keep package tests beside their package; use root `tests/` only for
   cross-package integration tests.
7. Archive superseded documentation instead of leaving several files claiming
   to be authoritative.

## Remaining Optional Work

- Split the flat `tools/` directory into `reservoir/`, `validation/`,
  `diagnostics/` and `visualization/` after all script references are traced.
- Replace remaining hard-coded `/root/quanbi` tool paths with
  `tools/project_paths.py`.
- Classify historical `numpy_3x*` data and remove it only after explicit
  checksum and rollback approval.
- Move `legacy/` outside the active repository once its historical value has
  been reviewed.

The authoritative current layout is documented in the root `STRUCTURE.md`.
