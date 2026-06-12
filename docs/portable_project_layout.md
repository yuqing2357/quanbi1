# Portable Project Layout

The portable project keeps code, data, generated caches, and runtime output under
one project root so the application can run from a copied folder without relying
on workstation-specific drive letters.

```text
YJ_Studio_Portable/
  apps/
  docs/
  legacy/
  libs/
  server/
  local/
  tools/
  weights/
  runtime/
  data/
    seismic/
      YJ-ALL-SEISMIC_depth_0_653.npy
      YJ-ALL-SEISMIC.npy
      YJ-ALL-SEISMIC.segy
      processed/
    reservoir/
      grdecl/        # legacy offline source for regenerating numpy volumes
        １２３４.GRDECL
        １２３４_COORD.GRDECL
        １２３４_ZCORN.GRDECL
        １２３４_ACTNUM.GRDECL
        .yj_cache/
      numpy/         # 1x intermediate reservoir numpy
      numpy_3x/      # runtime reservoir numpy volumes
        lithology_binary_3x_uint8.npy
        porosity_3x_float16.npy
        metadata.json
  cache/
    triton/
```

Runtime defaults are resolved from `WORKSPACE_ROOT / "data"` in
`apps/yj_studio/src/yj_studio/config/paths.py`.

`server/` is for remote service code and deployment files. `local/` is for
local development and remote-connection helpers. Generated logs, cache, and job
state should go under `runtime/` rather than inside source folders.
