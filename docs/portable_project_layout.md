# Portable Project Layout

The portable workspace keeps code and optional runtime assets under one root,
but separates them by ownership and lifetime:

```text
YJ_Studio_Portable/
  local/       desktop runtime
  server/      remote service runtime
  shared/      common pure Python core
  config/      config templates and environment definitions
  data/        persistent scientific data
  weights/     model assets
  outputs/     retained reports, images, videos and exports
  runtime/     disposable process state
  cache/       disposable shared caches
  libs/        vendored dependencies
  tools/       offline utilities
  docs/        documentation
```

The active reservoir volume is
`data/reservoir/npy_625x625x2_v3/`. Its metadata defines the cropped seismic
origin, scale and `6.25 x 6.25 x 2 m` spacing.

Generated diagnostics belong under `outputs/diagnostics/`. Logs, job state,
slice caches, PID files and temporary test files belong under `runtime/`.

Large data paths are intentionally not renamed during ordinary code cleanup.
Moving them requires a dedicated migration with configuration updates and
checksum verification.
