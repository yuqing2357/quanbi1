# YJ Studio Local

This folder contains local-only development, debugging, and client-side helper
files. It is not the server deployment surface.

## Scope

- Local desktop launch helpers.
- Local remote-connection tests.
- Local cache and log locations.
- Local development environment notes.

## Non-Scope

- Server service scripts and systemd files.
- Large project data.
- Server runtime logs and task state.

## Typical Local Flow

Run from VSCode:

1. Open `local/run_viewer.py`.
2. Choose the local Python interpreter, usually `py312`.
3. Click Run Python File, or use the launch target `YJ Studio: Run Viewer`.

The server address and local launch behavior are configured in:

```text
local/config/local.yaml
```

When `volume_backend: remote` is enabled, the viewer discovers volumes from the
server and requests only the current 2D slice over HTTP. It should not open the
local multi-GB seismic or reservoir numpy files for normal viewing.

When `sam3_backend: remote` is enabled, the AI dock submits SAM3 jobs to the
server through `/sam3/jobs` instead of loading the local GPU model. If the key
is omitted and `mode: remote` is set, `local/run_viewer.py` also selects the
remote SAM3 backend by default.

Command-line equivalent:

```powershell
cd G:\YJ_Studio_Portable
powershell -ExecutionPolicy Bypass -File local\scripts\run_desktop.ps1
```

or:

```powershell
E:\miniconda\envs\py312\python.exe local\run_viewer.py
```

Check a remote server:

```powershell
powershell -ExecutionPolicy Bypass -File local\scripts\test_remote_connection.ps1
```

Quick config-only check:

```powershell
E:\miniconda\envs\py312\python.exe local\run_viewer.py --check-only
```

Fetch one remote slice without opening the GUI:

```powershell
E:\miniconda\envs\py312\python.exe local\scripts\fetch_remote_slice.py `
  --volume-id model_porosity --axis z
```
