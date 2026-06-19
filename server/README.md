# YJ Studio Server

This folder contains server-side deployment code and operational files only.
It is meant to be copied or synced to the remote machine together with the
shared source folders it depends on.

## Scope

- FastAPI service entry points.
- Server configuration examples.
- Server startup, validation, and healthcheck scripts.
- Server dependency notes.
- Foreground terminal runtime and diagnostics.

## Non-Scope

- Large data files. Keep them under `data/`.
- Desktop UI launch scripts. Keep them under `local/` or the project root.
- Generated task output. Keep it under `runtime/server/` or `data/results/`.

## Expected Server Layout

```text
/root/quanbi/
  local/app/
  shared/
  libs/
  data/
  server/
  runtime/server/
```

## Current Server Environment

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate yjstudio-server
```

## Run From VSCode

Open `server/run_server.py` and run it with the server interpreter
`yjstudio-server`, or use the launch target `YJ Studio: Run Server`.

Server-side parameters live in one file:

```text
server/config/server.yaml
```

Quick config-only check:

```bash
/root/anaconda3/envs/yjstudio-server/bin/python server/run_server.py --check-only
```

Main API endpoints:

```text
GET /health
GET /volumes
GET /slice?volume_id=seismic&axis=z&index=600
POST /sam3/jobs
GET /sam3/jobs/{job_id}
GET /sam3/jobs/{job_id}/result
GET /sam3/jobs/{job_id}/mask/{candidate_index}
POST /sam3/jobs/{job_id}/cancel
```

Run validation:

```bash
cd /root/quanbi
bash server/scripts/run_tests.sh
python server/scripts/validate_data.py
```

Start the service after `server/config/server.yaml` is created:

```bash
cd /root/quanbi
bash server/scripts/start_server.sh
```

The process stays attached to the current terminal. Uvicorn status, request
logs, SAM3 job state changes, and exception tracebacks are printed there in
real time. Press `Ctrl+C` in the same terminal to stop it.

Legacy compatibility entry:

```bash
cd /root/quanbi
bash server/scripts/start_background.sh
```

This command now delegates to `start_server.sh` and also runs in the foreground.
It no longer starts `nohup` or redirects output to `runtime/server/logs`.

Clean up an old background process from earlier versions:

```bash
cd /root/quanbi
bash server/scripts/stop_server.sh
```
