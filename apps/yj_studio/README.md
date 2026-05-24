# YJ Studio

YJ Studio is the new desktop software implementation for the YJ seismic interpretation
workflow. The old project under `D:\商书记项目` is treated as read-only reference material;
all new implementation lives in this workspace.

Current milestone: Phase 0 scaffolding plus a small Phase 1 data-layer foundation.

Run from this folder with:

```powershell
E:\miniconda\envs\py312\python.exe -m yj_studio
```

Run tests with bytecode writes disabled in this Windows workspace:

```powershell
E:\miniconda\envs\py312\python.exe -B -m pytest
```
