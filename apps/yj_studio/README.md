# YJ Studio

YJ Studio is the new desktop software implementation for the YJ seismic interpretation
workflow. Code, bundled libraries, and project data are expected to live inside this
portable workspace.

Current milestone: Phase 0 scaffolding plus a small Phase 1 data-layer foundation.

Run from this folder with:

```powershell
conda activate py312
python -m yj_studio
```

Run tests with bytecode writes disabled in this Windows workspace:

```powershell
conda activate py312
python -B -m pytest
```
