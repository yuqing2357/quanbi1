# YJ Studio Deployment Notes

## Server

Current server:

```text
host: 114.214.170.109
ssh_port: 2401
user: root
project_root: /root/quanbi
conda_env: yjstudio-server
```

Activate:

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate yjstudio-server
cd /root/quanbi
```

Validate:

```bash
python server/scripts/validate_data.py
bash server/scripts/run_tests.sh
```

Prepare config:

```bash
cp server/config/server.example.yaml server/config/server.yaml
```

Start manually:

```bash
bash server/scripts/start_server.sh
```

Healthcheck:

```bash
bash server/scripts/healthcheck.sh
```

## Local

Use local helpers from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File local\scripts\run_desktop.ps1
```

After the server service is running:

```powershell
powershell -ExecutionPolicy Bypass -File local\scripts\test_remote_connection.ps1 `
  -ServerUrl http://114.214.170.109:8765
```

## Data

Data stays outside `server/` and `local/`:

```text
data/reservoir/numpy_3x/lithology_binary_3x_uint8.npy
data/reservoir/numpy_3x/porosity_3x_float16.npy
data/seismic/YJ-ALL-SEISMIC.npy
weights/sam3.pt
```

The uploaded server copy was checked by file count and file size:

```text
local files: 5820
remote files: 5820
missing: 0
extra: 0
size mismatch: 0
```
