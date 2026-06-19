"""Verify the (un-finalized) baked .partial volume by rendering inline axis0=479
from the FILE and comparing stats; render a 3-colour image to eyeball vs the
on-the-fly check (inline_479_baked.png). Does not finalize."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from project_paths import DIAGNOSTICS_ROOT

OUT = Path("/root/quanbi/data/reservoir/npy_625x625x2_v3")
IMG = DIAGNOSTICS_ROOT / "bake_check"
IMG.mkdir(parents=True, exist_ok=True)
meta = json.loads((OUT / "metadata.json").read_text())
N0, N1, NZ = meta["shape"]
org0 = meta["seismic_index_origin"]["axis0"]

lith = np.load(OUT / "lithology_binary_uint8.npy", mmap_mode="r")
poro = np.load(OUT / "porosity_float16.npy", mmap_mode="r")
print("shapes", lith.shape, poro.shape, "expected", (N0, N1, NZ))

o0 = (479 - org0) * 2  # output index for seismic axis0=479
L = np.asarray(lith[o0])                      # (N1, NZ)
P = np.isfinite(np.asarray(poro[o0]).astype(np.float32))
print(f"inline o0={o0} (axis0=479): present_frac={P.mean():.1%} target_frac={(L[P]==1).mean():.1%}")

rgb = np.full((NZ, N1, 3), 255, np.uint8)
Lt = L.T; Pt = P.T
rgb[Pt & (Lt == 0)] = (150, 150, 150)
rgb[Lt == 1] = (245, 214, 45)
s = max(1, int(max(rgb.shape[:2]) / 1400))
Image.fromarray(rgb[::s, ::s], "RGB").save(IMG / "inline_479_fromfile.png")

# global sanity on a strided subsample
ls = np.asarray(lith[::60, ::60, ::20])
ps = np.isfinite(np.asarray(poro[::60, ::60, ::20]).astype(np.float32))
print(f"global subsample: present_frac={ps.mean():.1%} target_frac(among present)={(ls[ps]==1).mean():.1%} "
      f"target_frac(all)={(ls==1).mean():.1%}")
print("DONE -> inline_479_fromfile.png")
