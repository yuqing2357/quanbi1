from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parent
VISUAL_ROOT = CODE_DIR.parent
BUNDLE_ROOT = VISUAL_ROOT.parent
PROCESSED_ROOT = BUNDLE_ROOT / "处理后文件"
FULL_DEPTH_SEISMIC_NPY = Path(r"F:\YJ-ALL-SEISMIC_depth_0_653.npy")
FULL_DEPTH_PROCESSED_ROOT = Path(r"F:\YJ-ALL-SEISMIC_depth_0_653_processed")
LITH_POR_MODEL_ROOT = Path(r"F:\YJ-LITH-POR_model_numpy")


def ensure_inside_bundle(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(BUNDLE_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{label} must stay inside bundle: {resolved}") from exc
    return resolved


def resolve_path(path: Path, _label: str) -> Path:
    return path.resolve()


def _prefer_local_cigvis() -> None:
    cigvis_dir = VISUAL_ROOT / "cigvis"
    if not cigvis_dir.is_dir():
        raise SystemExit(f"Missing bundled cigvis directory: {cigvis_dir}")
    root = str(VISUAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_prefer_local_cigvis()

from cigvis import viserplot
from cigvis.visernodes import SurfaceNode
from matplotlib.colors import ListedColormap
from well_section import attach_well_section_gui

Z_WINDOW_START = 0.0
DEPTH_STEP_TO_SAMPLE = 10.0
POR_COLUMN = "POR_shalizhuojiyanxiangkong-20221217-fupinbi-chouxi"

PALETTE = [
    (0.894, 0.102, 0.110, 0.88),
    (0.216, 0.494, 0.722, 0.88),
    (0.302, 0.686, 0.290, 0.88),
    (0.596, 0.306, 0.639, 0.88),
    (1.000, 0.498, 0.000, 0.88),
    (1.000, 1.000, 0.200, 0.88),
    (0.651, 0.337, 0.157, 0.88),
    (0.969, 0.506, 0.749, 0.88),
    (0.600, 0.600, 0.600, 0.88),
    (0.121, 0.466, 0.705, 0.88),
    (0.682, 0.780, 0.909, 0.88),
    (1.000, 0.733, 0.470, 0.88),
    (0.173, 0.627, 0.173, 0.88),
    (0.839, 0.153, 0.157, 0.88),
    (0.580, 0.404, 0.741, 0.88),
]

LITH_STYLE = {
    "coarse": {
        "class_names": {0: "泥岩", 1: "砂岩"},
        "cmap": ListedColormap(["#8f8f8f", "#f2c84b"]),
        "clim": [-0.5, 1.5],
    },
    "fine": {
        "class_names": {0: "泥岩", 1: "砂岩", 3: "浊积砂岩", 4: "石膏岩"},
        "cmap": "tab10",
        "clim": [-0.5, 4.5],
    },
    "raw": {
        "class_names": {0: "泥岩", 1: "砂岩", 3: "浊积砂岩", 4: "石膏岩", 5: "泥岩_编码5"},
        "cmap": "tab10",
        "clim": [-0.5, 5.5],
    },
}

LITH_BODY_STYLE = {
    0: {"name": "砾", "slug": "gravel", "color": (0, 220, 220)},
    1: {"name": "砂岩", "slug": "sandstone", "color": (245, 214, 45)},
    2: {"name": "泥巴", "slug": "mud", "color": (150, 150, 150)},
}

VOLUME_DISPLAY_STYLE = {
    "seismic": {"filename": "seismic.npy", "label": "地震体", "cmap": "Petrel"},
    "coherence": {"filename": "coherence.npy", "label": "相干体", "cmap": "viridis"},
    "dip_angle_deg": {"filename": "dip_angle_deg.npy", "label": "倾角", "cmap": "turbo"},
    "azimuth_deg": {"filename": "azimuth_deg.npy", "label": "方位角", "cmap": "hsv"},
    "curvature_most_positive": {
        "filename": "curvature_most_positive.npy",
        "label": "最大正曲率",
        "cmap": "RdBu",
    },
    "curvature_most_negative": {
        "filename": "curvature_most_negative.npy",
        "label": "最大负曲率",
        "cmap": "RdBu",
    },
}

MODEL_VOLUME_DISPLAY_STYLE = {
    "model_lithology": {
        "filename": "lithology_volume_seismic.npy",
        "label": "岩相模型",
        "cmap": "tab10",
    },
    "model_porosity": {
        "filename": "porosity_volume_seismic.npy",
        "label": "模型孔隙度",
        "cmap": "viridis",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open seismic + horizons + POR/LITH/PERM wells in one cigvis browser view."
    )
    parser.add_argument(
        "--seismic-npy",
        type=Path,
        default=FULL_DEPTH_SEISMIC_NPY,
        help="Path to the seismic cube.",
    )
    parser.add_argument(
        "--attribute-dir",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "地震属性",
        help="Directory containing computed seismic attribute volumes.",
    )
    parser.add_argument(
        "--volume",
        default="seismic",
        help="Initial 3D slice volume to show: seismic/model_lithology/model_porosity/coherence/dip_angle_deg/azimuth_deg/curvature_most_positive/curvature_most_negative.",
    )
    parser.add_argument(
        "--coords-csv",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT
        / "测井坐标"
        / "combined_well_coordinates_inside_new_seismic_depth_0_654.csv",
        help="Bundled merged coordinate table.",
    )
    parser.add_argument(
        "--por-dir",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "por",
        help="Directory containing bundled processed POR csv files.",
    )
    parser.add_argument(
        "--lith-root",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "lith",
        help="Bundled root directory containing lith/raw, lith/coarse, lith/fine.",
    )
    parser.add_argument(
        "--perm-dir",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "perm",
        help="Directory containing bundled processed PERM csv files.",
    )
    parser.add_argument(
        "--variant",
        choices=["raw", "coarse", "fine"],
        default="coarse",
        help="Which lith label version to visualize.",
    )
    parser.add_argument(
        "--display",
        choices=["both", "por", "lith"],
        default="both",
        help="Which well attributes to display in 3D.",
    )
    parser.add_argument(
        "--layer",
        default="all",
        help="Layer name to display, or 'all'.",
    )
    parser.add_argument(
        "--layer-dir",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "层位",
        help="Directory containing bundled windowed layer npz files.",
    )
    parser.add_argument(
        "--fault",
        default="all",
        help="Fault name to display, 'all', or 'none'. Defaults to all processed fault meshes.",
    )
    parser.add_argument(
        "--fault-dir",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "断层",
        help="Directory containing processed fault mesh npz files.",
    )
    parser.add_argument("--fault-alpha", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--slice-x", type=int)
    parser.add_argument("--slice-y", type=int)
    parser.add_argument("--slice-z", type=int)
    parser.add_argument("--surface-step", type=int, default=4)
    parser.add_argument("--surface-alpha", type=float, default=1.0)
    parser.add_argument("--por-width", type=float, default=2.4)
    parser.add_argument("--lith-width", type=float, default=3.4)
    parser.add_argument("--model-point-width", type=float, default=1.4)
    parser.add_argument("--well-limit", type=int)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=LITH_POR_MODEL_ROOT,
        help="Directory containing converted GRDECL lithology/PORO numpy point clouds.",
    )
    parser.add_argument(
        "--load-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load optional converted GRDECL lithology/PORO model point clouds when available.",
    )
    parser.add_argument(
        "--overlay-volume",
        choices=["none", "model_lithology", "model_porosity", "both"],
        default="none",
        help="Transparent model volume overlay on top of the seismic slices.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.35,
        help="Initial opacity for the transparent model volume overlay.",
    )
    parser.add_argument(
        "--show-lith-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show transparent 3D lithology body meshes around the seismic slices.",
    )
    parser.add_argument(
        "--lith-body-alpha",
        type=float,
        default=0.36,
        help="Initial opacity for transparent lithology body meshes.",
    )
    parser.add_argument(
        "--por-offset-x",
        type=float,
        default=1.2,
        help="Inline-index offset for POR points, used to avoid perfect overlap with LITH.",
    )
    parser.add_argument(
        "--por-offset-y",
        type=float,
        default=0.0,
        help="Crossline-index offset for POR points.",
    )
    parser.add_argument(
        "--lith-offset-x",
        type=float,
        default=-1.2,
        help="Inline-index offset for LITH points, used to avoid perfect overlap with POR.",
    )
    parser.add_argument(
        "--lith-offset-y",
        type=float,
        default=0.0,
        help="Crossline-index offset for LITH points.",
    )
    parser.add_argument(
        "--show-well-names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show floating well names above well heads. Use --no-show-well-names to disable.",
    )
    parser.add_argument("--label-screen-scale", type=float, default=1.2)
    parser.add_argument("--label-z-offset", type=float, default=18.0)
    parser.add_argument(
        "--section-output-dir",
        type=Path,
        default=FULL_DEPTH_PROCESSED_ROOT / "输出" / "well_sections",
        help="Directory for generated well-section HTML files.",
    )
    return parser.parse_args()


def estimate_clim(volume: np.ndarray, x_idx: int, y_idx: int, z_idx: int) -> list[float]:
    probes = [
        np.asarray(volume[x_idx, :, :], dtype=np.float32),
        np.asarray(volume[:, y_idx, :], dtype=np.float32),
        np.asarray(volume[:, :, z_idx], dtype=np.float32),
    ]
    merged = np.concatenate([p.ravel() for p in probes])
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return [0.0, 1.0]
    vmin, vmax = np.percentile(merged, [2.0, 98.0])
    return [float(vmin), float(vmax)]


def estimate_volume_clim(volume_key: str, volume: np.ndarray, x_idx: int, y_idx: int, z_idx: int) -> list[float]:
    if volume_key == "coherence":
        return [0.0, 1.0]
    if volume_key == "azimuth_deg":
        return [0.0, 360.0]
    if volume_key == "model_lithology":
        return [-0.5, 2.5]
    clim = estimate_clim(volume, x_idx, y_idx, z_idx)
    if volume_key.startswith("curvature_"):
        limit = max(abs(clim[0]), abs(clim[1]))
        return [-float(limit), float(limit)]
    return clim


def choose_default_slice_positions(mask: np.ndarray, sample: np.ndarray) -> tuple[int, int, int]:
    valid_xy = np.argwhere(mask)
    x_idx = int(np.median(valid_xy[:, 0]))
    y_idx = int(np.median(valid_xy[:, 1]))
    z_idx = int(np.nanmedian(sample[mask]))
    return x_idx, y_idx, z_idx


def default_slice_index(axis_length: int, ratio: float = 0.85) -> int:
    return int(np.clip(round((axis_length - 1) * ratio), 0, axis_length - 1))


def load_layers(layer_dir: Path, target: str) -> list[dict]:
    npz_paths = sorted(layer_dir.glob("*.npz")) if target == "all" else [layer_dir / f"{target}.npz"]
    if target == "all":
        npz_paths = [path for path in npz_paths if not path.stem.endswith("_fault")]
    if not npz_paths:
        raise SystemExit(f"No layer files found in {layer_dir}")
    layers = []
    for path in npz_paths:
        data = np.load(path)
        layers.append(
            {
                "name": path.stem,
                "sample": data["sample"].astype(np.float32),
                "mask": data["mask"].astype(bool),
                "meta": json.loads(str(data["metadata_json"])),
            }
        )
    return layers


def build_surface(sample: np.ndarray, mask: np.ndarray) -> np.ndarray:
    surface = sample.copy()
    surface[~mask] = np.nan
    return surface


def rgba_float_to_uint8(rgba: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(np.clip(channel, 0.0, 1.0) * 255) for channel in rgba)


def load_fault_meshes(fault_dir: Path, target: str, z_count: int, alpha: float) -> tuple[list, list[str]]:
    if target == "none":
        return [], []
    npz_paths = sorted(fault_dir.glob("*_mesh.npz")) if target == "all" else [fault_dir / f"{target}_mesh.npz"]
    existing_paths = [path for path in npz_paths if path.exists()]
    if not existing_paths:
        return [], [f"No fault mesh files found for target={target!r} in {fault_dir}"]

    fault_nodes = []
    descriptions = []
    for idx, path in enumerate(existing_paths):
        base_rgba = PALETTE[idx % len(PALETTE)]
        color = rgba_float_to_uint8(
            (
                base_rgba[0],
                base_rgba[1],
                base_rgba[2],
                float(np.clip(alpha, 0.0, 1.0)),
            )
        )
        data = np.load(path)
        vertices = data["vertices_ijk"].astype(np.float32)
        faces = data["faces"].astype(np.int32)
        if faces.size == 0:
            descriptions.append(f"{path.stem}: skipped, no faces")
            continue

        in_window = (vertices[:, 2] >= 0.0) & (vertices[:, 2] <= float(z_count - 1))
        face_mask = np.any(in_window[faces], axis=1)
        faces = faces[face_mask]
        if faces.size == 0:
            descriptions.append(f"{path.stem}: skipped, no faces inside current z window")
            continue

        used = np.unique(faces.ravel())
        remap = np.full(vertices.shape[0], -1, dtype=np.int32)
        remap[used] = np.arange(used.size, dtype=np.int32)
        vertices = vertices[used]
        faces = remap[faces]

        node = SurfaceNode(
            vertices=vertices,
            faces=faces,
            color=color,
        )
        node.name = path.stem.replace("_mesh", "")
        node.colored_by = "uniform"
        node._vertices_values = None
        node._vertex_colors = None
        node._color = color
        node._opacity = color[3]
        node._set_color = False
        node._vertices = node.vertices.copy()
        fault_nodes.append(node)

        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
        descriptions.append(
            f"{node.name}: vertices={vertices.shape[0]}, faces={faces.shape[0]}, "
            f"raw_points={meta.get('raw_point_count', 'unknown')}, fill={meta.get('fill_method', 'mesh')}"
        )
    return fault_nodes, descriptions


def load_attribute_logs(
    *,
    coords_csv: Path,
    log_dir: Path,
    value_column: str,
    label: str,
    z_count: int,
    well_limit: int | None,
    x_offset: float,
    y_offset: float,
) -> tuple[list[np.ndarray], list[str], list[str], list[str], dict[str, tuple[float, float, float]]]:
    coord_rows = list(csv.DictReader(coords_csv.open("r", encoding="utf-8-sig", newline="")))
    logs: list[np.ndarray] = []
    used_wells: list[str] = []
    notes: list[str] = []
    skipped: list[str] = []
    head_positions: dict[str, tuple[float, float, float]] = {}

    for row in coord_rows:
        inside_flag = row.get("inside_new_seismic_depth_0_654", row.get("inside_current_window_150_654"))
        if inside_flag != "True":
            continue

        well_name = row["chosen_wellbore"]
        csv_name = (row.get("matched_wells_csv_name") or "").strip()
        if not csv_name:
            skipped.append(f"{well_name}: no matched wells CSV")
            continue

        log_path = log_dir / f"{csv_name}.csv"
        if not log_path.exists():
            skipped.append(f"{well_name}: missing {log_path.name}")
            continue

        inline_idx = float(row["inline_index"])
        xline_idx = float(row["crossline_index"])
        points = []
        with log_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for record in reader:
                depth_text = (record.get("DEPT") or "").strip()
                value_text = (record.get(value_column) or "").strip()
                if not depth_text or not value_text:
                    continue
                try:
                    depth_m = float(depth_text)
                    value = float(value_text)
                except ValueError:
                    continue
                sample = depth_m / DEPTH_STEP_TO_SAMPLE - Z_WINDOW_START
                if 0.0 <= sample < float(z_count):
                    points.append([inline_idx + x_offset, xline_idx + y_offset, sample, value])

        if not points:
            skipped.append(f"{well_name}: no valid {label} samples in current z window")
            continue

        log = np.asarray(points, dtype=np.float32)
        logs.append(log)
        used_wells.append(well_name)
        head_positions[well_name] = (inline_idx, xline_idx, float(np.nanmin(log[:, 2])))
        if value_column == "lith":
            unique_classes = sorted({int(v) for v in log[:, 3].tolist()})
            notes.append(
                f"{well_name}: samples={log.shape[0]}, classes={unique_classes}, "
                f"sample_range=({float(np.min(log[:, 2])):.1f}, {float(np.max(log[:, 2])):.1f})"
            )
        else:
            notes.append(
                f"{well_name}: samples={log.shape[0]}, value_range=({float(np.nanmin(log[:, 3])):.3f}, "
                f"{float(np.nanmax(log[:, 3])):.3f}), sample_range=({float(np.min(log[:, 2])):.1f}, "
                f"{float(np.max(log[:, 2])):.1f})"
            )
        if well_limit is not None and len(logs) >= well_limit:
            break

    return logs, used_wells, notes, skipped, head_positions


def load_model_point_clouds(model_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    notes: list[str] = []
    lith_path = model_dir / "lithology_points_seismic_vis.npy"
    poro_path = model_dir / "porosity_points_seismic_vis.npy"
    metadata_path = model_dir / "metadata.json"

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            grid = metadata.get("grid", {})
            point_summary = metadata.get("point_summary", {})
            notes.append(
                "metadata: "
                f"native_shape=({grid.get('nx', '?')}, {grid.get('ny', '?')}, {grid.get('nz_cropped', '?')}), "
                f"lith_valid={point_summary.get('lith_valid_cells_before_sampling', '?')}, "
                f"poro_valid={point_summary.get('poro_valid_cells_before_sampling', '?')}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"metadata: skipped ({exc})")

    lith_points = None
    if lith_path.exists():
        lith_points = np.load(lith_path, mmap_mode="r")
        if lith_points.ndim != 2 or lith_points.shape[1] != 4:
            raise SystemExit(f"Model lith point cloud shape must be (N, 4): {lith_path}, got {lith_points.shape}")
        values = np.asarray(lith_points[:, 3])
        unique_classes = sorted({int(v) for v in np.unique(values).tolist()})
        notes.append(
            f"lithology points: file={lith_path.name}, count={lith_points.shape[0]}, classes={unique_classes}"
        )
    else:
        notes.append(f"lithology points: missing {lith_path}")

    poro_points = None
    if poro_path.exists():
        poro_points = np.load(poro_path, mmap_mode="r")
        if poro_points.ndim != 2 or poro_points.shape[1] != 4:
            raise SystemExit(f"Model porosity point cloud shape must be (N, 4): {poro_path}, got {poro_points.shape}")
        values = np.asarray(poro_points[:, 3], dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size:
            notes.append(
                f"porosity points: file={poro_path.name}, count={poro_points.shape[0]}, "
                f"value_range=({float(np.nanmin(finite)):.4f}, {float(np.nanmax(finite)):.4f})"
            )
        else:
            notes.append(f"porosity points: file={poro_path.name}, count={poro_points.shape[0]}, no finite values")
    else:
        notes.append(f"porosity points: missing {poro_path}")

    return lith_points, poro_points, notes


def load_lithology_body_meshes(model_dir: Path, alpha: float) -> tuple[list, list[str], list[str]]:
    nodes = []
    descriptions: list[str] = []
    missing: list[str] = []
    alpha_uint8 = int(np.clip(float(alpha), 0.0, 1.0) * 255)

    for class_value, style in LITH_BODY_STYLE.items():
        path = model_dir / f"lithology_body_class_{class_value}_{style['slug']}_mesh.npz"
        if not path.exists():
            missing.append(path.name)
            continue
        data = np.load(path)
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            descriptions.append(f"{style['name']}: skipped empty mesh {path.name}")
            continue
        color = tuple(int(v) for v in (*style["color"], alpha_uint8))
        node = SurfaceNode(vertices=vertices, faces=faces, color=color)
        node.name = f"/岩性透明体/{class_value}_{style['name']}"
        node.colored_by = "uniform"
        node._vertices_values = None
        node._vertex_colors = None
        node._color = color
        node._opacity = color[3]
        node._set_color = False
        node._vertices = node.vertices.copy()
        nodes.append(node)
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
        descriptions.append(
            f"{class_value}={style['name']}: vertices={vertices.shape[0]}, faces={faces.shape[0]}, "
            f"color=rgb{style['color']}, alpha={float(alpha):.2f}, stride={meta.get('stride', '?')}"
        )

    return nodes, descriptions, missing


def create_well_name_labels(
    server,
    head_positions: dict[str, tuple[float, float, float]],
    z_offset: float,
    font_screen_scale: float,
):
    current_scale = (1.0, 1.0, 1.0)
    if hasattr(server, "_gui_scale"):
        current_scale = tuple(float(x) for x in server._gui_scale.value)
    init_scale = tuple(float(x) for x in getattr(server, "init_scale", [1.0, 1.0, 1.0]))

    label_handles = []
    for well_name in sorted(head_positions):
        x, y, z = head_positions[well_name]
        sx = float(x) * init_scale[0] * current_scale[0]
        sy = float(y) * init_scale[1] * current_scale[1]
        sz = float(z) * init_scale[2] * current_scale[2]
        scaled_offset = float(z_offset) * init_scale[2] * current_scale[2]
        safe_name = well_name.replace("/", "_").replace(" ", "_")
        handle = server.scene.add_label(
            f"/well-labels/{safe_name}",
            well_name,
            position=(sx, sy, sz - scaled_offset),
            font_size_mode="screen",
            font_screen_scale=float(font_screen_scale),
            depth_test=False,
            anchor="bottom-center",
        )
        label_handles.append((handle, (float(x), float(y), float(z))))

    return label_handles


def update_well_name_labels(server, label_handles, z_offset: float) -> None:
    current_scale = (1.0, 1.0, 1.0)
    if hasattr(server, "_gui_scale"):
        current_scale = tuple(float(x) for x in server._gui_scale.value)
    init_scale = tuple(float(x) for x in getattr(server, "init_scale", [1.0, 1.0, 1.0]))

    for handle, (x, y, z) in label_handles:
        sx = float(x) * init_scale[0] * current_scale[0]
        sy = float(y) * init_scale[1] * current_scale[1]
        sz = float(z) * init_scale[2] * current_scale[2]
        scaled_offset = float(z_offset) * init_scale[2] * current_scale[2]
        handle.position = (sx, sy, sz - scaled_offset)


def set_label_visibility(label_handles, visible: bool) -> None:
    for handle, _pos in label_handles:
        if hasattr(handle, "visible"):
            handle.visible = visible


def remove_well_name_labels(label_handles) -> None:
    for handle, _pos in label_handles:
        if hasattr(handle, "remove"):
            handle.remove()


def set_node_group_visible(nodes: list, visible: bool) -> None:
    for node in nodes:
        handle = getattr(node, "nodes", None)
        if handle is not None and hasattr(handle, "visible"):
            handle.visible = visible


def node_group_visible(nodes: list) -> bool:
    for node in nodes:
        handle = getattr(node, "nodes", None)
        if handle is not None and hasattr(handle, "visible"):
            return bool(handle.visible)
    return False


def set_node_group_name(nodes: list, group_name: str, item_names: list[str] | None = None) -> None:
    for idx, node in enumerate(nodes):
        item_name = f"item_{idx}"
        if item_names is not None and idx < len(item_names):
            item_name = item_names[idx]
        safe_item_name = item_name.replace("/", "_").replace(" ", "_")
        node.name = f"/{group_name}/{safe_item_name}"


def load_available_volume_specs(
    seismic_path: Path,
    attribute_dir: Path,
    model_dir: Path | None = None,
) -> tuple[dict[str, dict], list[str]]:
    styles: dict[str, dict] = {}
    notes: list[str] = []

    for key, style in VOLUME_DISPLAY_STYLE.items():
        path = seismic_path if key == "seismic" else attribute_dir / style["filename"]
        if not path.exists():
            if key != "seismic":
                notes.append(f"{key}: missing {path.name}, skipped")
            continue
        styles[key] = dict(style)
        styles[key]["path"] = path

    if model_dir is not None:
        for key, style in MODEL_VOLUME_DISPLAY_STYLE.items():
            path = model_dir / style["filename"]
            if not path.exists():
                notes.append(f"{key}: missing {path.name}, skipped")
                continue
            styles[key] = dict(style)
            styles[key]["path"] = path
    return styles, notes


def load_volume_by_key(volume_key: str, volume_styles: dict[str, dict]) -> np.ndarray:
    spec = volume_styles.get(volume_key)
    if spec is None:
        raise KeyError(volume_key)
    return np.load(Path(spec["path"]), mmap_mode="r")


def resolve_overlay_volume_keys(selection: str) -> list[str]:
    if selection == "none":
        return []
    if selection == "both":
        return ["model_lithology", "model_porosity"]
    return [selection]


def swap_slice_volume(
    slice_nodes: list,
    volume: np.ndarray,
    clim: list[float],
    cmap: str,
    server,
) -> None:
    for node in slice_nodes:
        node.volume = volume
        node.clim = list(clim)
        node.cmap = str(cmap)
        node._cmap_preset = str(cmap)
        node.update_node(node.pos)

    if hasattr(server, "_guiclim"):
        server._guiclim.value = tuple(float(v) for v in clim)
    if hasattr(server, "_guicmap"):
        server._guicmap.value = "pre-set"


def main() -> None:
    args = parse_args()
    seismic_path = resolve_path(args.seismic_npy, "seismic_npy")
    attribute_dir = resolve_path(args.attribute_dir, "attribute_dir")
    coords_csv = resolve_path(args.coords_csv, "coords_csv")
    por_dir = resolve_path(args.por_dir, "por_dir")
    perm_dir = resolve_path(args.perm_dir, "perm_dir")
    lith_root = resolve_path(args.lith_root, "lith_root")
    model_dir = resolve_path(args.model_dir, "model_dir")
    layer_dir = resolve_path(args.layer_dir, "layer_dir")
    fault_dir = resolve_path(args.fault_dir, "fault_dir")
    section_output_dir = resolve_path(args.section_output_dir, "section_output_dir")
    lith_dir = resolve_path(lith_root / args.variant, "lith_dir")
    if not lith_dir.exists():
        raise SystemExit(f"Missing lith variant directory: {lith_dir}")

    volume_styles, volume_notes = load_available_volume_specs(seismic_path, attribute_dir, model_dir)
    volume_option_labels = {key: value["label"] for key, value in volume_styles.items()}
    if "seismic" not in volume_styles:
        raise SystemExit(f"Missing seismic volume: {seismic_path}")
    seismic = load_volume_by_key("seismic", volume_styles)
    initial_volume = str(args.volume)
    if initial_volume not in volume_styles:
        print(f"Warning: requested volume {initial_volume!r} is not available, falling back to 'seismic'.")
        initial_volume = "seismic"

    layers = load_layers(layer_dir, args.layer)

    combined_mask = np.zeros_like(layers[0]["mask"], dtype=bool)
    z_candidates = []
    for layer in layers:
        combined_mask |= layer["mask"]
        valid = layer["sample"][layer["mask"]]
        if valid.size > 0:
            z_candidates.append(np.nanmedian(valid))

    if combined_mask.shape != seismic.shape[:2]:
        raise SystemExit(
            f"Layer grid shape {combined_mask.shape} does not match seismic xy shape {seismic.shape[:2]}"
        )

    default_x = default_slice_index(seismic.shape[0])
    default_y = default_slice_index(seismic.shape[1])
    default_z = default_slice_index(seismic.shape[2])
    x_idx = int(np.clip(default_x if args.slice_x is None else args.slice_x, 0, seismic.shape[0] - 1))
    y_idx = int(np.clip(default_y if args.slice_y is None else args.slice_y, 0, seismic.shape[1] - 1))
    z_idx = int(np.clip(default_z if args.slice_z is None else args.slice_z, 0, seismic.shape[2] - 1))
    initial_volume_data = seismic if initial_volume == "seismic" else load_volume_by_key(initial_volume, volume_styles)
    if tuple(initial_volume_data.shape) != tuple(seismic.shape):
        raise SystemExit(
            f"Volume shape mismatch for {initial_volume}: {tuple(initial_volume_data.shape)} vs seismic {tuple(seismic.shape)}"
        )
    initial_clim = estimate_volume_clim(initial_volume, initial_volume_data, x_idx, y_idx, z_idx)
    slice_nodes = viserplot.create_slices(
        initial_volume_data,
        pos={"x": [x_idx], "y": [y_idx], "z": [z_idx]},
        clim=initial_clim,
        cmap=str(volume_styles[initial_volume]["cmap"]),
    )
    requested_overlay_keys = resolve_overlay_volume_keys(args.overlay_volume)
    overlay_keys = [
        key for key in requested_overlay_keys if key in volume_styles and key != initial_volume
    ]
    missing_overlay_keys = [key for key in requested_overlay_keys if key not in volume_styles]
    duplicate_overlay_keys = [key for key in requested_overlay_keys if key == initial_volume]
    overlay_volumes = []
    overlay_clims = []
    overlay_cmaps = []
    for key in overlay_keys:
        overlay_volume = load_volume_by_key(key, volume_styles)
        if tuple(overlay_volume.shape) != tuple(seismic.shape):
            print(f"Warning: overlay volume {key} shape {tuple(overlay_volume.shape)} != seismic {tuple(seismic.shape)}, skipped.")
            continue
        overlay_volumes.append(overlay_volume)
        overlay_clims.append(estimate_volume_clim(key, overlay_volume, x_idx, y_idx, z_idx))
        overlay_cmaps.append(str(volume_styles[key]["cmap"]))
    if overlay_volumes:
        viserplot.add_mask(
            slice_nodes,
            overlay_volumes,
            clims=overlay_clims,
            cmaps=overlay_cmaps,
            alpha=float(np.clip(args.overlay_alpha, 0.0, 1.0)),
            excpt="none",
        )

    surface_nodes = []
    surface_node_names = []
    layer_descriptions = []
    for idx, layer in enumerate(layers):
        surface = build_surface(layer["sample"], layer["mask"])
        valid_nonnegative = np.isfinite(surface) & (surface >= 0)
        if not valid_nonnegative.any():
            continue
        base_rgba = PALETTE[idx % len(PALETTE)]
        rgba = (
            base_rgba[0],
            base_rgba[1],
            base_rgba[2],
            float(np.clip(args.surface_alpha, 0.0, 1.0)),
        )
        nodes_for_layer = viserplot.create_surfaces(
            surface,
            value_type="depth",
            clim=[0.0, float(seismic.shape[2] - 1)],
            cmap="jet",
            step1=args.surface_step,
            step2=args.surface_step,
        )
        color_uint8 = rgba_float_to_uint8(rgba)
        for node in nodes_for_layer:
            node.name = layer["name"]
            node.colored_by = "uniform"
            node._vertices_values = None
            node._vertex_colors = None
            node._color = color_uint8
            node._opacity = color_uint8[3]
            node._set_color = False
            node._vertices = node.vertices.copy()
        surface_nodes += nodes_for_layer
        surface_node_names += [layer["name"]] * len(nodes_for_layer)
        layer_descriptions.append(
            f"{layer['name']}: rgba=({rgba[0]:.3f}, {rgba[1]:.3f}, {rgba[2]:.3f}, {rgba[3]:.2f}), "
            f"valid={layer['meta']['valid_ratio']:.4f}"
        )

    por_logs: list[np.ndarray] = []
    por_wells: list[str] = []
    por_notes: list[str] = []
    por_skipped: list[str] = []
    por_heads: dict[str, tuple[float, float, float]] = {}
    if args.display in {"both", "por"}:
        por_logs, por_wells, por_notes, por_skipped, por_heads = load_attribute_logs(
            coords_csv=coords_csv,
            log_dir=por_dir,
            value_column="por",
            label="POR",
            z_count=seismic.shape[2],
            well_limit=args.well_limit,
            x_offset=float(args.por_offset_x),
            y_offset=float(args.por_offset_y),
        )

    perm_logs: list[np.ndarray] = []
    perm_wells: list[str] = []
    perm_notes: list[str] = []
    perm_skipped: list[str] = []
    perm_heads: dict[str, tuple[float, float, float]] = {}
    perm_logs, perm_wells, perm_notes, perm_skipped, perm_heads = load_attribute_logs(
        coords_csv=coords_csv,
        log_dir=perm_dir,
        value_column="perm",
        label="PERM",
        z_count=seismic.shape[2],
        well_limit=args.well_limit,
        x_offset=0.0,
        y_offset=0.0,
    )

    lith_logs: list[np.ndarray] = []
    lith_wells: list[str] = []
    lith_notes: list[str] = []
    lith_skipped: list[str] = []
    lith_heads: dict[str, tuple[float, float, float]] = {}
    if args.display in {"both", "lith"}:
        lith_logs, lith_wells, lith_notes, lith_skipped, lith_heads = load_attribute_logs(
            coords_csv=coords_csv,
            log_dir=lith_dir,
            value_column="lith",
            label=f"岩性-{args.variant}",
            z_count=seismic.shape[2],
            well_limit=args.well_limit,
            x_offset=float(args.lith_offset_x),
            y_offset=float(args.lith_offset_y),
        )

    if args.display in {"both", "por"} and not por_logs:
        print("Warning: no POR wells found for the current dataset window.")
    if args.display in {"both", "lith"} and not lith_logs:
        print(f"Warning: no LITH wells found for variant {args.variant}.")
    if not por_logs and not lith_logs and not perm_logs:
        raise SystemExit("No POR, LITH, or PERM wells found for the selected display mode.")

    por_nodes = []
    por_clim = None
    if por_logs:
        por_values = np.concatenate([log[:, 3] for log in por_logs])
        por_clim = [float(np.nanpercentile(por_values, 2.0)), float(np.nanpercentile(por_values, 98.0))]
        por_nodes = viserplot.create_well_logs(
            por_logs,
            logs_type="point",
            cmap="viridis",
            clim=por_clim,
            width=args.por_width,
            point_shape="circle",
        )
        set_node_group_name(por_nodes, "井数据/孔隙度", por_wells)

    lith_nodes = []
    if lith_logs:
        style = LITH_STYLE[args.variant]
        lith_nodes = viserplot.create_well_logs(
            lith_logs,
            logs_type="point",
            cmap=style["cmap"],
            clim=style["clim"],
            width=args.lith_width,
            point_shape="square",
        )
        set_node_group_name(lith_nodes, f"井数据/岩性_{args.variant}", lith_wells)

    perm_nodes = []
    perm_clim = None
    if perm_logs:
        perm_values = np.concatenate([log[:, 3] for log in perm_logs])
        perm_clim = [float(np.nanpercentile(perm_values, 2.0)), float(np.nanpercentile(perm_values, 98.0))]
        perm_nodes = viserplot.create_well_logs(
            perm_logs,
            logs_type="point",
            cmap="plasma",
            clim=perm_clim,
            width=args.por_width,
            point_shape="diamond",
        )
        set_node_group_name(perm_nodes, "井数据/渗透率", perm_wells)

    model_lith_nodes = []
    model_poro_nodes = []
    model_notes: list[str] = []
    model_poro_clim = None
    if args.load_model:
        model_lith_points, model_poro_points, model_notes = load_model_point_clouds(model_dir)
        if model_lith_points is not None and model_lith_points.shape[0] > 0:
            model_lith_nodes = viserplot.create_well_logs(
                model_lith_points,
                logs_type="point",
                cmap="tab10",
                clim=[-0.5, 2.5],
                width=args.model_point_width,
                point_shape="square",
            )
            set_node_group_name(model_lith_nodes, "岩相模型/岩性")
        if model_poro_points is not None and model_poro_points.shape[0] > 0:
            model_poro_values = np.asarray(model_poro_points[:, 3], dtype=np.float32)
            finite_model_poro = model_poro_values[np.isfinite(model_poro_values)]
            if finite_model_poro.size:
                model_poro_clim = [
                    float(np.nanpercentile(finite_model_poro, 2.0)),
                    float(np.nanpercentile(finite_model_poro, 98.0)),
                ]
            model_poro_nodes = viserplot.create_well_logs(
                model_poro_points,
                logs_type="point",
                cmap="viridis",
                clim=model_poro_clim,
                width=args.model_point_width,
                point_shape="circle",
            )
            set_node_group_name(model_poro_nodes, "岩相模型/孔隙度")

    lith_body_nodes, lith_body_descriptions, missing_lith_body_meshes = load_lithology_body_meshes(
        model_dir,
        alpha=float(args.lith_body_alpha),
    )

    fault_nodes, fault_descriptions = load_fault_meshes(
        fault_dir=fault_dir,
        target=str(args.fault),
        z_count=seismic.shape[2],
        alpha=float(args.fault_alpha),
    )

    set_node_group_name(surface_nodes, "层位", surface_node_names)
    set_node_group_name(fault_nodes, "断层")

    server = viserplot.create_server(
        port=args.port,
        label=f"地震-孔隙度-岩性-渗透率浏览器-{args.variant}",
        verbose=False,
    )
    print(f"Starting seismic + horizons + POR/LITH view, display={args.display}, seismic_shape={tuple(seismic.shape)}")
    print(f"Using cigvis source: {Path(viserplot.__file__).resolve()}")
    print(f"Primary POR column: {POR_COLUMN}")
    print(f"Using slice positions x={x_idx}, y={y_idx}, z={z_idx}")
    print(f"Initial 3D volume: {initial_volume}, clim={initial_clim}, cmap={volume_styles[initial_volume]['cmap']}")
    if overlay_volumes:
        overlay_desc = ", ".join(
            f"{key}(clim={clim}, cmap={cmap})"
            for key, clim, cmap in zip(overlay_keys, overlay_clims, overlay_cmaps)
        )
        print(
            f"Transparent overlays: {overlay_desc}, "
            f"alpha={float(np.clip(args.overlay_alpha, 0.0, 1.0)):.2f}"
        )
    if missing_overlay_keys:
        print(f"Skipped requested overlays because files are unavailable: {', '.join(missing_overlay_keys)}")
    if duplicate_overlay_keys:
        print(f"Skipped overlays already used as the base volume: {', '.join(duplicate_overlay_keys)}")
    print("Available 3D volumes:")
    for key in volume_styles:
        print(f"  {key}: path={volume_styles[key]['path'].name}, cmap={volume_styles[key]['cmap']}")
    print(f"POR dir: {por_dir}")
    print(f"Lith dir: {lith_dir}")
    print(f"Perm dir: {perm_dir}")
    print(f"Layer set: {args.layer}, surface_step={args.surface_step}, surface_alpha={float(np.clip(args.surface_alpha, 0.0, 1.0)):.2f}")
    print(f"Fault set: {args.fault}, fault_alpha={float(np.clip(args.fault_alpha, 0.0, 1.0)):.2f}")
    print(f"Offsets: POR=({float(args.por_offset_x):.2f}, {float(args.por_offset_y):.2f}), LITH=({float(args.lith_offset_x):.2f}, {float(args.lith_offset_y):.2f})")
    print(f"Displayed POR wells ({len(por_wells)}): {', '.join(por_wells) if por_wells else '(none)'}")
    print(f"Displayed LITH wells ({len(lith_wells)}): {', '.join(lith_wells) if lith_wells else '(none)'}")
    print(f"Displayed PERM wells ({len(perm_wells)}): {', '.join(perm_wells) if perm_wells else '(none)'}")
    if por_clim is not None:
        print(f"POR visual clim={por_clim}, cmap=viridis")
    if lith_logs:
        print(f"LITH class names: {LITH_STYLE[args.variant]['class_names']}")
        print(f"LITH visual clim={LITH_STYLE[args.variant]['clim']}, cmap={LITH_STYLE[args.variant]['cmap']}")
    if perm_clim is not None:
        print(f"PERM visual clim={perm_clim}, cmap=plasma")
    if model_notes:
        print(f"Model dir: {model_dir}")
        print("GRDECL model details:")
        for desc in model_notes:
            print(f"  {desc}")
    if model_lith_nodes:
        print("GRDECL model LITH visual clim=[-0.5, 2.5], cmap=tab10")
    if model_poro_clim is not None:
        print(f"GRDECL model POR visual clim={model_poro_clim}, cmap=viridis")
    if lith_body_descriptions:
        print("Lithology transparent body meshes:")
        for desc in lith_body_descriptions:
            print(f"  {desc}")
    if missing_lith_body_meshes:
        print("Missing lithology body meshes:")
        for name in missing_lith_body_meshes:
            print(f"  {name}")
    print("Layer colors:")
    for desc in layer_descriptions:
        print(f"  {desc}")
    if fault_descriptions:
        print("Fault meshes:")
        for desc in fault_descriptions:
            print(f"  {desc}")
    if por_notes:
        print("POR well details:")
        for desc in por_notes:
            print(f"  {desc}")
    if lith_notes:
        print("LITH well details:")
        for desc in lith_notes:
            print(f"  {desc}")
    if perm_notes:
        print("PERM well details:")
        for desc in perm_notes:
            print(f"  {desc}")
    if por_skipped:
        print("Skipped POR wells:")
        for desc in por_skipped[:30]:
            print(f"  {desc}")
        if len(por_skipped) > 30:
            print(f"  ... and {len(por_skipped) - 30} more")
    if lith_skipped:
        print("Skipped LITH wells:")
        for desc in lith_skipped[:30]:
            print(f"  {desc}")
        if len(lith_skipped) > 30:
            print(f"  ... and {len(lith_skipped) - 30} more")
    if perm_skipped:
        print("Skipped PERM wells:")
        for desc in perm_skipped[:30]:
            print(f"  {desc}")
        if len(perm_skipped) > 30:
            print(f"  ... and {len(perm_skipped) - 30} more")
    if volume_notes:
        print("Volume notes:")
        for desc in volume_notes:
            print(f"  {desc}")
    print(f"Open in browser: http://127.0.0.1:{args.port}")
    print("Stop the server with Ctrl+C in this terminal.")

    viserplot.plot3D(
        slice_nodes
        + lith_body_nodes
        + surface_nodes
        + fault_nodes
        + por_nodes
        + lith_nodes
        + perm_nodes
        + model_lith_nodes
        + model_poro_nodes,
        server=server,
        run_app=False,
    )
    visibility_state = {
        "layers": False,
        "faults": False,
        "por": False,
        "lith": False,
        "perm": False,
        "model_lith": False,
        "model_poro": False,
        "lith_body": bool(args.show_lith_body),
        "well_names": False,
    }
    label_state = {"handles": []}

    set_node_group_visible(surface_nodes, False)
    set_node_group_visible(fault_nodes, False)
    set_node_group_visible(por_nodes, False)
    set_node_group_visible(lith_nodes, False)
    set_node_group_visible(perm_nodes, False)
    set_node_group_visible(model_lith_nodes, False)
    set_node_group_visible(model_poro_nodes, False)
    set_node_group_visible(lith_body_nodes, bool(args.show_lith_body))

    head_positions = dict(lith_heads)
    head_positions.update(por_heads)
    head_positions.update(perm_heads)
    if hasattr(server, "_gui_scale"):
        def _on_scale_change(_):
            if label_state["handles"]:
                update_well_name_labels(
                    server=server,
                    label_handles=label_state["handles"],
                    z_offset=float(args.label_z_offset),
                )

        server._gui_scale.on_update(
            _on_scale_change
        )
    if args.show_well_names and head_positions:
        print(
            f"Well name labels available: count={len(head_positions)}, initial_visible=False, "
            f"font_screen_scale={float(args.label_screen_scale):.2f}, "
            f"z_offset={float(args.label_z_offset):.2f}"
        )

    with server.gui.add_folder("体数据"):
        if volume_styles:
            volume_dropdown = server.gui.add_dropdown(
                "切片体数据",
                options=tuple(volume_option_labels[key] for key in volume_styles.keys()),
                initial_value=volume_option_labels[initial_volume],
            )
            volume_state = {
                "current_key": initial_volume,
                "current_volume": initial_volume_data,
            }

            def _on_volume_change(_):
                selected_label = str(volume_dropdown.value)
                selected = next(
                    (key for key, label in volume_option_labels.items() if label == selected_label),
                    volume_state["current_key"],
                )
                if selected == volume_state["current_key"]:
                    return
                next_volume = load_volume_by_key(selected, volume_styles)
                if tuple(next_volume.shape) != tuple(seismic.shape):
                    print(
                        f"Skip volume switch: {selected} shape {tuple(next_volume.shape)} != seismic {tuple(seismic.shape)}"
                    )
                    volume_dropdown.value = volume_state["current_key"]
                    return
                next_clim = estimate_volume_clim(selected, next_volume, x_idx, y_idx, z_idx)
                swap_slice_volume(
                    slice_nodes=slice_nodes,
                    volume=next_volume,
                    clim=next_clim,
                    cmap=str(volume_styles[selected]["cmap"]),
                    server=server,
                )
                volume_state["current_key"] = selected
                volume_state["current_volume"] = next_volume
                print(
                    f"Switched 3D volume to {selected}, path={volume_styles[selected]['path'].name}, "
                    f"clim={next_clim}, cmap={volume_styles[selected]['cmap']}"
                )

            volume_dropdown.on_update(_on_volume_change)

    with server.gui.add_folder("岩性透明体"):
        show_lith_body = server.gui.add_checkbox("显示岩性透明体", initial_value=bool(args.show_lith_body))
        lith_body_alpha = server.gui.add_slider(
            "透明度",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=float(np.clip(args.lith_body_alpha, 0.0, 1.0)),
        )

        def _on_lith_body_visibility_change(_):
            visibility_state["lith_body"] = bool(show_lith_body.value)
            if lith_body_nodes:
                set_node_group_visible(lith_body_nodes, visibility_state["lith_body"])

        def _on_lith_body_alpha_change(_):
            if lith_body_nodes:
                for node in lith_body_nodes:
                    node.opacity = float(lith_body_alpha.value)

        show_lith_body.on_update(_on_lith_body_visibility_change)
        lith_body_alpha.on_update(_on_lith_body_alpha_change)

        if not lith_body_nodes:
            missing_lith_body_text = "\n".join(f"- `{name}`" for name in missing_lith_body_meshes)
            generate_lith_body_command = (
                r"E:\miniconda\envs\py312\python.exe -B "
                r"D:\商书记项目\tools\create_lithology_body_meshes.py --overwrite"
            )
            server.gui.add_markdown(
                "未找到岩性透明体网格文件，暂时无法显示透明体。\n\n"
                f"{missing_lith_body_text}\n\n"
                "请先生成网格后重启脚本：\n\n"
                f"`{generate_lith_body_command}`"
            )

    with server.gui.add_folder("层位"):
        if surface_nodes:
            show_layers = server.gui.add_checkbox("显示层位", initial_value=False)

            def _on_layers_visibility_change(_):
                visibility_state["layers"] = bool(show_layers.value)
                set_node_group_visible(surface_nodes, visibility_state["layers"])

            show_layers.on_update(_on_layers_visibility_change)

    with server.gui.add_folder("断层"):
        if fault_nodes:
            show_faults = server.gui.add_checkbox("显示断层", initial_value=False)

            def _on_faults_visibility_change(_):
                visibility_state["faults"] = bool(show_faults.value)
                set_node_group_visible(fault_nodes, visibility_state["faults"])

            show_faults.on_update(_on_faults_visibility_change)

    with server.gui.add_folder("井数据"):
        if por_nodes:
            show_por = server.gui.add_checkbox("显示孔隙度(POR)", initial_value=False)

            def _on_por_visibility_change(_):
                visibility_state["por"] = bool(show_por.value)
                set_node_group_visible(por_nodes, visibility_state["por"])

            show_por.on_update(_on_por_visibility_change)
        if lith_nodes:
            show_lith = server.gui.add_checkbox("显示岩性(LITH)", initial_value=False)

            def _on_lith_visibility_change(_):
                visibility_state["lith"] = bool(show_lith.value)
                set_node_group_visible(lith_nodes, visibility_state["lith"])

            show_lith.on_update(_on_lith_visibility_change)
        if perm_nodes:
            show_perm = server.gui.add_checkbox("显示渗透率(PERM)", initial_value=False)

            def _on_perm_visibility_change(_):
                visibility_state["perm"] = bool(show_perm.value)
                set_node_group_visible(perm_nodes, visibility_state["perm"])

            show_perm.on_update(_on_perm_visibility_change)
        if args.show_well_names and head_positions:
            show_well_names = server.gui.add_checkbox("显示井名", initial_value=False)

            def _on_well_name_visibility_change(_):
                visibility_state["well_names"] = bool(show_well_names.value)
                if visibility_state["well_names"]:
                    if not label_state["handles"]:
                        label_state["handles"] = create_well_name_labels(
                            server=server,
                            head_positions=head_positions,
                            z_offset=float(args.label_z_offset),
                            font_screen_scale=float(label_size_slider.value),
                        )
                        update_well_name_labels(
                            server=server,
                            label_handles=label_state["handles"],
                            z_offset=float(args.label_z_offset),
                        )
                    else:
                        set_label_visibility(label_state["handles"], True)
                else:
                    remove_well_name_labels(label_state["handles"])
                    label_state["handles"] = []

            label_size_slider = server.gui.add_slider(
                "井名字号",
                min=0.2,
                max=4.0,
                step=0.1,
                initial_value=float(args.label_screen_scale),
            )

            def _on_label_size_change(_):
                for handle, _pos in label_state["handles"]:
                    handle.font_screen_scale = float(label_size_slider.value)

            show_well_names.on_update(_on_well_name_visibility_change)
            label_size_slider.on_update(_on_label_size_change)

    with server.gui.add_folder("岩相模型"):
        if model_lith_nodes:
            show_model_lith = server.gui.add_checkbox("显示模型岩性(LITH)", initial_value=False)

            def _on_model_lith_visibility_change(_):
                visibility_state["model_lith"] = bool(show_model_lith.value)
                set_node_group_visible(model_lith_nodes, visibility_state["model_lith"])

            show_model_lith.on_update(_on_model_lith_visibility_change)
        if model_poro_nodes:
            show_model_poro = server.gui.add_checkbox("显示模型孔隙度(POR)", initial_value=False)

            def _on_model_poro_visibility_change(_):
                visibility_state["model_poro"] = bool(show_model_poro.value)
                set_node_group_visible(model_poro_nodes, visibility_state["model_poro"])

            show_model_poro.on_update(_on_model_poro_visibility_change)

    section_mode = None
    section_wells = []
    section_log_dir = None
    if lith_wells:
        section_mode = "lith"
        section_wells = lith_wells
        section_log_dir = lith_dir
    elif por_wells:
        section_mode = "por"
        section_wells = por_wells
        section_log_dir = por_dir
    elif perm_wells:
        section_mode = "por"
        section_wells = perm_wells
        section_log_dir = perm_dir

    if section_mode is not None and section_log_dir is not None:
        attach_well_section_gui(
            server=server,
            mode=section_mode,
            available_wells=section_wells,
            coords_csv=coords_csv,
            log_dir=section_log_dir,
            layer_dir=layer_dir,
            output_dir=section_output_dir,
            seismic_path=seismic_path,
            z_count=seismic.shape[2],
            folder_name="井剖面",
            scene_prefix="well-section",
            fault_dir=fault_dir,
            por_dir=por_dir,
            perm_dir=perm_dir,
        )

    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.stop()
        print("Execution interrupted")


if __name__ == "__main__":
    main()
