from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import webbrowser

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parent
VISUAL_ROOT = CODE_DIR.parent
BUNDLE_ROOT = VISUAL_ROOT.parent
PROCESSED_ROOT = BUNDLE_ROOT / "处理后文件"


def ensure_inside_bundle(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(BUNDLE_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{label} must stay inside bundle: {resolved}") from exc
    return resolved


def _prefer_local_cigvis() -> None:
    cigvis_dir = VISUAL_ROOT / "cigvis"
    if not cigvis_dir.is_dir():
        raise SystemExit(f"Missing bundled cigvis directory: {cigvis_dir}")
    root = str(VISUAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_prefer_local_cigvis()

import cigvis
import cigvis.vispyplot as vispyplot
from cigvis import colormap
from cigvis.meshs import surface2mesh
from cigvis.vispynodes import AxisAlignedImage, VisCanvas
from cigvis.vispynodes.axis_aligned_image import get_image_func
from matplotlib.colors import ListedColormap
from PyQt5 import QtCore, QtWidgets
from vispy.app import use_app
from vispy.scene.visuals import Mesh, Text
from well_section import build_well_section_html
from well_section.section import (
    POR_STYLES,
    SEISMIC_COLOR_OPTIONS,
    SEISMIC_DISPLAY_OPTIONS,
    _seismic_display_setting,
)

Z_WINDOW_START = 150.0
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

VOLUME_OPTION_LABELS = {
    key: value["label"] for key, value in VOLUME_DISPLAY_STYLE.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open seismic + horizons + POR/LITH/PERM wells in one cigvis desktop view."
    )
    parser.add_argument(
        "--seismic-npy",
        type=Path,
        default=PROCESSED_ROOT / "地震" / "seismic.npy",
        help="Path to the current windowed seismic cube.",
    )
    parser.add_argument(
        "--attribute-dir",
        type=Path,
        default=PROCESSED_ROOT / "地震属性",
        help="Directory containing computed seismic attribute volumes.",
    )
    parser.add_argument(
        "--volume",
        default="seismic",
        help="Initial 3D slice volume to show: seismic/coherence/dip_angle_deg/azimuth_deg/curvature_most_positive/curvature_most_negative.",
    )
    parser.add_argument(
        "--coords-csv",
        type=Path,
        default=PROCESSED_ROOT / "测井坐标" / "combined_well_coordinates_inside_current_window.csv",
        help="Bundled merged coordinate table.",
    )
    parser.add_argument(
        "--por-dir",
        type=Path,
        default=PROCESSED_ROOT / "por",
        help="Directory containing bundled processed POR csv files.",
    )
    parser.add_argument(
        "--lith-root",
        type=Path,
        default=PROCESSED_ROOT / "lith",
        help="Bundled root directory containing lith/raw, lith/coarse, lith/fine.",
    )
    parser.add_argument(
        "--perm-dir",
        type=Path,
        default=PROCESSED_ROOT / "perm",
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
        default=PROCESSED_ROOT / "层位",
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
        default=PROCESSED_ROOT / "断层",
        help="Directory containing processed fault mesh npz files.",
    )
    parser.add_argument("--fault-alpha", type=float, default=1.0)
    parser.add_argument(
        "--show-layers",
        action="store_true",
        default=False,
        help="Show horizon layers when the desktop window opens.",
    )
    parser.add_argument(
        "--show-faults",
        action="store_true",
        default=False,
        help="Show fault meshes when the desktop window opens.",
    )
    parser.add_argument(
        "--show-por",
        action="store_true",
        default=False,
        help="Show POR wells when the desktop window opens.",
    )
    parser.add_argument(
        "--show-lith",
        action="store_true",
        default=False,
        help="Show LITH wells when the desktop window opens.",
    )
    parser.add_argument(
        "--show-perm",
        action="store_true",
        default=False,
        help="Show PERM wells when the desktop window opens.",
    )
    parser.add_argument("--slice-x", type=int)
    parser.add_argument("--slice-y", type=int)
    parser.add_argument("--slice-z", type=int)
    parser.add_argument("--surface-step", type=int, default=4)
    parser.add_argument("--surface-alpha", type=float, default=1.0)
    parser.add_argument("--por-width", type=float, default=2.4)
    parser.add_argument("--lith-width", type=float, default=3.4)
    parser.add_argument("--well-limit", type=int)
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
        default=VISUAL_ROOT / "输出" / "well_sections",
        help="Directory for generated well-section HTML files.",
    )
    parser.add_argument(
        "--desktop-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 860),
        help="Desktop VisPy window size.",
    )
    return parser.parse_args()


def estimate_clim(volume: np.ndarray, x_idx: int, y_idx: int, z_idx: int) -> list[float]:
    probes = [
        np.asarray(volume[x_idx, :, :], dtype=np.float32),
        np.asarray(volume[:, y_idx, :], dtype=np.float32),
        np.asarray(volume[:, :, z_idx], dtype=np.float32),
    ]
    merged = np.concatenate([p.ravel() for p in probes])
    vmin, vmax = np.percentile(merged, [2.0, 98.0])
    return [float(vmin), float(vmax)]


def estimate_volume_clim(volume_key: str, volume: np.ndarray, x_idx: int, y_idx: int, z_idx: int) -> list[float]:
    if volume_key == "coherence":
        return [0.0, 1.0]
    if volume_key == "azimuth_deg":
        return [0.0, 360.0]
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


def rgba_float(rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(float(np.clip(channel, 0.0, 1.0)) for channel in rgba)


def make_uniform_surface_node(
    surface: np.ndarray,
    *,
    color: tuple[float, float, float, float],
    step: int,
    name: str,
) -> Mesh | None:
    mask = np.logical_or(~np.isfinite(surface), surface < 0)
    vertices, faces = surface2mesh(
        surface,
        mask,
        anti_rot=True,
        step1=max(1, int(step)),
        step2=max(1, int(step)),
    )
    if faces.size == 0:
        return None
    node = Mesh(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        color=rgba_float(color),
        shading="smooth",
    )
    node.unfreeze()
    node.name = name
    node.freeze()
    return node


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
        color = rgba_float((base_rgba[0], base_rgba[1], base_rgba[2], float(np.clip(alpha, 0.0, 1.0))))
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

        node = Mesh(
            vertices=vertices,
            faces=faces,
            color=color,
            shading="smooth",
        )
        node.unfreeze()
        node.name = path.stem.replace("_mesh", "")
        node.freeze()
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
        if row["inside_current_window_150_654"] != "True":
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


def set_node_group_visible(nodes: list, visible: bool) -> None:
    for node in nodes:
        if hasattr(node, "visible"):
            node.visible = visible


def node_group_visible(nodes: list) -> bool:
    for node in nodes:
        if hasattr(node, "visible"):
            return bool(node.visible)
    return False


def set_node_group_name(nodes: list, group_name: str, item_names: list[str] | None = None) -> None:
    for idx, node in enumerate(nodes):
        item_name = f"item_{idx}"
        if item_names is not None and idx < len(item_names):
            item_name = item_names[idx]
        safe_item_name = item_name.replace("/", "_").replace(" ", "_")
        if hasattr(node, "unfreeze"):
            node.unfreeze()
        node.name = f"/{group_name}/{safe_item_name}"
        if hasattr(node, "freeze"):
            node.freeze()


def create_desktop_log_point_nodes(
    logs: list[np.ndarray],
    *,
    cmap,
    clim: list[float],
    radius: float,
    names: list[str],
) -> list:
    nodes = []
    vispy_cmap = colormap.cmap_to_vispy(cmap)
    for idx, log in enumerate(logs):
        if log.size == 0:
            continue
        point_nodes = vispyplot.create_points(
            log[:, :3],
            r=float(radius),
            color=None,
            cmap=vispy_cmap,
            clim=clim,
            vertex_values=log[:, 3],
            shading="flat",
        )
        set_node_group_name(point_nodes, "井数据", [names[idx] if idx < len(names) else f"well_{idx}"])
        nodes.extend(point_nodes)
    return nodes


def create_desktop_well_name_labels(
    head_positions: dict[str, tuple[float, float, float]],
    *,
    z_offset: float,
    font_screen_scale: float,
) -> list:
    labels = []
    font_size = max(24.0, 40.0 * float(font_screen_scale))
    for well_name in sorted(head_positions):
        x, y, z = head_positions[well_name]
        node = Text(
            well_name,
            pos=(float(x), float(y), float(z) - float(z_offset)),
            color=(0.02, 0.02, 0.02, 1.0),
            font_size=font_size,
            anchor_x="center",
            anchor_y="bottom",
            depth_test=False,
        )
        node.set_gl_state(depth_test=False, blend=True)
        if hasattr(node, "unfreeze"):
            node.unfreeze()
        node.base_position = (float(x), float(y), float(z))
        if hasattr(node, "freeze"):
            node.freeze()
        node.visible = False
        labels.append(node)
    return labels


def create_desktop_well_pick_target_nodes(
    wells: list[str],
    head_positions: dict[str, tuple[float, float, float]],
    *,
    radius: float = 5.5,
) -> tuple[list, dict[str, tuple[float, float, float]]]:
    nodes = []
    positions: dict[str, tuple[float, float, float]] = {}
    for well_name in wells:
        position = head_positions.get(well_name)
        if position is None:
            continue
        point_nodes = vispyplot.create_points(
            np.asarray([position], dtype=np.float32),
            r=float(radius),
            color="#ffe600",
            shading="flat",
        )
        if not point_nodes:
            continue
        node = point_nodes[0]
        if hasattr(node, "unfreeze"):
            node.unfreeze()
        node.name = f"/井剖面/pick-targets/{well_name}"
        node.well_name = well_name
        if hasattr(node, "interactive"):
            node.interactive = True
        if hasattr(node, "freeze"):
            node.freeze()
        node.visible = False
        nodes.append(node)
        positions[well_name] = tuple(float(v) for v in position)
    return nodes, positions


def update_slice_volume_desktop(slice_nodes: list, volume: np.ndarray, clim: list[float], cmap) -> None:
    vispy_cmap = colormap.cmap_to_vispy(cmap)
    for node in slice_nodes:
        if not isinstance(node, AxisAlignedImage):
            continue
        node.image_funcs[0] = get_image_func(node.axis, volume, None)
        node.cmap = vispy_cmap
        node.clim = list(clim)
        node._update_location(node.pos)


def update_slice_clim_desktop(slice_nodes: list, clim: list[float]) -> None:
    for node in slice_nodes:
        if isinstance(node, AxisAlignedImage):
            node.clim = list(clim)


def update_slice_cmap_desktop(slice_nodes: list, cmap_name: str) -> None:
    if cmap_name == "pre-set":
        return
    vispy_cmap = colormap.cmap_to_vispy(cmap_name)
    for node in slice_nodes:
        if isinstance(node, AxisAlignedImage):
            node.cmap = vispy_cmap


def update_scene_scale_desktop(canvas: VisCanvas, scale: tuple[float, float, float]) -> None:
    canvas.update_axis_scales(tuple(float(max(0.1, value)) for value in scale))
    canvas.update()


class DesktopVisCanvas(VisCanvas):
    KEY_GROUPS = {
        "1": ("layers", "层位"),
        "2": ("faults", "断层"),
        "3": ("por", "POR"),
        "4": ("lith", "LITH"),
        "5": ("perm", "PERM"),
        "n": ("well_names", "井名"),
    }

    def __init__(self, *args, groups: dict[str, list], cycle_volume, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfreeze()
        self.desktop_groups = groups
        self.desktop_cycle_volume = cycle_volume
        self.desktop_pick_state = {
            "enabled": False,
            "targets": {},
            "positions": {},
            "on_pick": None,
        }
        self.freeze()
        self.print_desktop_help()

    def print_desktop_help(self) -> None:
        print("Desktop controls:")
        print("  1: toggle layers, 2: toggle faults, 3: toggle POR, 4: toggle LITH, 5: toggle PERM")
        print("  n: toggle well names, v: switch slice volume, arrows: move active slice axis")
        print("  d: drag mode, Space: reset camera, s: save screenshot, a: print view state, h: print this help")

    def on_key_press(self, event):
        super().on_key_press(event)
        key = event.text
        if key in self.KEY_GROUPS:
            group_key, label = self.KEY_GROUPS[key]
            nodes = self.desktop_groups.get(group_key, [])
            next_visible = not node_group_visible(nodes)
            set_node_group_visible(nodes, next_visible)
            print(f"{label}: {'visible' if next_visible else 'hidden'}")
            self.update()
        elif key == "v":
            self.desktop_cycle_volume()
            self.update()
        elif key == "h":
            self.print_desktop_help()

    def configure_well_picking(
        self,
        *,
        targets: dict[object, str],
        positions: dict[str, tuple[float, float, float]],
        on_pick,
    ) -> None:
        self.desktop_pick_state["targets"] = dict(targets)
        self.desktop_pick_state["positions"] = dict(positions)
        self.desktop_pick_state["on_pick"] = on_pick

    def set_well_picking_enabled(self, enabled: bool) -> None:
        self.desktop_pick_state["enabled"] = bool(enabled)

    def on_mouse_press(self, event):
        if self._try_pick_well(event):
            return
        super().on_mouse_press(event)

    def _try_pick_well(self, event) -> bool:
        if not self.desktop_pick_state.get("enabled"):
            return False
        if getattr(event, "button", None) != 1:
            return False
        picked = self._pick_well_from_visual(event)
        if picked is None:
            picked = self._pick_well_from_screen_projection(event)
        if picked is None:
            return False
        on_pick = self.desktop_pick_state.get("on_pick")
        if on_pick is not None:
            on_pick(str(picked))
        return True

    def _pick_well_from_visual(self, event) -> str | None:
        targets = self.desktop_pick_state.get("targets") or {}
        if not targets:
            return None
        hover_on = None
        try:
            for view in self.view:
                view.interactive = False
            hover_on = self.visual_at(event.pos)
        except Exception:
            return None
        finally:
            for view in self.view:
                view.interactive = True
        while hover_on is not None:
            if hover_on in targets:
                return targets[hover_on]
            hover_on = getattr(hover_on, "parent", None)
        return None

    def _pick_well_from_screen_projection(self, event) -> str | None:
        positions = self.desktop_pick_state.get("positions") or {}
        if not positions or not getattr(self, "view", None):
            return None
        mouse = np.asarray(event.pos, dtype=float)
        best_name = None
        best_dist = float("inf")
        for name, position in positions.items():
            screen = self._world_to_canvas(position)
            if screen is None:
                continue
            dist = float(np.linalg.norm(screen[:2] - mouse[:2]))
            if dist < best_dist:
                best_name = name
                best_dist = dist
        return best_name if best_name is not None and best_dist <= 22.0 else None

    def _world_to_canvas(self, position: tuple[float, float, float]) -> np.ndarray | None:
        try:
            tr = self.scene.node_transform(self.view[0].scene)
            screen = tr.imap([float(position[0]), float(position[1]), float(position[2]), 1.0])
            screen = np.asarray(screen, dtype=float)
            if screen.shape[0] >= 4 and screen[3] != 0:
                screen = screen / screen[3]
            if not np.all(np.isfinite(screen[:2])):
                return None
            return screen
        except Exception:
            return None


class DesktopControlWindow(QtWidgets.QMainWindow):
    GROUP_LABELS = [
        ("layers", "显示层位"),
        ("faults", "显示断层"),
        ("por", "显示孔隙度(POR)"),
        ("lith", "显示岩性(LITH)"),
        ("perm", "显示渗透率(PERM)"),
        ("well_names", "显示井名"),
    ]

    def __init__(
        self,
        *,
        canvas: DesktopVisCanvas,
        groups: dict[str, list],
        initial_visibility: dict[str, bool],
        volume_order: list[str],
        current_volume: str,
        switch_volume,
        get_current_clim,
        slice_nodes: list,
        seismic_shape: tuple[int, int, int],
        slice_positions: tuple[int, int, int],
        section_config: dict[str, object] | None,
        label_screen_scale: float,
        label_z_offset: float,
        title: str,
    ) -> None:
        super().__init__()
        self.canvas = canvas
        self.groups = groups
        self.switch_volume = switch_volume
        self.get_current_clim = get_current_clim
        self.slice_nodes = slice_nodes
        self.section_config = section_config or {}
        self._syncing_clim = False
        self.setWindowTitle(title)
        self.resize(max(1080, int(canvas.size[0]) + 380), max(760, int(canvas.size[1])))
        self.setCentralWidget(canvas.native)

        dock = QtWidgets.QDockWidget("操作菜单", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        dock.setWidget(
            self._build_panel(
                volume_order,
                current_volume,
                initial_visibility,
                seismic_shape,
                slice_positions,
                label_screen_scale,
                label_z_offset,
            )
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        target_nodes = list(self.section_config.get("pick_target_nodes") or [])
        target_positions = dict(self.section_config.get("pick_target_positions") or {})
        if target_nodes and target_positions:
            self.canvas.configure_well_picking(
                targets={node: getattr(node, "well_name", "") for node in target_nodes},
                positions=target_positions,
                on_pick=self._add_section_well,
            )

    def _build_panel(
        self,
        volume_order: list[str],
        current_volume: str,
        initial_visibility: dict[str, bool],
        seismic_shape: tuple[int, int, int],
        slice_positions: tuple[int, int, int],
        label_screen_scale: float,
        label_z_offset: float,
    ) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        volume_box = QtWidgets.QGroupBox("体数据")
        volume_layout = QtWidgets.QVBoxLayout(volume_box)
        volume_combo = QtWidgets.QComboBox()
        for key in volume_order:
            volume_combo.addItem(VOLUME_OPTION_LABELS.get(key, key), key)
        current_idx = max(0, volume_order.index(current_volume) if current_volume in volume_order else 0)
        volume_combo.setCurrentIndex(current_idx)
        volume_combo.currentIndexChanged.connect(
            lambda idx: self._switch_volume_from_combo(volume_combo.itemData(idx))
        )
        volume_layout.addWidget(volume_combo)
        layout.addWidget(volume_box)

        params_box = QtWidgets.QGroupBox("paramters")
        params_layout = QtWidgets.QFormLayout(params_box)
        cmin, cmax = self.get_current_clim()
        self.clim_min_spin = QtWidgets.QDoubleSpinBox()
        self.clim_max_spin = QtWidgets.QDoubleSpinBox()
        for spin, value in [(self.clim_min_spin, cmin), (self.clim_max_spin, cmax)]:
            spin.setRange(-1.0e12, 1.0e12)
            spin.setDecimals(6)
            spin.setSingleStep(max(abs(cmax - cmin) / 100.0, 1.0e-6))
            spin.setValue(float(value))
            spin.valueChanged.connect(lambda _value: self._set_clim_from_spins())
        params_layout.addRow("clim min", self.clim_min_spin)
        params_layout.addRow("clim max", self.clim_max_spin)

        cmap_combo = QtWidgets.QComboBox()
        for cmap_name in ["pre-set", "gray", "seismic", "Petrel", "stratum", "jet", "bwp", "viridis", "turbo", "plasma"]:
            cmap_combo.addItem(cmap_name)
        cmap_combo.currentTextChanged.connect(self._set_cmap)
        params_layout.addRow("cmap", cmap_combo)

        self.scale_spins: list[QtWidgets.QDoubleSpinBox] = []
        scale_row = QtWidgets.QWidget()
        scale_layout = QtWidgets.QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        for label in ["x", "y", "z"]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.1, 20.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(1.0)
            spin.valueChanged.connect(lambda _value: self._set_scale_from_spins())
            self.scale_spins.append(spin)
            scale_layout.addWidget(QtWidgets.QLabel(label))
            scale_layout.addWidget(spin)
        params_layout.addRow("scale", scale_row)
        layout.addWidget(params_box)

        visible_box = QtWidgets.QGroupBox("显示内容")
        visible_layout = QtWidgets.QVBoxLayout(visible_box)
        for group_key, label in self.GROUP_LABELS:
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setEnabled(bool(self.groups.get(group_key)))
            checkbox.setChecked(bool(initial_visibility.get(group_key, False)))
            checkbox.toggled.connect(lambda checked, key=group_key: self._set_group_visible(key, checked))
            visible_layout.addWidget(checkbox)
        label_size_spin = QtWidgets.QDoubleSpinBox()
        label_size_spin.setRange(0.2, 4.0)
        label_size_spin.setDecimals(1)
        label_size_spin.setSingleStep(0.1)
        label_size_spin.setValue(float(label_screen_scale))
        label_size_spin.valueChanged.connect(lambda value: self._set_label_size(float(value)))
        visible_layout.addWidget(QtWidgets.QLabel("井名字号"))
        visible_layout.addWidget(label_size_spin)
        label_offset_spin = QtWidgets.QDoubleSpinBox()
        label_offset_spin.setRange(-500.0, 500.0)
        label_offset_spin.setDecimals(1)
        label_offset_spin.setSingleStep(1.0)
        label_offset_spin.setValue(float(label_z_offset))
        label_offset_spin.valueChanged.connect(lambda value: self._set_label_offset(float(value)))
        visible_layout.addWidget(QtWidgets.QLabel("井名 Z 偏移"))
        visible_layout.addWidget(label_offset_spin)
        layout.addWidget(visible_box)

        slices_box = QtWidgets.QGroupBox("切片位置")
        slices_layout = QtWidgets.QVBoxLayout(slices_box)
        for axis, axis_label, maximum, initial in [
            ("x", "Inline / X", seismic_shape[0] - 1, slice_positions[0]),
            ("y", "Crossline / Y", seismic_shape[1] - 1, slice_positions[1]),
            ("z", "Depth / Z", seismic_shape[2] - 1, slice_positions[2]),
        ]:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            value_label = QtWidgets.QLabel(f"{axis_label}: {initial}")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, int(maximum))
            slider.setValue(int(initial))
            slider.valueChanged.connect(
                lambda value, ax=axis, lab=value_label, prefix=axis_label: self._set_slice_pos(ax, value, lab, prefix)
            )
            row_layout.addWidget(value_label)
            row_layout.addWidget(slider)
            slices_layout.addWidget(row)
        layout.addWidget(slices_box)

        screenshot_box = QtWidgets.QGroupBox("screenshot")
        screenshot_layout = QtWidgets.QVBoxLayout(screenshot_box)
        screenshot_button = QtWidgets.QPushButton("Render and save PNG")
        screenshot_button.clicked.connect(self._save_screenshot)
        screenshot_layout.addWidget(screenshot_button)
        layout.addWidget(screenshot_box)

        states_box = QtWidgets.QGroupBox("states")
        states_layout = QtWidgets.QVBoxLayout(states_box)
        state_button = QtWidgets.QPushButton("print states")
        state_button.clicked.connect(self._print_states)
        states_layout.addWidget(state_button)
        layout.addWidget(states_box)

        section_box = self._build_section_panel()
        if section_box is not None:
            layout.addWidget(section_box)

        help_box = QtWidgets.QGroupBox("快捷键")
        help_layout = QtWidgets.QVBoxLayout(help_box)
        help_label = QtWidgets.QLabel(
            "1 层位  2 断层  3 POR\n"
            "4 LITH  5 PERM  n 井名\n"
            "v 切换体数据  Space 重置视角\n"
            "方向键移动当前切片轴，s 截图"
        )
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        layout.addWidget(help_box)

        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _switch_volume_from_combo(self, key: str) -> None:
        if key:
            self.switch_volume(str(key))
            self._refresh_clim_widgets()
            self.canvas.update()

    def _refresh_clim_widgets(self) -> None:
        cmin, cmax = self.get_current_clim()
        self._syncing_clim = True
        try:
            self.clim_min_spin.setValue(float(cmin))
            self.clim_max_spin.setValue(float(cmax))
        finally:
            self._syncing_clim = False

    def _set_clim_from_spins(self) -> None:
        if self._syncing_clim:
            return
        cmin = float(self.clim_min_spin.value())
        cmax = float(self.clim_max_spin.value())
        if cmax <= cmin:
            return
        update_slice_clim_desktop(self.slice_nodes, [cmin, cmax])
        self.canvas.update()

    def _set_cmap(self, cmap_name: str) -> None:
        update_slice_cmap_desktop(self.slice_nodes, str(cmap_name))
        self.canvas.update()

    def _set_scale_from_spins(self) -> None:
        update_scene_scale_desktop(self.canvas, tuple(spin.value() for spin in self.scale_spins))

    def _set_group_visible(self, group_key: str, visible: bool) -> None:
        set_node_group_visible(self.groups.get(group_key, []), bool(visible))
        self.canvas.update()

    def _set_label_size(self, font_screen_scale: float) -> None:
        font_size = max(24.0, 40.0 * float(font_screen_scale))
        for node in self.groups.get("well_names", []):
            if hasattr(node, "font_size"):
                node.font_size = font_size
        self.canvas.update()

    def _set_label_offset(self, z_offset: float) -> None:
        for node in self.groups.get("well_names", []):
            base_position = getattr(node, "base_position", None)
            if base_position is None:
                continue
            x, y, z = base_position
            node.pos = (float(x), float(y), float(z) - float(z_offset))
        self.canvas.update()

    def _set_slice_pos(self, axis: str, value: int, label: QtWidgets.QLabel, prefix: str) -> None:
        label.setText(f"{prefix}: {int(value)}")
        for node in self.slice_nodes:
            if isinstance(node, AxisAlignedImage) and node.axis == axis:
                node._update_location(int(value))
        self.canvas.update()

    def _save_screenshot(self) -> None:
        output_dir = VISUAL_ROOT / "输出" / "desktop_screenshots"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"desktop_view_{QtCore.QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.png"
        image = self.canvas.render()
        from vispy import io as vispy_io

        vispy_io.write_png(str(path), image)
        print(f"[desktop] screenshot saved: {path}")

    def _print_states(self) -> None:
        print("")
        print("----------- Current States ------------")
        if getattr(self.canvas, "view", None):
            camera = self.canvas.view[0].camera
            print(f"fov: {camera.fov}")
            print(f"center: {camera.center}")
            print(f"scale_factor: {camera.scale_factor}")
            print(f"azimuth: {camera.azimuth}")
            print(f"elevation: {camera.elevation}")
            print(f"axis scale: {getattr(camera, '_flip_factors', None)}")
        print("----------- axis position -------------")
        positions = {"x": [], "y": [], "z": []}
        for node in self.slice_nodes:
            if isinstance(node, AxisAlignedImage):
                positions[node.axis].append(int(node.pos))
        print(f"x: {positions['x']}, y: {positions['y']}, z: {positions['z']}")
        print("----------- parameters ----------------")
        print(f"clim: {[self.clim_min_spin.value(), self.clim_max_spin.value()]}")
        print(f"scale: {[spin.value() for spin in self.scale_spins]}")
        print("")

    def _build_section_panel(self) -> QtWidgets.QGroupBox | None:
        wells = list(self.section_config.get("available_wells") or [])
        if len(wells) < 2:
            return None
        box = QtWidgets.QGroupBox("井剖面")
        layout = QtWidgets.QVBoxLayout(box)

        self.section_selected: list[str] = []
        self.section_selected_markers: list = []
        self.section_target_nodes = list(self.section_config.get("pick_target_nodes") or [])

        start_button = QtWidgets.QPushButton("开始选井")
        stop_button = QtWidgets.QPushButton("停止选井")
        start_button.clicked.connect(self._start_section_picking)
        stop_button.clicked.connect(self._stop_section_picking)
        layout.addWidget(start_button)
        layout.addWidget(stop_button)

        selected_row = QtWidgets.QFormLayout()
        self.section_selected_text = QtWidgets.QLineEdit("（无）")
        self.section_selected_text.setReadOnly(True)
        selected_row.addRow("已选井", self.section_selected_text)
        layout.addLayout(selected_row)

        self.section_status = QtWidgets.QLineEdit("请先开始选井，然后点击黄色井口标记。")
        self.section_status.setReadOnly(True)
        status_row = QtWidgets.QFormLayout()
        status_row.addRow("状态", self.section_status)
        layout.addLayout(status_row)

        style_form = QtWidgets.QFormLayout()
        self.section_por_style = QtWidgets.QComboBox()
        self.section_por_style.addItems(list(POR_STYLES))
        self.section_por_style.setCurrentText("条形")
        style_form.addRow("孔隙度显示方式", self.section_por_style)

        self.section_seismic_cmap = QtWidgets.QComboBox()
        self.section_seismic_cmap.addItems(list(SEISMIC_COLOR_OPTIONS.keys()))
        self.section_seismic_cmap.setCurrentText("RdBu")
        style_form.addRow("地震体配色", self.section_seismic_cmap)

        self.section_seismic_display = QtWidgets.QComboBox()
        self.section_seismic_display.addItems(list(SEISMIC_DISPLAY_OPTIONS))
        self.section_seismic_display.setCurrentText(_seismic_display_setting("color"))
        style_form.addRow("地震体显示方式", self.section_seismic_display)
        layout.addLayout(style_form)

        remove_button = QtWidgets.QPushButton("移除最后一个")
        clear_button = QtWidgets.QPushButton("清空")
        generate_button = QtWidgets.QPushButton("生成剖面 HTML")
        remove_button.clicked.connect(self._remove_last_section_well)
        clear_button.clicked.connect(self._clear_section_wells)
        generate_button.clicked.connect(self._generate_section_html)
        layout.addWidget(remove_button)
        layout.addWidget(clear_button)
        layout.addWidget(generate_button)
        self.section_link = QtWidgets.QLabel("")
        self.section_link.setOpenExternalLinks(True)
        self.section_link.setWordWrap(True)
        layout.addWidget(self.section_link)
        if not self.section_target_nodes:
            self.section_status.setText("未找到可点击井口标记。")
            start_button.setEnabled(False)
        return box

    def _selected_section_text(self) -> str:
        return " -> ".join(self.section_selected) if self.section_selected else "（无）"

    def _set_section_status(self, message: str) -> None:
        self.section_selected_text.setText(self._selected_section_text())
        self.section_status.setText(message)

    def _set_section_targets_visible(self, visible: bool) -> None:
        for node in getattr(self, "section_target_nodes", []):
            node.visible = bool(visible)
        self.canvas.update()

    def _start_section_picking(self) -> None:
        self.canvas.set_well_picking_enabled(True)
        self._set_section_targets_visible(True)
        self._set_section_status("已开启选井。请按剖面顺序点击黄色井口标记。")

    def _stop_section_picking(self) -> None:
        self.canvas.set_well_picking_enabled(False)
        self._set_section_targets_visible(False)
        self._set_section_status("已停止选井。")

    def _add_section_well(self, well_name: str) -> None:
        if not well_name:
            return
        if well_name in self.section_selected:
            self._set_section_status(f"{well_name} 已经选中过了。")
            return
        self.section_selected.append(str(well_name))
        self._set_section_status(f"已添加 {well_name}。")
        self._update_section_selected_markers()

    def _remove_last_section_well(self) -> None:
        if self.section_selected:
            removed = self.section_selected.pop()
            self._set_section_status(f"已移除 {removed}。")
            self._update_section_selected_markers()
        else:
            self._set_section_status("当前没有可移除的已选井。")

    def _clear_section_wells(self) -> None:
        self.section_selected.clear()
        self._set_section_status("已清空选择。")
        self._update_section_selected_markers()

    def _update_section_selected_markers(self) -> None:
        for node in self.section_selected_markers:
            node.parent = None
        self.section_selected_markers.clear()
        positions = dict(self.section_config.get("pick_target_positions") or {})
        view = self.canvas.view[0] if getattr(self.canvas, "view", None) else None
        if view is None:
            return
        for idx, name in enumerate(self.section_selected, start=1):
            position = positions.get(name)
            if position is None:
                continue
            point_nodes = vispyplot.create_points(
                np.asarray([position], dtype=np.float32),
                r=7.0,
                color="#ff3c28",
                shading="flat",
            )
            if point_nodes:
                marker = point_nodes[0]
                view.add(marker)
                self.section_selected_markers.append(marker)
            x, y, z = position
            label = Text(
                str(idx),
                pos=(float(x), float(y), float(z) - 9.0),
                color=(1.0, 0.08, 0.02, 1.0),
                font_size=44,
                anchor_x="center",
                anchor_y="bottom",
                depth_test=False,
            )
            label.set_gl_state(depth_test=False, blend=True)
            view.add(label)
            self.section_selected_markers.append(label)
        self.canvas.update()

    def _generate_section_html(self) -> None:
        selected = list(getattr(self, "section_selected", []))
        if len(selected) < 2:
            self._set_section_status("请至少先选择两口井。")
            return
        try:
            path = build_well_section_html(
                selected,
                mode=str(self.section_config["mode"]),
                coords_csv=self.section_config["coords_csv"],
                log_dir=self.section_config["log_dir"],
                layer_dir=self.section_config["layer_dir"],
                output_dir=self.section_config["output_dir"],
                seismic_path=self.section_config["seismic_path"],
                z_count=int(self.section_config["z_count"]),
                fault_dir=self.section_config["fault_dir"],
                por_dir=self.section_config["por_dir"],
                perm_dir=self.section_config["perm_dir"],
                por_style=str(self.section_por_style.currentText()),
                seismic_colorscale=str(self.section_seismic_cmap.currentText()),
                seismic_display=str(self.section_seismic_display.currentText()),
            )
        except Exception as exc:
            self._set_section_status(f"生成失败：{exc}")
            print(f"[井剖面] 生成失败：{exc}")
            return
        self._set_section_status(f"已打开：{path.name}")
        self.section_link.setText(f'<a href="{path.resolve().as_uri()}">打开已生成剖面</a>')
        print(f"[井剖面] 已保存：{path}")
        webbrowser.open(path.resolve().as_uri(), new=2)


def load_available_volume_specs(
    seismic_path: Path,
    attribute_dir: Path,
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
    return styles, notes


def load_volume_by_key(volume_key: str, volume_styles: dict[str, dict]) -> np.ndarray:
    spec = volume_styles.get(volume_key)
    if spec is None:
        raise KeyError(volume_key)
    return np.load(Path(spec["path"]), mmap_mode="r")


def main() -> None:
    args = parse_args()
    seismic_path = ensure_inside_bundle(args.seismic_npy, "seismic_npy")
    attribute_dir = ensure_inside_bundle(args.attribute_dir, "attribute_dir")
    coords_csv = ensure_inside_bundle(args.coords_csv, "coords_csv")
    por_dir = ensure_inside_bundle(args.por_dir, "por_dir")
    perm_dir = ensure_inside_bundle(args.perm_dir, "perm_dir")
    lith_root = ensure_inside_bundle(args.lith_root, "lith_root")
    layer_dir = ensure_inside_bundle(args.layer_dir, "layer_dir")
    fault_dir = ensure_inside_bundle(args.fault_dir, "fault_dir")
    section_output_dir = ensure_inside_bundle(args.section_output_dir, "section_output_dir")
    lith_dir = ensure_inside_bundle(lith_root / args.variant, "lith_dir")
    if not lith_dir.exists():
        raise SystemExit(f"Missing lith variant directory: {lith_dir}")

    volume_styles, volume_notes = load_available_volume_specs(seismic_path, attribute_dir)
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
    slice_nodes = vispyplot.create_slices(
        initial_volume_data,
        pos={"x": [x_idx], "y": [y_idx], "z": [z_idx]},
        clim=initial_clim,
        cmap=str(volume_styles[initial_volume]["cmap"]),
        texture_format="auto",
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
        node = make_uniform_surface_node(
            surface,
            color=rgba,
            step=args.surface_step,
            name=layer["name"],
        )
        if node is None:
            continue
        surface_nodes.append(node)
        surface_node_names.append(layer["name"])
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
        por_nodes = create_desktop_log_point_nodes(
            por_logs,
            cmap="viridis",
            clim=por_clim,
            radius=args.por_width,
            names=por_wells,
        )
        set_node_group_name(por_nodes, "井数据/孔隙度", por_wells)

    lith_nodes = []
    if lith_logs:
        style = LITH_STYLE[args.variant]
        lith_nodes = create_desktop_log_point_nodes(
            lith_logs,
            cmap=style["cmap"],
            clim=style["clim"],
            radius=args.lith_width,
            names=lith_wells,
        )
        set_node_group_name(lith_nodes, f"井数据/岩性_{args.variant}", lith_wells)

    perm_nodes = []
    perm_clim = None
    if perm_logs:
        perm_values = np.concatenate([log[:, 3] for log in perm_logs])
        perm_clim = [float(np.nanpercentile(perm_values, 2.0)), float(np.nanpercentile(perm_values, 98.0))]
        perm_nodes = create_desktop_log_point_nodes(
            perm_logs,
            cmap="plasma",
            clim=perm_clim,
            radius=args.por_width,
            names=perm_wells,
        )
        set_node_group_name(perm_nodes, "井数据/渗透率", perm_wells)

    fault_nodes, fault_descriptions = load_fault_meshes(
        fault_dir=fault_dir,
        target=str(args.fault),
        z_count=seismic.shape[2],
        alpha=float(args.fault_alpha),
    )

    set_node_group_name(surface_nodes, "层位", surface_node_names)
    set_node_group_name(fault_nodes, "断层")

    head_positions = dict(lith_heads)
    head_positions.update(por_heads)
    head_positions.update(perm_heads)
    label_nodes = []
    if args.show_well_names and head_positions:
        label_nodes = create_desktop_well_name_labels(
            head_positions,
            z_offset=float(args.label_z_offset),
            font_screen_scale=float(args.label_screen_scale),
        )

    initial_visibility = {
        "layers": bool(args.show_layers),
        "faults": bool(args.show_faults),
        "por": bool(args.show_por),
        "lith": bool(args.show_lith),
        "perm": bool(args.show_perm),
        "well_names": False,
    }
    groups = {
        "layers": surface_nodes,
        "faults": fault_nodes,
        "por": por_nodes,
        "lith": lith_nodes,
        "perm": perm_nodes,
        "well_names": label_nodes,
    }
    for group_name, nodes in groups.items():
        set_node_group_visible(nodes, initial_visibility.get(group_name, False))

    volume_order = list(volume_styles.keys())
    volume_state = {
        "current_key": initial_volume,
        "current_volume": initial_volume_data,
        "current_clim": list(initial_clim),
    }

    def get_current_clim() -> list[float]:
        return list(volume_state["current_clim"])

    def current_slice_positions() -> tuple[int, int, int]:
        positions = {"x": x_idx, "y": y_idx, "z": z_idx}
        for node in slice_nodes:
            if isinstance(node, AxisAlignedImage):
                positions[node.axis] = int(node.pos)
        return positions["x"], positions["y"], positions["z"]

    def switch_volume(selected: str) -> None:
        if not selected or selected == volume_state["current_key"]:
            return
        next_volume = load_volume_by_key(selected, volume_styles)
        if tuple(next_volume.shape) != tuple(seismic.shape):
            print(f"Skip volume switch: {selected} shape {tuple(next_volume.shape)} != seismic {tuple(seismic.shape)}")
            return
        cx, cy, cz = current_slice_positions()
        next_clim = estimate_volume_clim(selected, next_volume, cx, cy, cz)
        update_slice_volume_desktop(
            slice_nodes=slice_nodes,
            volume=next_volume,
            clim=next_clim,
            cmap=str(volume_styles[selected]["cmap"]),
        )
        volume_state["current_key"] = selected
        volume_state["current_volume"] = next_volume
        volume_state["current_clim"] = list(next_clim)
        print(
            f"Switched 3D volume to {selected}, path={volume_styles[selected]['path'].name}, "
            f"clim={next_clim}, cmap={volume_styles[selected]['cmap']}"
        )

    def cycle_volume() -> None:
        if not volume_order:
            return
        current_idx = volume_order.index(volume_state["current_key"])
        switch_volume(volume_order[(current_idx + 1) % len(volume_order)])

    print(f"Starting desktop seismic + horizons + POR/LITH/PERM view, display={args.display}, seismic_shape={tuple(seismic.shape)}")
    print(f"Using cigvis source: {Path(cigvis.__file__).resolve()}")
    print(f"Primary POR column: {POR_COLUMN}")
    print(f"Using slice positions x={x_idx}, y={y_idx}, z={z_idx}")
    print(f"Initial 3D volume: {initial_volume}, clim={initial_clim}, cmap={volume_styles[initial_volume]['cmap']}")
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
    if args.show_well_names and head_positions:
        print(
            f"Well name labels available: count={len(head_positions)}, press n to toggle, "
            f"font_screen_scale={float(args.label_screen_scale):.2f}, "
            f"z_offset={float(args.label_z_offset):.2f}"
        )
    section_mode = None
    section_wells = []
    section_log_dir = None
    section_head_positions: dict[str, tuple[float, float, float]] = {}
    if lith_wells:
        section_mode = "lith"
        section_wells = lith_wells
        section_log_dir = lith_dir
        section_head_positions = lith_heads
    elif por_wells:
        section_mode = "por"
        section_wells = por_wells
        section_log_dir = por_dir
        section_head_positions = por_heads
    elif perm_wells:
        section_mode = "por"
        section_wells = perm_wells
        section_log_dir = perm_dir
        section_head_positions = perm_heads
    section_config = None
    pick_target_nodes = []
    pick_target_positions: dict[str, tuple[float, float, float]] = {}
    if section_mode is not None and section_log_dir is not None:
        pick_target_nodes, pick_target_positions = create_desktop_well_pick_target_nodes(
            section_wells,
            section_head_positions,
        )
        section_config = {
            "mode": section_mode,
            "available_wells": section_wells,
            "head_positions": section_head_positions,
            "pick_target_nodes": pick_target_nodes,
            "pick_target_positions": pick_target_positions,
            "coords_csv": coords_csv,
            "log_dir": section_log_dir,
            "layer_dir": layer_dir,
            "output_dir": section_output_dir,
            "seismic_path": seismic_path,
            "z_count": int(seismic.shape[2]),
            "fault_dir": fault_dir,
            "por_dir": por_dir,
            "perm_dir": perm_dir,
        }
    print("Opening VisPy desktop window with operation menu. Close the window or press Ctrl+C in this terminal to stop.")

    qt_backend = use_app("pyqt5")
    qt_backend.create()
    all_nodes = slice_nodes + surface_nodes + fault_nodes + por_nodes + lith_nodes + perm_nodes + label_nodes + pick_target_nodes
    canvas = DesktopVisCanvas(
        visual_nodes=all_nodes,
        groups=groups,
        cycle_volume=cycle_volume,
        size=tuple(int(v) for v in args.desktop_size),
        title=f"地震-孔隙度-岩性-渗透率桌面端-{args.variant}",
        dyn_light=True,
    )
    window = DesktopControlWindow(
        canvas=canvas,
        groups=groups,
        initial_visibility=initial_visibility,
        volume_order=volume_order,
        current_volume=initial_volume,
        switch_volume=switch_volume,
        get_current_clim=get_current_clim,
        slice_nodes=slice_nodes,
        seismic_shape=tuple(int(v) for v in seismic.shape),
        slice_positions=(x_idx, y_idx, z_idx),
        section_config=section_config,
        label_screen_scale=float(args.label_screen_scale),
        label_z_offset=float(args.label_z_offset),
        title=f"地震-孔隙度-岩性-渗透率桌面端-{args.variant}",
    )
    window.show()
    try:
        qt_backend.run()
    except KeyboardInterrupt:
        print("Execution interrupted")


if __name__ == "__main__":
    main()
