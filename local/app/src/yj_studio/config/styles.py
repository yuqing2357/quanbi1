from __future__ import annotations

def _configure_matplotlib_fonts() -> None:
    try:
        import matplotlib as mpl
    except Exception:  # pragma: no cover - optional runtime dependency
        return
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()

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
        "cmap": ["#8f8f8f", "#f2c84b"],
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

LITH_COLORS = {
    0: "#8f8f8f",
    1: "#f2c84b",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#17becf",
}

LITH_BODY_STYLE = {
    0: {"name": "砾", "slug": "gravel", "color": (0, 220, 220)},
    1: {"name": "砂岩", "slug": "sandstone", "color": (245, 214, 45)},
    2: {"name": "泥巴", "slug": "mud", "color": (150, 150, 150)},
}

VOLUME_DISPLAY_STYLE = {
    "seismic": {"filename": "seismic.npy", "label": "地震体数据", "cmap": "Petrel"},
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
        "filename": "lithology_binary_3x_uint8.npy",
        "label": "岩性模型",
        "cmap": "tab10",
    },
    "model_porosity": {
        "filename": "porosity_3x_float16.npy",
        "label": "孔隙度模型",
        "cmap": "viridis",
    },
}


def rgba_float_to_uint8(rgba: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(max(0.0, min(1.0, channel)) * 255) for channel in rgba)
