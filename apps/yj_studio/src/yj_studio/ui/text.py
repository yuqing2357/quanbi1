from __future__ import annotations

LAYER_KIND_LABELS = {
    "volume": "体数据",
    "arbitrary_section": "任意剖面",
    "horizon": "层位",
    "horizon_stick": "层位杆",
    "fault_surface": "断层面",
    "fault_stick": "断层杆",
    "well": "井",
    "well_log": "测井曲线",
    "lith_body": "岩性体",
    "mask": "掩膜",
    "polygon": "多边形",
    "annotation": "标注",
    "measurement": "测量",
    "trap": "圈闭",
}

SECTION_AXIS_LABELS = {
    "inline": "纵向剖面",
    "xline": "横向剖面",
    "z": "Z向剖面",
    "arbitrary": "任意剖面",
    "well": "井剖面",
}

ALGORITHM_CATEGORY_LABELS = {
    "horizon": "层位",
    "fault": "断层",
    "reservoir": "储层",
    "trap": "圈闭",
    "measure": "测量",
    "ai": "AI",
}

PARAM_ROLE_LABELS = {
    "volume": "体数据",
    "seed": "种子",
    "seed_mask": "种子掩膜",
    "edited_mask": "编辑后掩膜",
    "top": "顶部",
    "bottom": "底部",
    "path": "路径",
    "polygon": "多边形",
    "horizon": "层位",
    "mask": "掩膜",
    "trap": "圈闭",
}

PARAM_NAME_LABELS = {
    "axis": "轴",
    "slice_index": "剖面索引",
    "text_prompt": "文本提示",
    "boxes": "框提示",
    "points": "点提示",
    "point_box_radius_px": "点框半宽",
    "confidence_threshold": "置信度阈值",
    "keep_top_k": "保留前 K",
    "name_prefix": "名称前缀",
    "forward_steps": "向前步数",
    "backward_steps": "向后步数",
    "drop_low_confidence_frames": "低置信度停止",
    "text_prompt_override": "文本提示覆盖",
    "pad_box_px": "外扩像素",
    "depth_step_m": "每采样米数",
    "name": "名称",
    "structural_only": "仅结构",
    "score_threshold": "分数阈值",
    "use_well_data": "使用井数据",
    "risk_weight": "风险权重",
    "neighbourhood": "邻域",
    "min_component_size": "最小连通块",
    "similarity_tolerance": "相似性容差",
    "max_cells": "最大单元数",
    "min_thickness_m": "最小厚度",
    "amplitude_threshold": "振幅阈值",
    "seed_layer_role": "种子角色",
    "similarity_window": "相似性窗口",
    "max_iterations": "最大迭代次数",
    "attribute_threshold": "属性阈值",
    "min_surface_size": "最小表面大小",
    "contour_interval_m": "等值线间隔",
    "min_closure_area": "最小闭合面积",
    "smoothness_weight": "平滑权重",
    "max_extent": "最大范围",
}

MEASUREMENT_VALUE_LABELS = {
    "mean_m": "平均值",
    "min_m": "最小值",
    "max_m": "最大值",
    "std_m": "标准差",
    "coverage": "覆盖率",
    "valid_cells": "有效单元",
    "total_xy": "平面长度",
    "total_3d_m": "三维长度",
    "segment_count": "线段数",
    "area_xy": "面积",
    "perimeter_xy": "周长",
    "distance": "距离",
    "thickness": "厚度",
}

AI_STATE_LABELS = {
    "idle": "空闲",
    "loading": "加载中",
    "ready": "就绪",
    "busy": "运行中",
    "error": "错误",
}

WELL_DISPLAY_MODE_LABELS = {
    "none": "仅井轨迹",
    "lith": "岩性",
    "por": "孔隙度",
    "perm": "渗透率",
}


def layer_kind_label(kind: str) -> str:
    return LAYER_KIND_LABELS.get(kind, kind)


def algorithm_category_label(category: str) -> str:
    return ALGORITHM_CATEGORY_LABELS.get(category, category)


def section_axis_label(axis: str) -> str:
    return SECTION_AXIS_LABELS.get(axis, axis)


def parameter_role_label(role: str) -> str:
    return PARAM_ROLE_LABELS.get(role, role)


def parameter_name_label(name: str) -> str:
    return PARAM_NAME_LABELS.get(name, name)


def measurement_value_label(key: str) -> str:
    return MEASUREMENT_VALUE_LABELS.get(key, key)


def ai_state_label(state: str) -> str:
    return AI_STATE_LABELS.get(state, state)


def well_display_mode_label(mode: str) -> str:
    return WELL_DISPLAY_MODE_LABELS.get(mode, mode)
