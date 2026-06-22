"""User-facing entry point for SAM3.

Layout (top-down):

  ┌ Service status banner + Start/Stop button ─────┐
  ├ Current slice (axis + index, from active view) │
  ├ Text prompt (free-form) ────────────────────────┤
  ├ Geometric prompts list (boxes + points) ───────┤
  │  [Pick Box] [Pick Point] [Clear]                │
  ├ Run options:                                    │
  │   confidence_threshold / keep_top_k             │
  ├ [Run SAM3 Segment] ─────────────────────────────┤
  ├ Progress bar + status text ─────────────────────┤
  └ Last result summary ────────────────────────────┘

The dock owns no model state. It forwards Run clicks to the remote SAM3 job
client through ``AlgorithmRunner.submit(RemoteSAM3SegmentAlgorithm, ...)``; the
runner then submits ``/sam3/jobs`` and turns returned candidates into
``MaskLayer`` instances. Ctrl+Z still removes the local visual masks, while
the server-side GeoTargets stay in the target store.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QUndoStack
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yj_studio.ai.state import AIServiceState
from yj_studio.algorithms.runner import (
    AlgorithmRunner,
    RemoteSAM3TemplateMatchTask,
    RemoteSAM3TrackTask,
)
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import MaskLayer, VolumeLayer
from yj_studio.scene.undo_commands import AddLayerCommand, RemoveLayerCommand
from yj_studio.tools.tool_manager import ToolManager
from yj_studio_core.targets import BUILTIN_TARGET_TYPES
from yj_studio.ui.text import ai_state_label, section_axis_label

logger = logging.getLogger(__name__)

_AXES = ("inline", "xline", "z")


class AIDock(QDockWidget):
    job_finished = pyqtSignal(dict)
    track_finished = pyqtSignal(dict)

    def __init__(
        self,
        layer_store: LayerStore,
        ai_service: Any,
        runner: AlgorithmRunner,
        tool_manager: ToolManager,
        undo_stack: QUndoStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("AI 助手", parent)
        self._layer_store = layer_store
        self._ai_service = ai_service
        self._runner = runner
        self._tool_manager = tool_manager
        self._undo_stack = undo_stack
        self._boxes: list[tuple[float, float, float, float]] = []
        self._points: list[tuple[float, float]] = []
        self._template: list[list[float]] | None = None
        self._current_task: Any = None

        self._build_ui()
        ai_service.state_changed.connect(self._on_service_state)
        ai_service.box_prompt_added.connect(self._on_box_prompt)
        ai_service.point_prompt_added.connect(self._on_point_prompt)
        self._refresh_state_label(ai_service.state, ai_service.message)

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(6, 6, 6, 6)

        # Service status row
        status_row = QHBoxLayout()
        self._status_label = QLabel("AI：空闲", body)
        self._start_button = QPushButton("启动 AI", body)
        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button = QPushButton("卸载", body)
        self._stop_button.clicked.connect(self._ai_service.shutdown)
        self._stop_button.setEnabled(False)
        status_row.addWidget(self._status_label, 1)
        status_row.addWidget(self._start_button)
        status_row.addWidget(self._stop_button)
        outer.addLayout(status_row)

        # Slice selector
        form = QFormLayout()
        self._axis_combo = QComboBox(body)
        for axis in _AXES:
            self._axis_combo.addItem(section_axis_label(axis), axis)
        form.addRow("轴", self._axis_combo)
        self._slice_spin = QSpinBox(body)
        self._slice_spin.setRange(0, 9999)
        form.addRow("剖面索引", self._slice_spin)

        sync_button = QPushButton("← 使用当前剖面", body)
        sync_button.clicked.connect(self._sync_from_volume)
        form.addRow("", sync_button)
        outer.addLayout(form)

        # Text prompt
        outer.addWidget(QLabel("文本提示：", body))
        self._text_edit = QTextEdit(body)
        self._text_edit.setPlaceholderText(
            "例如：盐丘、河道砂体、断层带"
        )
        self._text_edit.setFixedHeight(60)
        outer.addWidget(self._text_edit)

        # Geometric prompts
        outer.addWidget(QLabel("几何提示：", body))
        self._prompts_list = QListWidget(body)
        self._prompts_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        outer.addWidget(self._prompts_list, 1)

        prompt_buttons = QHBoxLayout()
        self._pick_box_button = QPushButton("选框", body)
        self._pick_box_button.clicked.connect(self._activate_box_tool)
        self._pick_point_button = QPushButton("选点", body)
        self._pick_point_button.clicked.connect(self._activate_point_tool)
        self._clear_button = QPushButton("清空", body)
        self._clear_button.clicked.connect(self._clear_prompts)
        prompt_buttons.addWidget(self._pick_box_button)
        prompt_buttons.addWidget(self._pick_point_button)
        prompt_buttons.addWidget(self._clear_button)
        outer.addLayout(prompt_buttons)

        # Run options
        run_form = QFormLayout()
        self._confidence_spin = QDoubleSpinBox(body)
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setValue(0.4)
        run_form.addRow("置信度", self._confidence_spin)
        self._top_k_spin = QSpinBox(body)
        self._top_k_spin.setRange(1, 50)
        self._top_k_spin.setValue(3)
        run_form.addRow("保留前 K", self._top_k_spin)
        # When checked, the segmentation result is restricted to the single
        # detection that best fills each selection box (feature: 只保留框内目标).
        # This supersedes "保留前 K", which is only used in the unchecked /
        # text-only case, so the spin is disabled while strict mode is on.
        self._box_strict_check = QCheckBox("仅保留框内目标", body)
        self._box_strict_check.setChecked(True)
        self._box_strict_check.toggled.connect(self._on_box_strict_toggled)
        run_form.addRow("", self._box_strict_check)
        self._target_type_combo = QComboBox(body)
        self._target_type_combo.setEditable(True)
        self._target_type_combo.addItems(list(BUILTIN_TARGET_TYPES))
        self._target_type_combo.setCurrentText("unknown")
        run_form.addRow("目标类型", self._target_type_combo)
        self._track_back_spin = QSpinBox(body)
        self._track_back_spin.setRange(0, 5000)
        self._track_back_spin.setValue(20)
        run_form.addRow("种子前帧数", self._track_back_spin)
        self._track_fwd_spin = QSpinBox(body)
        self._track_fwd_spin.setRange(0, 5000)
        self._track_fwd_spin.setValue(20)
        run_form.addRow("种子后帧数", self._track_fwd_spin)
        # Auto range: ignore the manual front/back counts and let the server
        # propagate until the target disappears, then lock the valid extent.
        self._auto_range_check = QCheckBox("自动判断帧数", body)
        self._auto_range_check.toggled.connect(self._on_auto_range_toggled)
        run_form.addRow("", self._auto_range_check)
        outer.addLayout(run_form)

        # Run row
        run_row = QHBoxLayout()
        self._run_button = QPushButton("运行 SAM3 分割", body)
        self._run_button.clicked.connect(self._on_run_clicked)
        self._track_button = QPushButton("追踪", body)
        self._track_button.clicked.connect(self._on_track_clicked)
        self._cancel_button = QPushButton("取消", body)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        run_row.addWidget(self._run_button)
        run_row.addWidget(self._track_button)
        run_row.addWidget(self._cancel_button)
        outer.addLayout(run_row)

        # Wipe every SAM3 mask visualisation from the scene without restarting,
        # so a fresh box selection starts from a clean slate.
        clear_row = QHBoxLayout()
        self._clear_masks_button = QPushButton("清空所有 Mask 显示", body)
        self._clear_masks_button.clicked.connect(self._clear_all_masks)
        clear_row.addWidget(self._clear_masks_button)
        outer.addLayout(clear_row)

        # Morphology template search (形态模板识别): an independent module from the
        # point/box/text prompts. The user sketches an abstract shape on a blank
        # whiteboard (NOT on the section), and the server searches a RESERVOIR
        # MODEL slice for structures of similar morphology (size-independent).
        outer.addWidget(QLabel("形态模板识别（储层模型）：", body))
        template_form = QFormLayout()
        self._template_volume_combo = QComboBox(body)
        template_form.addRow("储层模型体", self._template_volume_combo)
        self._template_top_k_spin = QSpinBox(body)
        self._template_top_k_spin.setRange(1, 50)
        self._template_top_k_spin.setValue(8)
        template_form.addRow("返回候选数", self._template_top_k_spin)
        outer.addLayout(template_form)

        template_buttons = QHBoxLayout()
        self._draw_template_button = QPushButton("绘制模板", body)
        self._draw_template_button.clicked.connect(self._open_template_canvas)
        self._clear_template_button = QPushButton("清除模板", body)
        self._clear_template_button.clicked.connect(self._clear_template)
        template_buttons.addWidget(self._draw_template_button)
        template_buttons.addWidget(self._clear_template_button)
        outer.addLayout(template_buttons)

        self._template_status_label = QLabel("未绘制模板", body)
        self._template_status_label.setWordWrap(True)
        outer.addWidget(self._template_status_label)

        self._template_search_button = QPushButton("按模板搜索相似", body)
        self._template_search_button.clicked.connect(self._on_template_search_clicked)
        outer.addWidget(self._template_search_button)

        self._progress_bar = QProgressBar(body)
        self._progress_bar.setRange(0, 100)
        outer.addWidget(self._progress_bar)

        self._summary_label = QLabel("", body)
        self._summary_label.setWordWrap(True)
        outer.addWidget(self._summary_label)

        # Reflect the default toggle states (strict box on, auto off).
        self._on_box_strict_toggled(self._box_strict_check.isChecked())
        self._on_auto_range_toggled(self._auto_range_check.isChecked())
        self._refresh_template_volume_combo()

        self.setWidget(body)

    # ------------------------------------------------------------------ slots

    def _on_start_clicked(self) -> None:
        if not _has_remote_segment_backend(self._ai_service):
            QMessageBox.information(
                self,
                "SAM3",
                "SAM3 分割/追踪已统一到远程服务器，请配置 YJ_STUDIO_SERVER_URL。",
            )
            return
        if not self._ai_service.config.checkpoint_exists():
            QMessageBox.warning(
                self,
                "SAM3 检查点缺失",
                f"找不到 {self._ai_service.config.checkpoint_path}。\n"
                "请更新 settings.json 或把 .pt 文件放到对应位置。",
            )
            return
        self._ai_service.start()

    def _on_service_state(self, state: AIServiceState, message: str) -> None:
        self._refresh_state_label(state, message)

    def _refresh_state_label(self, state: AIServiceState, message: str) -> None:
        colour = {
            AIServiceState.IDLE: "#aaa",
            AIServiceState.LOADING: "#d59f00",
            AIServiceState.READY: "#2ca02c",
            AIServiceState.BUSY: "#1f77b4",
            AIServiceState.ERROR: "#d62728",
        }.get(state, "#aaa")
        self._status_label.setText(f"<span style='color:{colour};'>● {ai_state_label(state.value)}</span> {message}")
        self._start_button.setEnabled(state in {AIServiceState.IDLE, AIServiceState.ERROR})
        self._stop_button.setEnabled(state in {AIServiceState.READY, AIServiceState.ERROR})
        self._run_button.setEnabled(
            state == AIServiceState.READY and _has_remote_segment_backend(self._ai_service)
        )
        self._track_button.setEnabled(
            state == AIServiceState.READY and _has_remote_track_backend(self._ai_service)
        )
        self._update_template_search_enabled()

    def _on_box_prompt(
        self, axis: str, slice_index: int, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        current_axis = str(self._axis_combo.currentData() or "")
        if axis != current_axis or slice_index != self._slice_spin.value():
            # Sync the dock to wherever the user actually clicked.
            index = self._axis_combo.findData(axis)
            if index >= 0:
                self._axis_combo.setCurrentIndex(index)
            self._slice_spin.setValue(int(slice_index))
        self._boxes.append((x_min, y_min, x_max, y_max))
        self._prompts_list.addItem(
            QListWidgetItem(
                f"框：[{x_min:.0f}, {y_min:.0f}] → [{x_max:.0f}, {y_max:.0f}]"
            )
        )

    def _on_point_prompt(self, axis: str, slice_index: int, x: float, y: float) -> None:
        current_axis = str(self._axis_combo.currentData() or "")
        if axis != current_axis or slice_index != self._slice_spin.value():
            index = self._axis_combo.findData(axis)
            if index >= 0:
                self._axis_combo.setCurrentIndex(index)
            self._slice_spin.setValue(int(slice_index))
        self._points.append((x, y))
        self._prompts_list.addItem(QListWidgetItem(f"点：({x:.0f}, {y:.0f})"))

    def _clear_prompts(self) -> None:
        self._boxes.clear()
        self._points.clear()
        self._prompts_list.clear()

    def _clear_all_masks(self) -> None:
        """Remove every SAM3 mask layer from the scene and reset the prompts.

        Targets the AI-sourced ``MaskLayer`` visualisations only, so manually
        painted masks survive. Routed through the undo stack so the wipe is one
        reversible step. Tracking results live in the server-side target store
        and are managed from the target dock, not here.
        """
        mask_ids = [
            layer.id
            for layer in self._layer_store.iter_by_type(MaskLayer)
            if str((getattr(layer, "provenance", {}) or {}).get("source", "")).startswith("ai")
        ]
        if mask_ids:
            if self._undo_stack is not None:
                self._undo_stack.beginMacro("清空 SAM3 Mask 显示")
                try:
                    for layer_id in mask_ids:
                        self._undo_stack.push(RemoveLayerCommand(self._layer_store, layer_id))
                finally:
                    self._undo_stack.endMacro()
            else:
                for layer_id in mask_ids:
                    self._layer_store.remove(layer_id)
        self._clear_prompts()
        self._progress_bar.setValue(0)
        self._summary_label.setText(f"已清空 {len(mask_ids)} 个 Mask 显示。")

    def _on_box_strict_toggled(self, checked: bool) -> None:
        # "保留前 K" is only meaningful when not restricting to the framed target.
        self._top_k_spin.setEnabled(not checked)

    def _on_auto_range_toggled(self, checked: bool) -> None:
        # Manual front/back counts are ignored while the server auto-detects the
        # valid extent, so grey them out to make the active mode obvious.
        self._track_back_spin.setEnabled(not checked)
        self._track_fwd_spin.setEnabled(not checked)

    def _activate_box_tool(self) -> None:
        try:
            self._tool_manager.activate("ai_box_prompt")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "AI 工具", str(exc))

    def _activate_point_tool(self) -> None:
        try:
            self._tool_manager.activate("ai_point_prompt")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "AI 工具", str(exc))

    def _open_template_canvas(self) -> None:
        # Drawing happens on an independent blank whiteboard, never on the
        # section — the template expresses a shape, not a region of the image.
        from yj_studio.ui.dialogs.template_canvas_dialog import TemplateCanvasDialog

        self._refresh_template_volume_combo()
        dialog = TemplateCanvasDialog(self)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        points = dialog.template_points()
        if len(points) < 3:
            self._clear_template()
            return
        self._template = [[float(x), float(y)] for x, y in points]
        self._template_status_label.setText(
            f"已保存模板（{len(self._template)} 点），可点击“按模板搜索相似”。"
        )
        self._update_template_search_enabled()

    def _clear_template(self) -> None:
        self._template = None
        self._template_status_label.setText("未绘制模板")
        self._update_template_search_enabled()

    def _reservoir_volume_layers(self) -> list[VolumeLayer]:
        """Loaded volume layers that are reservoir models (i.e. not seismic).

        Template search runs on reservoir model slices, never on the seismic
        cube, so the seismic volume is excluded from the search-target list.
        """
        return [
            layer
            for layer in self._layer_store.iter_by_type(VolumeLayer)
            if layer.shape is not None and str(layer.volume_id) != "seismic"
        ]

    def _refresh_template_volume_combo(self) -> None:
        previous = str(self._template_volume_combo.currentData() or "")
        self._template_volume_combo.blockSignals(True)
        self._template_volume_combo.clear()
        for layer in self._reservoir_volume_layers():
            self._template_volume_combo.addItem(layer.name or layer.volume_id, layer.volume_id)
        if previous:
            idx = self._template_volume_combo.findData(previous)
            if idx >= 0:
                self._template_volume_combo.setCurrentIndex(idx)
        self._template_volume_combo.blockSignals(False)
        self._update_template_search_enabled()

    def _selected_template_volume(self) -> VolumeLayer | None:
        volume_id = str(self._template_volume_combo.currentData() or "")
        if not volume_id:
            return None
        for layer in self._reservoir_volume_layers():
            if str(layer.volume_id) == volume_id:
                return layer
        return None

    def _update_template_search_enabled(self) -> None:
        ready = (
            self._ai_service.state == AIServiceState.READY
            and _has_remote_template_backend(self._ai_service)
        )
        has_template = bool(self._template) and len(self._template or []) >= 3
        has_volume = self._template_volume_combo.count() > 0
        self._template_search_button.setEnabled(ready and has_template and has_volume)

    def _on_template_search_clicked(self) -> None:
        self._refresh_template_volume_combo()
        volume_layer = self._selected_template_volume()
        if volume_layer is None:
            QMessageBox.information(
                self, "形态模板搜索", "请先加载并选择一个储层模型体（地震体不参与形态模板识别）。"
            )
            return
        if not _has_remote_template_backend(self._ai_service):
            QMessageBox.information(self, "形态模板搜索", "形态模板搜索需要远程 SAM3 后端。")
            return
        if not self._template or len(self._template) < 3:
            QMessageBox.information(self, "形态模板搜索", "请先用“绘制模板”在白板上画出一个形状。")
            return
        axis = str(self._axis_combo.currentData() or self._axis_combo.currentText())
        index = int(self._slice_spin.value())
        self._set_slice_spin_limit(volume_layer, axis)
        index = min(index, self._slice_spin.maximum())
        params = {
            "axis": axis,
            "slice_index": index,
            "template": list(self._template),
            "confidence_threshold": float(self._confidence_spin.value()),
            "keep_top_k": int(self._template_top_k_spin.value()),
            "target_type": self._target_type_combo.currentText().strip() or "unknown",
            "name_prefix": "模板",
        }
        self._summary_label.setText("")
        self._progress_bar.setValue(0)
        self._template_search_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        task = RemoteSAM3TemplateMatchTask(
            self._ai_service, params, {"volume": volume_layer}, parent=self
        )
        self._current_task = task
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_finished)
        task.errored.connect(self._on_errored)
        task.cancelled.connect(self._on_cancelled)
        task.start()

    def _sync_from_volume(self) -> None:
        current_axis = str(self._axis_combo.currentData() or self._axis_combo.currentText())
        layer = _preferred_visible_volume_layer(self._layer_store)
        if layer is None:
            return
        self._set_slice_spin_limit(layer, current_axis)
        indices = layer.slice_indices or {}
        if current_axis in indices:
            self._slice_spin.setValue(int(indices[current_axis]))
        self._refresh_template_volume_combo()

    # ------------------------------------------------------------------ run

    def _on_run_clicked(self) -> None:
        volume_layer = self._active_volume_layer()
        if volume_layer is None:
            QMessageBox.information(
                self, "SAM3", "请先选择一个可见体数据，并确认剖面索引在该体范围内。"
            )
            return
        if not _has_remote_segment_backend(self._ai_service):
            QMessageBox.information(self, "SAM3", "SAM3 分割需要远程 SAM3 后端。")
            return
        params = {
            "axis": str(self._axis_combo.currentData() or self._axis_combo.currentText()),
            "slice_index": int(self._slice_spin.value()),
            "text_prompt": self._text_edit.toPlainText().strip(),
            "boxes": list(self._boxes),
            "points": list(self._points),
            "confidence_threshold": float(self._confidence_spin.value()),
            "keep_top_k": int(self._top_k_spin.value()),
            "box_strict": bool(self._box_strict_check.isChecked()),
            "target_type": self._target_type_combo.currentText().strip() or "unknown",
            "name_prefix": "SAM3",
        }
        if not params["text_prompt"] and not params["boxes"] and not params["points"]:
            QMessageBox.information(
                self,
                "SAM3",
                "请至少添加一个提示（文本、框或点）后再运行。",
            )
            return

        from yj_studio.algorithms.remote_sam3 import RemoteSAM3SegmentAlgorithm

        self._summary_label.setText("")
        self._progress_bar.setValue(0)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        task = self._runner.submit(
            RemoteSAM3SegmentAlgorithm,
            params,
            {"volume": volume_layer},
        )
        self._current_task = task
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_finished)
        task.errored.connect(self._on_errored)
        task.cancelled.connect(self._on_cancelled)

    def _on_track_clicked(self) -> None:
        volume_layer = self._active_volume_layer()
        if volume_layer is None:
            QMessageBox.information(self, "SAM3 追踪", "请先选择一个可见体数据，并确认剖面索引在该体范围内。")
            return
        if not _has_remote_track_backend(self._ai_service):
            QMessageBox.information(self, "SAM3 追踪", "追踪需要远程 SAM3 后端。")
            return
        if not self._boxes:
            QMessageBox.information(self, "SAM3 追踪", "请先在剖面上框选至少一个目标。")
            return
        params = {
            "volume_id": volume_layer.volume_id,
            "axis": str(self._axis_combo.currentData() or self._axis_combo.currentText()),
            "seed": int(self._slice_spin.value()),
            "back": int(self._track_back_spin.value()),
            "fwd": int(self._track_fwd_spin.value()),
            "boxes": list(self._boxes),
            "text": self._text_edit.toPlainText().strip(),
            "confidence": float(self._confidence_spin.value()),
            "keep_top_k": int(self._top_k_spin.value()),
            "box_strict": bool(self._box_strict_check.isChecked()),
            "auto": bool(self._auto_range_check.isChecked()),
            "target_type": self._target_type_combo.currentText().strip() or "unknown",
        }
        self._summary_label.setText("")
        self._progress_bar.setValue(0)
        self._run_button.setEnabled(False)
        self._track_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        task = RemoteSAM3TrackTask(self._ai_service, params, parent=self)
        self._current_task = task
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_track_finished)
        task.errored.connect(self._on_errored)
        task.cancelled.connect(self._on_cancelled)
        task.start()

    def _on_cancel_clicked(self) -> None:
        if self._current_task is not None:
            self._current_task.cancel()

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        if message:
            self._summary_label.setText(message)

    def _on_finished(self, output_layers: list, summary: str) -> None:
        self._reset_run_buttons()
        self._progress_bar.setValue(100)
        if output_layers:
            if self._undo_stack is not None:
                self._undo_stack.beginMacro("运行 SAM3 分割")
                try:
                    for layer in output_layers:
                        self._undo_stack.push(AddLayerCommand(self._layer_store, layer))
                finally:
                    self._undo_stack.endMacro()
            else:
                for layer in output_layers:
                    self._layer_store.add(layer)
        target_ids = _target_ids_from_layers(output_layers)
        self.job_finished.emit({"kind": "segment", "target_ids": target_ids})
        self._summary_label.setText(summary or f"已生成 {len(output_layers)} 个掩膜。")

    def _on_track_finished(self, result: dict, summary: str) -> None:
        self._reset_run_buttons()
        self._progress_bar.setValue(100)
        self._summary_label.setText(summary or "追踪完成。")
        payload = dict(result)
        payload.setdefault("kind", "track")
        self.job_finished.emit(payload)
        self.track_finished.emit(payload)

    def _on_errored(self, message: str, traceback_text: str) -> None:
        self._reset_run_buttons()
        self._summary_label.setText(f"错误：{message}")
        logger.error("SAM3 error: %s\n%s", message, traceback_text)

    def _on_cancelled(self) -> None:
        self._reset_run_buttons()
        self._summary_label.setText("已取消")

    def _reset_run_buttons(self) -> None:
        self._run_button.setEnabled(
            self._ai_service.state == AIServiceState.READY
            and _has_remote_segment_backend(self._ai_service)
        )
        self._track_button.setEnabled(
            self._ai_service.state == AIServiceState.READY
            and _has_remote_track_backend(self._ai_service)
        )
        self._update_template_search_enabled()
        self._cancel_button.setEnabled(False)
        self._current_task = None

    def _active_volume_layer(self) -> VolumeLayer | None:
        axis = str(self._axis_combo.currentData() or self._axis_combo.currentText())
        index = int(self._slice_spin.value())
        layer = _select_volume_layer_for_ai(
            self._layer_store,
            axis,
            index,
        )
        if layer is not None:
            self._set_slice_spin_limit(layer, axis)
        return layer

    def _set_slice_spin_limit(self, layer: VolumeLayer, axis: str) -> None:
        limit = _axis_limit(layer, axis)
        if limit is None:
            return
        self._slice_spin.setMaximum(max(0, limit - 1))


def _select_volume_layer_for_ai(
    layer_store: LayerStore,
    axis: str,
    index: int,
) -> VolumeLayer | None:
    """Pick the volume that the AI dock should submit to SAM3.

    Multiple volume layers can exist at once (seismic + 3x reservoir models).
    The old behaviour picked the first layer in insertion order, which is
    usually seismic; that let a reservoir index such as 2226 be submitted as
    ``seismic/inline/2226`` and caused server-side 416 errors.
    """

    layers = [layer for layer in layer_store.iter_by_type(VolumeLayer) if layer.shape is not None]
    if not layers:
        return None
    selected = set(layer_store.selection)

    passes = (
        lambda layer: layer.id in selected and layer.visible and _index_in_bounds(layer, axis, index),
        lambda layer: layer.visible and _layer_slice_matches(layer, axis, index),
        lambda layer: layer.visible and _index_in_bounds(layer, axis, index),
        lambda layer: layer.id in selected and _index_in_bounds(layer, axis, index),
        lambda layer: _layer_slice_matches(layer, axis, index),
        lambda layer: _index_in_bounds(layer, axis, index),
    )
    for predicate in passes:
        for layer in layers:
            if predicate(layer):
                return layer
    return None


def _preferred_visible_volume_layer(layer_store: LayerStore) -> VolumeLayer | None:
    layers = [layer for layer in layer_store.iter_by_type(VolumeLayer) if layer.shape is not None]
    selected = set(layer_store.selection)
    for layer in layers:
        if layer.id in selected and layer.visible:
            return layer
    for layer in layers:
        if layer.visible:
            return layer
    for layer in layers:
        if layer.id in selected:
            return layer
    return layers[0] if layers else None


def _axis_limit(layer: VolumeLayer, axis: str) -> int | None:
    if layer.shape is None:
        return None
    axis_key = str(axis)
    if axis_key == "inline":
        return int(layer.shape[0])
    if axis_key == "xline":
        return int(layer.shape[1])
    if axis_key == "z":
        return int(layer.shape[2])
    return None


def _index_in_bounds(layer: VolumeLayer, axis: str, index: int) -> bool:
    limit = _axis_limit(layer, axis)
    return limit is not None and 0 <= int(index) < limit


def _layer_slice_matches(layer: VolumeLayer, axis: str, index: int) -> bool:
    if not _index_in_bounds(layer, axis, index):
        return False
    try:
        return int(layer.slice_indices.get(axis, -1)) == int(index)
    except (TypeError, ValueError):
        return False


def _has_remote_segment_backend(service: Any) -> bool:
    return (
        callable(getattr(service, "submit_segment", None))
        and callable(getattr(service, "fetch_mask", None))
    )


def _has_remote_track_backend(service: Any) -> bool:
    return _has_remote_segment_backend(service) and callable(getattr(service, "submit_track", None))


def _has_remote_template_backend(service: Any) -> bool:
    return _has_remote_segment_backend(service) and callable(
        getattr(service, "submit_template_match", None)
    )


def _target_ids_from_layers(layers: list[Any]) -> list[str]:
    ids: list[str] = []
    for layer in layers:
        metadata = getattr(layer, "metadata", {}) or {}
        target_id = str(metadata.get("target_id", "") or "")
        if target_id:
            ids.append(target_id)
    return ids
