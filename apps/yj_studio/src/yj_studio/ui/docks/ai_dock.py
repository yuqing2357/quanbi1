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

The dock owns no model state — it forwards Run clicks to
``AlgorithmRunner.submit(SAM3SegmentAlgorithm, ...)``. Outputs come back as
``MaskLayer`` instances and are added through the undo stack so a single
Ctrl+Z removes the whole run.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QUndoStack
from PyQt6.QtWidgets import (
    QComboBox,
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

from yj_studio.ai.service import AIService, AIServiceState
from yj_studio.algorithms.runner import AlgorithmRunner
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import VolumeLayer
from yj_studio.scene.undo_commands import AddLayerCommand
from yj_studio.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

_AXES = ("inline", "xline", "z")


class AIDock(QDockWidget):
    def __init__(
        self,
        layer_store: LayerStore,
        ai_service: AIService,
        runner: AlgorithmRunner,
        tool_manager: ToolManager,
        undo_stack: QUndoStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("AI Assist", parent)
        self._layer_store = layer_store
        self._ai_service = ai_service
        self._runner = runner
        self._tool_manager = tool_manager
        self._undo_stack = undo_stack
        self._boxes: list[tuple[float, float, float, float]] = []
        self._points: list[tuple[float, float]] = []
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
        self._status_label = QLabel("AI: idle", body)
        self._start_button = QPushButton("Start AI", body)
        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button = QPushButton("Unload", body)
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
            self._axis_combo.addItem(axis)
        form.addRow("Axis", self._axis_combo)
        self._slice_spin = QSpinBox(body)
        self._slice_spin.setRange(0, 9999)
        form.addRow("Slice index", self._slice_spin)

        sync_button = QPushButton("← Use active 3D slice", body)
        sync_button.clicked.connect(self._sync_from_volume)
        form.addRow("", sync_button)
        outer.addLayout(form)

        # Text prompt
        outer.addWidget(QLabel("Text prompt:", body))
        self._text_edit = QTextEdit(body)
        self._text_edit.setPlaceholderText(
            "e.g. salt body, channel sand, fault zone"
        )
        self._text_edit.setFixedHeight(60)
        outer.addWidget(self._text_edit)

        # Geometric prompts
        outer.addWidget(QLabel("Geometric prompts:", body))
        self._prompts_list = QListWidget(body)
        self._prompts_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        outer.addWidget(self._prompts_list, 1)

        prompt_buttons = QHBoxLayout()
        self._pick_box_button = QPushButton("Pick Box", body)
        self._pick_box_button.clicked.connect(self._activate_box_tool)
        self._pick_point_button = QPushButton("Pick Point", body)
        self._pick_point_button.clicked.connect(self._activate_point_tool)
        self._clear_button = QPushButton("Clear", body)
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
        run_form.addRow("Confidence", self._confidence_spin)
        self._top_k_spin = QSpinBox(body)
        self._top_k_spin.setRange(1, 50)
        self._top_k_spin.setValue(3)
        run_form.addRow("Top K", self._top_k_spin)
        outer.addLayout(run_form)

        # Run row
        run_row = QHBoxLayout()
        self._run_button = QPushButton("Run SAM3 Segment", body)
        self._run_button.clicked.connect(self._on_run_clicked)
        self._cancel_button = QPushButton("Cancel", body)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        run_row.addWidget(self._run_button)
        run_row.addWidget(self._cancel_button)
        outer.addLayout(run_row)

        self._progress_bar = QProgressBar(body)
        self._progress_bar.setRange(0, 100)
        outer.addWidget(self._progress_bar)

        self._summary_label = QLabel("", body)
        self._summary_label.setWordWrap(True)
        outer.addWidget(self._summary_label)

        self.setWidget(body)

    # ------------------------------------------------------------------ slots

    def _on_start_clicked(self) -> None:
        if not self._ai_service.config.checkpoint_exists():
            QMessageBox.warning(
                self,
                "SAM3 checkpoint missing",
                f"Cannot find {self._ai_service.config.checkpoint_path}.\n"
                "Update settings.json or move the .pt file into place.",
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
        self._status_label.setText(f"<span style='color:{colour};'>● {state.value}</span> {message}")
        self._start_button.setEnabled(state in {AIServiceState.IDLE, AIServiceState.ERROR})
        self._stop_button.setEnabled(state in {AIServiceState.READY, AIServiceState.ERROR})
        self._run_button.setEnabled(state == AIServiceState.READY)

    def _on_box_prompt(
        self, axis: str, slice_index: int, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        if axis != self._axis_combo.currentText() or slice_index != self._slice_spin.value():
            # Sync the dock to wherever the user actually clicked.
            self._axis_combo.setCurrentText(axis)
            self._slice_spin.setValue(int(slice_index))
        self._boxes.append((x_min, y_min, x_max, y_max))
        self._prompts_list.addItem(
            QListWidgetItem(
                f"Box: [{x_min:.0f}, {y_min:.0f}] → [{x_max:.0f}, {y_max:.0f}]"
            )
        )

    def _on_point_prompt(self, axis: str, slice_index: int, x: float, y: float) -> None:
        if axis != self._axis_combo.currentText() or slice_index != self._slice_spin.value():
            self._axis_combo.setCurrentText(axis)
            self._slice_spin.setValue(int(slice_index))
        self._points.append((x, y))
        self._prompts_list.addItem(QListWidgetItem(f"Point: ({x:.0f}, {y:.0f})"))

    def _clear_prompts(self) -> None:
        self._boxes.clear()
        self._points.clear()
        self._prompts_list.clear()

    def _activate_box_tool(self) -> None:
        try:
            self._tool_manager.activate("ai_box_prompt")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "AI tool", str(exc))

    def _activate_point_tool(self) -> None:
        try:
            self._tool_manager.activate("ai_point_prompt")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "AI tool", str(exc))

    def _sync_from_volume(self) -> None:
        for layer in self._layer_store.iter_by_type(VolumeLayer):
            indices = layer.slice_indices or {}
            current_axis = self._axis_combo.currentText()
            if current_axis in indices:
                self._slice_spin.setValue(int(indices[current_axis]))
                return

    # ------------------------------------------------------------------ run

    def _on_run_clicked(self) -> None:
        volume_layer = self._active_volume_layer()
        if volume_layer is None:
            QMessageBox.information(
                self, "SAM3", "Load a volume before running SAM3."
            )
            return
        params = {
            "axis": self._axis_combo.currentText(),
            "slice_index": int(self._slice_spin.value()),
            "text_prompt": self._text_edit.toPlainText().strip(),
            "boxes": list(self._boxes),
            "points": list(self._points),
            "confidence_threshold": float(self._confidence_spin.value()),
            "keep_top_k": int(self._top_k_spin.value()),
            "name_prefix": "SAM3",
        }
        if not params["text_prompt"] and not params["boxes"] and not params["points"]:
            QMessageBox.information(
                self,
                "SAM3",
                "Add at least one prompt (text, box, or point) before running.",
            )
            return

        from yj_studio.algorithms.builtin.ai.sam3_segment import SAM3SegmentAlgorithm

        self._summary_label.setText("")
        self._progress_bar.setValue(0)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        task = self._runner.submit(
            SAM3SegmentAlgorithm,
            params,
            {"volume": volume_layer},
        )
        self._current_task = task
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_finished)
        task.errored.connect(self._on_errored)
        task.cancelled.connect(self._on_cancelled)

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
                self._undo_stack.beginMacro("Run SAM3 Segment")
                try:
                    for layer in output_layers:
                        self._undo_stack.push(AddLayerCommand(self._layer_store, layer))
                finally:
                    self._undo_stack.endMacro()
            else:
                for layer in output_layers:
                    self._layer_store.add(layer)
        self._summary_label.setText(summary or f"Produced {len(output_layers)} mask(s)")

    def _on_errored(self, message: str, traceback_text: str) -> None:
        self._reset_run_buttons()
        self._summary_label.setText(f"Error: {message}")
        logger.error("SAM3 error: %s\n%s", message, traceback_text)

    def _on_cancelled(self) -> None:
        self._reset_run_buttons()
        self._summary_label.setText("Cancelled")

    def _reset_run_buttons(self) -> None:
        self._run_button.setEnabled(self._ai_service.state == AIServiceState.READY)
        self._cancel_button.setEnabled(False)
        self._current_task = None

    def _active_volume_layer(self) -> VolumeLayer | None:
        for layer in self._layer_store.iter_by_type(VolumeLayer):
            if layer.shape is not None:
                return layer
        return None
