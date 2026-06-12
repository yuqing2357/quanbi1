from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QUndoStack
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from yj_studio.ai.adapters import build_mask_layer, sam3_mask_to_layer
from yj_studio.data import RemoteTargetStore
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import MaskLayer, ReservoirGridLayer, ReservoirPropertyLayer, ReservoirSelectionLayer
from yj_studio.scene.undo_commands import AddLayerCommand
from yj_studio.targets import BUILTIN_TARGET_TYPES, GeoTarget, TargetFrame, TargetSet, TargetStatus, review_queue


class TargetDock(QDockWidget):
    """Remote target manager for SAM3 geological objects."""

    def __init__(
        self,
        layer_store: LayerStore,
        target_store: RemoteTargetStore | None,
        *,
        undo_stack: QUndoStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("目标", parent)
        self._layer_store = layer_store
        self._target_store = target_store
        self._undo_stack = undo_stack
        self._targets: dict[str, GeoTarget] = {}
        self._pending_suggestion: dict[str, object] | None = None

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        self._status_label = QLabel("远程模式" if target_store is not None else "本地模式", root)
        self._refresh_button = QPushButton("刷新", root)
        self._review_button = QPushButton("审校", root)
        self._refresh_button.clicked.connect(self.refresh)
        self._review_button.clicked.connect(self._show_review_queue)
        top.addWidget(self._status_label)
        top.addStretch(1)
        top.addWidget(self._review_button)
        top.addWidget(self._refresh_button)
        layout.addLayout(top)

        suggestion_row = QHBoxLayout()
        self._suggestion_label = QLabel("", root)
        self._suggestion_label.setWordWrap(True)
        self._suggestion_confirm_button = QPushButton("确认", root)
        self._suggestion_confirm_button.clicked.connect(self._confirm_suggestion)
        self._suggestion_ignore_button = QPushButton("忽略", root)
        self._suggestion_ignore_button.clicked.connect(self._clear_suggestion)
        suggestion_row.addWidget(self._suggestion_label, 1)
        suggestion_row.addWidget(self._suggestion_confirm_button)
        suggestion_row.addWidget(self._suggestion_ignore_button)
        layout.addLayout(suggestion_row)
        self._clear_suggestion()

        self._table = QTableWidget(0, 6, root)
        self._table.setHorizontalHeaderLabels(["ID", "类型", "状态", "帧", "面积", "Score"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._sync_selection_controls)
        layout.addWidget(self._table, 1)

        form = QFormLayout()
        self._name_edit = QLineEdit(root)
        self._type_combo = QComboBox(root)
        self._type_combo.setEditable(True)
        self._type_combo.addItems(list(BUILTIN_TARGET_TYPES))
        form.addRow("名称", self._name_edit)
        form.addRow("类型", self._type_combo)
        layout.addLayout(form)

        action_row = QHBoxLayout()
        self._apply_button = QPushButton("应用", root)
        self._confirm_button = QPushButton("确认", root)
        self._delete_button = QPushButton("删除", root)
        self._merge_button = QPushButton("合并", root)
        self._split_button = QPushButton("拆分", root)
        for button in (
            self._apply_button,
            self._confirm_button,
            self._delete_button,
            self._merge_button,
            self._split_button,
        ):
            action_row.addWidget(button)
        layout.addLayout(action_row)

        load_row = QHBoxLayout()
        self._load_mask_button = QPushButton("2D", root)
        self._load_cells_button = QPushButton("3D", root)
        self._train_button = QPushButton("训练集", root)
        self._models_button = QPushButton("模型", root)
        for button in (
            self._load_mask_button,
            self._load_cells_button,
            self._train_button,
            self._models_button,
        ):
            load_row.addWidget(button)
        layout.addLayout(load_row)

        extract_row = QHBoxLayout()
        self._extract_type = QComboBox(root)
        self._extract_type.setEditable(True)
        self._extract_type.addItems(list(BUILTIN_TARGET_TYPES))
        self._extract_scope = QComboBox(root)
        self._extract_scope.addItems(["page", "volume"])
        self._extract_mode = QComboBox(root)
        self._extract_mode.addItems(["prompt", "track", "infer_volume"])
        self._extract_button = QPushButton("提取", root)
        extract_row.addWidget(self._extract_type)
        extract_row.addWidget(self._extract_scope)
        extract_row.addWidget(self._extract_mode)
        extract_row.addWidget(self._extract_button)
        layout.addLayout(extract_row)

        self._apply_button.clicked.connect(self._apply_selected)
        self._confirm_button.clicked.connect(self._confirm_selected)
        self._delete_button.clicked.connect(self._delete_selected)
        self._merge_button.clicked.connect(self._merge_selected)
        self._split_button.clicked.connect(self._split_selected)
        self._load_mask_button.clicked.connect(self._load_selected_mask)
        self._load_cells_button.clicked.connect(self._load_selected_cells)
        self._train_button.clicked.connect(self._submit_train_export)
        self._models_button.clicked.connect(self._show_models)
        self._extract_button.clicked.connect(self._extract_from_selected)

        self.setWidget(root)
        self._set_remote_enabled(target_store is not None)

    def refresh(self) -> None:
        if self._target_store is None:
            return
        try:
            target_set = self._target_store.load_targets(include_deleted=False)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            QMessageBox.warning(self, "目标", str(exc))
            return
        self._targets = dict(target_set.targets)
        rows = target_set.summaries(include_deleted=False)
        self._table.setRowCount(len(rows))
        for row, summary in enumerate(rows):
            values = [
                summary.get("id", ""),
                summary.get("type", ""),
                summary.get("status", ""),
                summary.get("frame_range", ""),
                summary.get("area_px", 0),
                "" if summary.get("score") is None else f"{float(summary['score']):.3f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, summary.get("id", ""))
                self._table.setItem(row, col, item)
        self._sync_selection_controls()

    def _set_remote_enabled(self, enabled: bool) -> None:
        for widget in (
            self._refresh_button,
            self._review_button,
            self._table,
            self._name_edit,
            self._type_combo,
            self._apply_button,
            self._confirm_button,
            self._delete_button,
            self._merge_button,
            self._split_button,
            self._load_mask_button,
            self._load_cells_button,
            self._train_button,
            self._models_button,
            self._extract_button,
            self._suggestion_confirm_button,
            self._suggestion_ignore_button,
        ):
            widget.setEnabled(enabled)

    def show_track_result(self, result: dict[str, object]) -> None:
        suggestions = result.get("suggestions", [])
        if not isinstance(suggestions, list):
            self._clear_suggestion()
            return
        useful = next((_normalise_suggestion(item) for item in suggestions if _normalise_suggestion(item)), None)
        if useful is None:
            self._clear_suggestion()
            return
        self._pending_suggestion = useful
        kind = str(useful.get("type", ""))
        if kind == "merge":
            ids = ", ".join(str(item) for item in useful.get("target_ids", []))
            self._suggestion_label.setText(f"检测到 {ids} 可能应合并。")
        elif kind == "split":
            self._suggestion_label.setText(f"检测到 {useful.get('target_id')} 可能应拆分。")
        self._suggestion_label.setVisible(True)
        self._suggestion_confirm_button.setVisible(True)
        self._suggestion_ignore_button.setVisible(True)

    def _selected_ids(self) -> list[str]:
        selection = self._table.selectionModel()
        if selection is None:
            return []
        ids: list[str] = []
        for index in selection.selectedRows():
            item = self._table.item(index.row(), 0)
            target_id = str(item.data(Qt.ItemDataRole.UserRole) if item is not None else "")
            if target_id and target_id not in ids:
                ids.append(target_id)
        return ids

    def _selected_target(self) -> GeoTarget | None:
        ids = self._selected_ids()
        if not ids:
            return None
        target = self._targets.get(ids[0])
        if target is not None:
            return target
        if self._target_store is None:
            return None
        try:
            target = self._target_store.fetch_target(ids[0])
        except Exception:  # noqa: BLE001
            return None
        self._targets[target.id] = target
        return target

    def _sync_selection_controls(self) -> None:
        target = self._selected_target()
        if target is None:
            self._name_edit.clear()
            self._type_combo.setCurrentText("unknown")
            return
        self._name_edit.setText(target.name or "")
        self._type_combo.setCurrentText(target.type or "unknown")

    def _apply_selected(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        try:
            self._target_store.patch_target(
                target.id,
                {
                    "name": self._name_edit.text().strip(),
                    "type": self._type_combo.currentText().strip() or "unknown",
                },
            )
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _confirm_selected(self) -> None:
        self._patch_status(TargetStatus.CONFIRMED.value)

    def _delete_selected(self) -> None:
        if self._target_store is None:
            return
        try:
            for target_id in self._selected_ids():
                self._target_store.delete_target(target_id)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _patch_status(self, status: str) -> None:
        if self._target_store is None:
            return
        try:
            for target_id in self._selected_ids():
                self._target_store.patch_target(target_id, {"status": status})
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _merge_selected(self) -> None:
        ids = self._selected_ids()
        if len(ids) < 2 or self._target_store is None:
            return
        try:
            self._target_store.merge_targets(ids)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _confirm_suggestion(self) -> None:
        if self._target_store is None or self._pending_suggestion is None:
            return
        suggestion = dict(self._pending_suggestion)
        try:
            if suggestion.get("type") == "merge":
                target_ids = [str(item) for item in suggestion.get("target_ids", [])]
                if len(target_ids) >= 2:
                    self._target_store.merge_targets(target_ids)
            elif suggestion.get("type") == "split":
                target_id = str(suggestion.get("target_id", ""))
                if target_id:
                    self._target_store.split_target(target_id)
            self._clear_suggestion()
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标建议", str(exc))

    def _clear_suggestion(self) -> None:
        self._pending_suggestion = None
        self._suggestion_label.clear()
        self._suggestion_label.setVisible(False)
        self._suggestion_confirm_button.setVisible(False)
        self._suggestion_ignore_button.setVisible(False)

    def _split_selected(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        groups = [[key] for key in (target.trajectory or sorted(target.frames))]
        if len(groups) < 2:
            return
        try:
            self._target_store.split_target(target.id, groups)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _load_selected_mask(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        frame = _first_frame(target)
        if frame is None:
            return
        try:
            mask = self._target_store.fetch_mask(target.id, frame.axis, frame.index, volume_id=target.volume_id)
            layer = build_mask_layer(
                sam3_mask_to_layer(mask),
                name=f"{target.id} {target.type}",
                axis=_desktop_axis(frame.axis),
                slice_index=frame.index,
                score=target.score,
                metadata={
                    "target_id": target.id,
                    "target_type": target.type,
                    "volume_id": target.volume_id,
                    "remote_target": True,
                },
                provenance={"source": "sam3.target"},
            )
            self._add_layer(layer)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _load_selected_cells(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        try:
            cells = np.asarray(self._target_store.fetch_cells(target.id, volume_id=target.volume_id), dtype=np.int32)
            if cells.ndim == 1 and cells.size == 0:
                cells = cells.reshape(0, 3)
            if cells.shape[0] == 0:
                self._load_selected_mask3d(target)
                return
            grid_layer_id, grid_id = self._best_grid_ref()
            frame = _first_frame(target)
            layer = ReservoirSelectionLayer(
                name=f"{target.id} 选择体",
                grid_layer_id=grid_layer_id,
                grid_id=grid_id,
                cell_ids=cells,
                source_axis=_ijk_axis(frame.axis if frame else None),
                source_index_lo=frame.index if frame else None,
                source_index_hi=frame.index if frame else None,
                score=target.score,
                color=(1.0, 0.75, 0.1, 0.72),
                opacity=0.72,
                metadata={
                    "target_id": target.id,
                    "target_type": target.type,
                    "remote_target": True,
                    "external_cells_ref": f"/sam3/targets/{target.id}/cells",
                },
                provenance={"source": "sam3.target.cells"},
            )
            self._add_layer(layer)
        except Exception as exc:  # noqa: BLE001
            try:
                self._load_selected_mask3d(target)
            except Exception as mask_exc:  # noqa: BLE001
                QMessageBox.warning(self, "目标", f"{exc}\n{mask_exc}")

    def _load_selected_mask3d(self, target: GeoTarget) -> None:
        if self._target_store is None:
            return
        mask = np.asarray(self._target_store.fetch_mask3d(target.id, volume_id=target.volume_id), dtype=np.uint8)
        if mask.ndim != 3 or not np.any(mask):
            QMessageBox.information(self, "目标", "该目标还没有可显示的 3D mask。")
            return
        frame = _first_frame(target)
        target_axis = frame.axis if frame else _dominant_axis(target)
        axis = _desktop_axis(target_axis)
        index_lo = _mask3d_index_lo(target, target_axis)
        layer = MaskLayer(
            name=f"{target.id} 目标体",
            mask=mask,
            axis=axis,
            slice_index=index_lo,
            confidence=target.score,
            color=_target_mask_color(target.type),
            opacity=0.46,
            metadata={
                "target_id": target.id,
                "target_type": target.type,
                "volume_id": target.volume_id,
                "remote_target": True,
                "mask3d": True,
                "mask3d_index_lo": index_lo,
                "source_axis": target_axis,
                "external_mask3d_ref": f"/sam3/targets/{target.id}/mask3d",
            },
            provenance={"source": "sam3.target.mask3d"},
        )
        self._add_layer(layer)

    def _submit_train_export(self) -> None:
        if self._target_store is None:
            return
        try:
            payload = self._target_store.submit_train_job({})
            QMessageBox.information(self, "训练集", f"已提交：{payload.get('job_id', '')}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "训练集", str(exc))

    def _show_models(self) -> None:
        if self._target_store is None:
            return
        try:
            payload = self._target_store.models()
            dialog = _ModelManagerDialog(self._target_store, payload, self)
            dialog.exec()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "模型", str(exc))

    def _show_review_queue(self) -> None:
        if self._target_store is None:
            return
        try:
            target_set = self._target_store.load_targets(include_deleted=False)
            self._targets = dict(target_set.targets)
            dialog = _ReviewQueueDialog(self._target_store, _review_rows(target_set), self)
            dialog.status_changed.connect(self.refresh)
            dialog.exec()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "审校", str(exc))

    def _extract_from_selected(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        frame = _first_frame(target)
        if frame is None:
            return
        try:
            payload: dict[str, object] = {
                "volume_id": target.volume_id,
                "axis": _desktop_axis(frame.axis),
                "index": frame.index,
            }
            if self._extract_scope.currentText() == "volume":
                indices = [f.index for f in target.frames.values() if f.axis == frame.axis]
                if indices:
                    payload["start_index"] = min(indices)
                    payload["end_index"] = max(indices)
            response = self._target_store.extract_all(
                target_type=self._extract_type.currentText().strip() or "unknown",
                scope=self._extract_scope.currentText(),
                mode=self._extract_mode.currentText(),
                **payload,
            )
            QMessageBox.information(self, "提取", f"已提交：{response.get('job_id', '')}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "提取", str(exc))

    def _best_grid_ref(self) -> tuple[str, str]:
        for layer in self._layer_store.iter_by_type(ReservoirGridLayer):
            return layer.id, layer.grid_id
        for layer in self._layer_store.iter_by_type(ReservoirPropertyLayer):
            return layer.grid_layer_id, layer.grid_id
        return "", ""

    def _add_layer(self, layer) -> None:
        if self._undo_stack is not None:
            command = AddLayerCommand(self._layer_store, layer)
            self._undo_stack.push(command)
            if command.layer_id is not None:
                self._layer_store.select([command.layer_id])
            return
        layer_id = self._layer_store.add(layer)
        self._layer_store.select([layer_id])


def _first_frame(target: GeoTarget) -> TargetFrame | None:
    keys = target.trajectory or sorted(target.frames)
    if not keys:
        return None
    return target.frames.get(keys[0])


def _desktop_axis(axis: str) -> str:
    return {"crossline": "xline", "timeslice": "z"}.get(axis, axis)


def _ijk_axis(axis: str | None) -> str | None:
    return {"inline": "i", "crossline": "j", "timeslice": "k"}.get(str(axis), None)


def _dominant_axis(target: GeoTarget) -> str:
    counts: dict[str, int] = {}
    for frame in target.frames.values():
        counts[frame.axis] = counts.get(frame.axis, 0) + 1
    if not counts:
        return "timeslice"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _mask3d_index_lo(target: GeoTarget, axis: str | None) -> int:
    indices = [int(frame.index) for frame in target.frames.values() if axis is None or frame.axis == axis]
    return min(indices) if indices else 0


def _target_mask_color(target_type: str) -> tuple[float, float, float, float]:
    from yj_studio.targets import target_type_color

    return target_type_color(target_type or "unknown", alpha=0.46)


def _normalise_suggestion(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    kind = str(value.get("type", ""))
    if kind == "merge":
        target_ids = [str(item) for item in value.get("target_ids", []) if str(item)]
        if len(set(target_ids)) >= 2:
            row = dict(value)
            row["target_ids"] = target_ids
            return row
    if kind == "split":
        target_id = str(value.get("target_id", ""))
        if target_id:
            row = dict(value)
            row["target_id"] = target_id
            return row
    return None


def _model_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    active_model = str(payload.get("active_model") or "")
    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list):
        return []
    rows: list[dict[str, object]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics", {})
        metrics_text = _metrics_text(metrics if isinstance(metrics, dict) else {})
        model_id = str(item.get("id", "") or "")
        if not model_id:
            continue
        rows.append(
            {
                "id": model_id,
                "active": model_id == active_model,
                "parent_model_id": str(item.get("parent_model_id") or ""),
                "dataset_version": str(item.get("dataset_version") or ""),
                "status": str(item.get("status") or ""),
                "checkpoint": str(item.get("checkpoint") or ""),
                "metrics": metrics_text,
                "created_at": str(item.get("created_at") or ""),
            }
        )
    rows.sort(key=lambda row: (not bool(row["active"]), str(row["created_at"])), reverse=False)
    return rows


def _metrics_text(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    for key in ("mask_iou", "dice", "mAP", "map", "precision", "recall"):
        if key not in metrics:
            continue
        value = metrics.get(key)
        try:
            parts.append(f"{key}={float(value):.3f}")
        except (TypeError, ValueError):
            parts.append(f"{key}={value}")
    if parts:
        return ", ".join(parts)
    if not metrics:
        return ""
    return ", ".join(f"{key}={value}" for key, value in sorted(metrics.items())[:3])


def _review_rows(target_set: TargetSet) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in review_queue(target_set):
        rows.append(
            {
                "id": str(row.get("id", "")),
                "type": str(row.get("type", "")),
                "status": str(row.get("status", "")),
                "frame_range": str(row.get("frame_range", "")),
                "area_px": int(row.get("area_px", 0) or 0),
                "score": "" if row.get("score") is None else f"{float(row['score']):.3f}",
                "uncertainty": f"{float(row.get('uncertainty', 0.0)):.3f}",
            }
        )
    return rows


class _ReviewQueueDialog(QDialog):
    status_changed = pyqtSignal()

    def __init__(
        self,
        target_store: RemoteTargetStore,
        rows: list[dict[str, object]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("审校队列")
        self.resize(720, 360)
        self._target_store = target_store
        self._rows = rows

        layout = QVBoxLayout(self)
        self._status = QLabel(f"待审目标：{len(rows)}", self)
        layout.addWidget(self._status)

        self._table = QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(["ID", "类型", "状态", "帧", "面积", "Score", "不确定度"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        row = QHBoxLayout()
        self._confirm_button = QPushButton("确认", self)
        self._reject_button = QPushButton("打回", self)
        self._close_button = QPushButton("关闭", self)
        row.addWidget(self._confirm_button)
        row.addWidget(self._reject_button)
        row.addStretch(1)
        row.addWidget(self._close_button)
        layout.addLayout(row)

        self._confirm_button.clicked.connect(lambda: self._patch_selected(TargetStatus.CONFIRMED.value))
        self._reject_button.clicked.connect(lambda: self._patch_selected(TargetStatus.REJECTED.value))
        self._close_button.clicked.connect(self.accept)
        self._populate()

    def _populate(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = [
                row.get("id", ""),
                row.get("type", ""),
                row.get("status", ""),
                row.get("frame_range", ""),
                row.get("area_px", 0),
                row.get("score", ""),
                row.get("uncertainty", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, row.get("id", ""))
                self._table.setItem(row_index, col, item)
        if self._rows:
            self._table.selectRow(0)
        self._status.setText(f"待审目标：{len(self._rows)}")

    def _selected_ids(self) -> list[str]:
        selection = self._table.selectionModel()
        if selection is None:
            return []
        ids: list[str] = []
        for index in selection.selectedRows():
            item = self._table.item(index.row(), 0)
            target_id = str(item.data(Qt.ItemDataRole.UserRole) if item is not None else "")
            if target_id and target_id not in ids:
                ids.append(target_id)
        return ids

    def _patch_selected(self, status: str) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        try:
            for target_id in ids:
                self._target_store.patch_target(target_id, {"status": status})
            selected = set(ids)
            self._rows = [row for row in self._rows if str(row.get("id", "")) not in selected]
            self._populate()
            self.status_changed.emit()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "审校", str(exc))


class _ModelManagerDialog(QDialog):
    def __init__(self, target_store: RemoteTargetStore, payload: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("模型")
        self.resize(760, 360)
        self._target_store = target_store
        self._payload = payload
        self._rows: list[dict[str, object]] = []

        layout = QVBoxLayout(self)
        self._status = QLabel("", self)
        layout.addWidget(self._status)

        self._table = QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(["Active", "ID", "Parent", "Dataset", "Status", "Metrics", "Checkpoint"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        row = QHBoxLayout()
        self._refresh_button = QPushButton("刷新", self)
        self._activate_button = QPushButton("激活", self)
        self._rollback_button = QPushButton("回滚到父模型", self)
        self._close_button = QPushButton("关闭", self)
        row.addWidget(self._refresh_button)
        row.addWidget(self._activate_button)
        row.addWidget(self._rollback_button)
        row.addStretch(1)
        row.addWidget(self._close_button)
        layout.addLayout(row)

        self._refresh_button.clicked.connect(self._refresh)
        self._activate_button.clicked.connect(self._activate_selected)
        self._rollback_button.clicked.connect(self._rollback_selected)
        self._close_button.clicked.connect(self.accept)

        self._populate()

    def _populate(self) -> None:
        active = str(self._payload.get("active_model") or "无")
        self._rows = _model_rows(self._payload)
        self._status.setText(f"active={active}    versions={len(self._rows)}")
        self._table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            values = [
                "●" if row.get("active") else "",
                row.get("id", ""),
                row.get("parent_model_id", ""),
                row.get("dataset_version", ""),
                row.get("status", ""),
                row.get("metrics", ""),
                row.get("checkpoint", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, row.get("id", ""))
                self._table.setItem(row_index, col, item)
        if self._rows:
            self._table.selectRow(0)

    def _selected_row(self) -> dict[str, object] | None:
        selection = self._table.selectionModel()
        if selection is None:
            return None
        selected = selection.selectedRows()
        if not selected:
            return None
        index = int(selected[0].row())
        if index < 0 or index >= len(self._rows):
            return None
        return self._rows[index]

    def _refresh(self) -> None:
        try:
            self._payload = self._target_store.models()
            self._populate()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "模型", str(exc))

    def _activate_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        model_id = str(row.get("id", "") or "")
        if not model_id:
            return
        try:
            self._target_store.activate_model(model_id)
            self._refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "模型", str(exc))

    def _rollback_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        parent_id = str(row.get("parent_model_id", "") or "")
        if not parent_id:
            QMessageBox.information(self, "模型", "该模型没有父模型可回滚。")
            return
        try:
            self._target_store.activate_model(parent_id)
            self._refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "模型", str(exc))
