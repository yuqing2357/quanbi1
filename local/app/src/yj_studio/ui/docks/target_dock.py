from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QUndoStack
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from yj_studio.data import RemoteTargetStore
from yj_studio.data.remote_target_store import Mask3DResult
from yj_studio.scene.layer_store import LayerStore
from yj_studio_core.targets import (
    DEFAULT_VOXEL_SPACING,
    GeoTarget,
    TargetFrame,
    TargetSet,
    TargetStage,
    TargetStatus,
    mask_volume_stats,
    review_queue,
)

# Stage -> tab title for the workspace.
_STAGE_TITLES: dict[str, str] = {
    TargetStage.TEMPORARY.value: "临时识别目标",
    TargetStage.SAVED.value: "保存识别目标",
    TargetStage.TRAINING.value: "训练目标",
}


class TargetDock(QDockWidget):
    """Tabbed workspace of the three target stages (临时/保存/训练).

    The AI assistant only generates results; tracks land in the *temporary*
    stage, the user confirms them into *saved*, then classifies them into the
    *training* dataset. Each tab is a :class:`StageTargetPanel`. The legacy
    single-panel constructor signature is preserved so existing callers keep
    working; ``refresh`` refreshes every stage.
    """

    def __init__(
        self,
        layer_store: LayerStore,
        target_store: RemoteTargetStore | None,
        *,
        volume_store=None,
        undo_stack: QUndoStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("目标", parent)
        self._target_store = target_store
        self._tabs = QTabWidget(self)
        self._panels: dict[str, StageTargetPanel] = {}
        for stage in (TargetStage.TEMPORARY, TargetStage.SAVED, TargetStage.TRAINING):
            panel = StageTargetPanel(
                layer_store,
                target_store,
                stage=stage.value,
                volume_store=volume_store,
                undo_stack=undo_stack,
                parent=self._tabs,
            )
            self._panels[stage.value] = panel
            self._tabs.addTab(panel, _STAGE_TITLES[stage.value])
        self.setWidget(self._tabs)

    def refresh(self) -> None:
        for panel in self._panels.values():
            panel.refresh()

    def panel(self, stage: str) -> "StageTargetPanel | None":
        return self._panels.get(stage)

    def show_track_result(self, result: dict[str, object]) -> None:
        # Tracks become temporary targets; surface them on that tab.
        temp = self._panels.get(TargetStage.TEMPORARY.value)
        if temp is None:
            return
        self._tabs.setCurrentWidget(temp)
        temp.refresh()
        temp.show_track_result(result)


class StageTargetPanel(QWidget):
    """Target manager scoped to a single lifecycle stage."""

    def __init__(
        self,
        layer_store: LayerStore,
        target_store: RemoteTargetStore | None,
        *,
        stage: str | None = None,
        volume_store=None,
        undo_stack: QUndoStack | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._layer_store = layer_store
        self._target_store = target_store
        self._stage = stage  # "temporary"/"saved"/"training" or None (legacy)
        self._volume_store = volume_store
        self._undo_stack = undo_stack
        self._targets: dict[str, GeoTarget] = {}
        self._volume_stats: dict[str, dict[str, object]] = {}
        self._visualization_windows: list[QDialog] = []

        is_temp = stage == TargetStage.TEMPORARY.value
        is_saved = stage == TargetStage.SAVED.value
        is_training = stage == TargetStage.TRAINING.value

        root = self
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        self._status_label = QLabel("远程模式" if target_store is not None else "本地模式", root)
        self._select_all_button = QPushButton("全选", root)
        self._refresh_button = QPushButton("刷新", root)
        self._review_button = QPushButton("审校", root)
        self._select_all_button.clicked.connect(self._select_all)
        self._refresh_button.clicked.connect(self.refresh)
        self._review_button.clicked.connect(self._show_review_queue)
        self._review_button.setVisible(is_saved)
        top.addWidget(self._status_label)
        top.addStretch(1)
        top.addWidget(self._select_all_button)
        top.addWidget(self._review_button)
        top.addWidget(self._refresh_button)
        layout.addLayout(top)

        self._table = QTableWidget(0, 9, root)
        headers = [
            "目标编号",
            "目标类型",
            "结果形式",
            "状态",
            "切片范围",
            "有效帧数",
            "累计面积(px)",
            "体积(m³)",
            "置信度",
        ]
        self._table.setHorizontalHeaderLabels(headers)
        header_tips = [
            "服务器项目内唯一编号；编号持久化保存，删除后不会复用。",
            "目标的地质类别，例如浊积体、圈闭、断层或未分类。",
            "区分单帧分割与跨切片追踪；“追踪仅1帧”表示传播没有产生有效相邻帧。",
            "目标当前生命周期状态。",
            "目标所在方向及最小—最大切片索引。",
            "当前保存了有效 mask 的切片数量。",
            "所有有效帧 mask 像素数之和，不是三维体积。",
            "根据三维 mask 与体素间距估算的物理体积。",
            "模型输出置信度；越接近 1 通常表示模型越确定。",
        ]
        for column, tip in enumerate(header_tips):
            item = self._table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tip)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        # Lifecycle action: the forward-promotion button is the heart of each
        # stage. temp -> 确认到保存; saved -> 分类并设为训练; training -> 导出.
        promote_row = QHBoxLayout()
        self._promote_button = QPushButton(root)
        if is_temp:
            self._promote_button.setText("确认 → 保存")
            self._promote_button.setToolTip("将选中的临时目标确认并移动到“保存识别目标”。")
        elif is_saved:
            self._promote_button.setText("分类 → 训练")
            self._promote_button.setToolTip("为选中目标设置类别并复制到“训练目标”数据集。")
        else:
            self._promote_button.setText("导出训练集")
            self._promote_button.setToolTip("导出训练目标为 COCO 数据集并提交训练任务。")
        self._promote_button.clicked.connect(self._on_promote)
        self._clear_button = QPushButton("清空", root)
        self._clear_button.setToolTip("清空该阶段的全部目标（不可恢复）。")
        self._clear_button.clicked.connect(self._on_clear_stage)
        self._clear_button.setVisible(is_temp)
        self._renumber_button = QPushButton("重新编号", root)
        self._renumber_button.setToolTip("将该阶段编号重排为连续序列。")
        self._renumber_button.clicked.connect(self._on_renumber)
        self._models_button = QPushButton("模型", root)
        self._models_button.clicked.connect(self._show_models)
        self._models_button.setVisible(is_training)
        promote_row.addWidget(self._promote_button)
        promote_row.addWidget(self._clear_button)
        promote_row.addWidget(self._renumber_button)
        promote_row.addWidget(self._models_button)
        layout.addLayout(promote_row)

        # Core review actions only: view/correct (2D), inspect (3D), delete.
        action_row = QHBoxLayout()
        self._load_mask_button = QPushButton("2D 查看/修正", root)
        self._load_cells_button = QPushButton("3D", root)
        self._delete_button = QPushButton("删除目标", root)
        self._load_mask_button.setToolTip("打开 2D 逐帧回放与掩膜修正（笔刷添加/擦除、删除错误帧）")
        self._load_cells_button.setToolTip("打开独立的三维目标体窗口")
        self._delete_button.setToolTip("删除选中的整个目标（含其 mask 文件）。")
        for button in (self._load_mask_button, self._load_cells_button, self._delete_button):
            action_row.addWidget(button)
        layout.addLayout(action_row)

        self._delete_button.clicked.connect(self._delete_selected)
        self._load_mask_button.clicked.connect(self._load_selected_mask)
        self._load_cells_button.clicked.connect(self._load_selected_cells)

        self._set_remote_enabled(target_store is not None)

    def refresh(self) -> None:
        if self._target_store is None:
            return
        try:
            target_set = self._target_store.load_targets(include_deleted=False, stage=self._stage)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            QMessageBox.warning(self, "目标", str(exc))
            return
        self._targets = dict(target_set.targets)
        rows = target_set.summaries(include_deleted=False)
        prefix = target_set.id_prefix or "T"
        self._status_label.setText(
            f"{_STAGE_TITLES.get(self._stage or '', '目标')} · 当前 {len(rows)} 个 · 下一个编号 {prefix}{target_set.next_seq}"
        )
        self._status_label.setToolTip(
            "目标编号由服务器持久化分配，删除/合并/拆分过的编号不会回收；可用“重新编号”整理。"
        )
        self._table.setRowCount(len(rows))
        for row, summary in enumerate(rows):
            values = [
                summary.get("id", ""),
                _target_type_text(str(summary.get("type", ""))),
                _result_kind_text(self._targets.get(str(summary.get("id", "")))),
                _target_status_text(str(summary.get("status", ""))),
                summary.get("frame_range", ""),
                summary.get("frame_count", 0),
                summary.get("area_px", 0),
                _volume_text(self._targets.get(str(summary.get("id", ""))), self._volume_stats),
                "" if summary.get("score") is None else f"{float(summary['score']):.3f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, summary.get("id", ""))
                self._table.setItem(row, col, item)

    def _set_remote_enabled(self, enabled: bool) -> None:
        for widget in (
            self._select_all_button,
            self._refresh_button,
            self._review_button,
            self._table,
            self._promote_button,
            self._clear_button,
            self._renumber_button,
            self._models_button,
            self._delete_button,
            self._load_mask_button,
            self._load_cells_button,
        ):
            widget.setEnabled(enabled)

    def _select_all(self) -> None:
        self._table.selectAll()

    def _on_promote(self) -> None:
        if self._target_store is None:
            return
        if self._stage == TargetStage.TRAINING.value:
            self._submit_train_export()
            return
        ids = self._selected_ids()
        if not ids:
            QMessageBox.information(self, "目标", "请先选择目标。")
            return
        try:
            if self._stage == TargetStage.TEMPORARY.value:
                self._target_store.promote_targets(ids, from_stage=TargetStage.TEMPORARY.value)
                QMessageBox.information(self, "保存识别目标", f"已确认 {len(ids)} 个目标到“保存识别目标”。")
            elif self._stage == TargetStage.SAVED.value:
                category, ok = QInputDialog.getText(
                    self, "分类并设为训练", "为选中目标设置训练类别（如 浊积体/圈闭/断层/砂体）："
                )
                if not ok or not category.strip():
                    return
                self._target_store.promote_targets(
                    ids, from_stage=TargetStage.SAVED.value, category=category.strip()
                )
                QMessageBox.information(self, "训练目标", f"已将 {len(ids)} 个目标加入“训练目标”。")
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _on_clear_stage(self) -> None:
        if self._target_store is None or not self._stage:
            return
        if QMessageBox.question(
            self, "清空", f"确定清空“{_STAGE_TITLES.get(self._stage, '')}”的全部目标吗？此操作不可恢复。"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._target_store.clear_stage(self._stage)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "清空", str(exc))

    def _on_renumber(self) -> None:
        if self._target_store is None or not self._stage:
            return
        try:
            self._target_store.renumber_stage(self._stage)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "重新编号", str(exc))

    def show_track_result(self, result: dict[str, object]) -> None:
        target_ids = result.get("target_ids", [])
        if isinstance(target_ids, list):
            self._select_target_ids([str(item) for item in target_ids if str(item)])
        diagnostics = result.get("tracking_diagnostics")
        if isinstance(diagnostics, dict):
            persisted = diagnostics.get("persisted_target_frames", {})
            requested = int(diagnostics.get("requested_frame_count", 0) or 0)
            counts = (
                [int(value) for value in persisted.values() if isinstance(value, int | float)]
                if isinstance(persisted, dict)
                else []
            )
            if counts and max(counts) <= 1 and requested > 1:
                collected = diagnostics.get("collected_object_frames", {})
                collected_text = (
                    ", ".join(f"{key}={value}" for key, value in collected.items())
                    if isinstance(collected, dict)
                    else ""
                )
                QMessageBox.warning(
                    self,
                    "追踪诊断",
                    "本次请求包含 "
                    f"{requested} 帧，但服务器最终只保存了 1 个有效帧。\n"
                    f"模型收集阶段：{collected_text or '无详细数据'}。\n"
                    "这表示问题发生在模型传播或传播输出收集阶段，而不是 2D/3D 显示阶段。",
                )
        else:
            QMessageBox.information(
                self,
                "追踪诊断",
                "服务器没有返回逐阶段帧统计。当前服务器可能尚未更新到新的追踪诊断版本，"
                "因此暂时无法判断相邻帧是在模型传播、结果收集还是保存阶段丢失。",
            )

    def _select_target_ids(self, target_ids: list[str]) -> None:
        if not target_ids:
            return
        preferred = target_ids[0]
        self._table.clearSelection()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            target_id = str(item.data(Qt.ItemDataRole.UserRole) if item is not None else "")
            if target_id != preferred:
                continue
            self._table.selectRow(row)
            if item is not None:
                self._table.scrollToItem(item)
            return

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
            target = self._target_store.fetch_target(ids[0], stage=self._stage)
        except Exception:  # noqa: BLE001
            return None
        self._targets[target.id] = target
        return target

    def _delete_selected(self) -> None:
        if self._target_store is None:
            return
        ids = self._selected_ids()
        if not ids:
            return
        if QMessageBox.question(
            self, "删除目标", f"确定删除选中的 {len(ids)} 个目标吗？此操作不可恢复。"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            for target_id in ids:
                self._target_store.delete_target(target_id, stage=self._stage, hard=True)
            self.refresh()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))

    def _load_selected_mask(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from yj_studio.ui.dialogs.target_visualization_dialog import (
                TargetTrack2DDialog,
                load_tracking_frames,
            )

            frames = load_tracking_frames(
                target,
                self._target_store,
                self._volume_store,
                stage=self._stage,
            )
            if not frames:
                QMessageBox.information(self, "目标", "该目标没有可显示的追踪帧。")
                return
            dialog = TargetTrack2DDialog(
                target,
                frames,
                self,
                target_store=self._target_store,
                stage=self._stage,
                on_changed=self.refresh,
            )
            self._show_visualization_window(dialog)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _load_selected_cells(self) -> None:
        target = self._selected_target()
        if target is None or self._target_store is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._load_selected_mask3d(target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "目标", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _load_selected_mask3d(self, target: GeoTarget) -> None:
        if self._target_store is None:
            return
        result = self._target_store.fetch_mask3d_with_metadata(
            target.id, volume_id=target.volume_id, stage=self._stage
        )
        mask = np.asarray(result.mask, dtype=np.uint8)
        if mask.ndim != 3 or not np.any(mask):
            QMessageBox.information(self, "目标", "该目标还没有可显示的 3D mask。")
            return
        frame = _first_frame(target)
        target_axis = frame.axis if frame else _dominant_axis(target)
        axis = _desktop_axis(target_axis)
        index_lo = result.index_lo if result.index_lo is not None else _mask3d_index_lo(target, target_axis)
        stats = _mask3d_metadata(target, mask, result)
        self._volume_stats[target.id] = stats
        target.metadata.update(stats)
        from yj_studio.ui.dialogs.target_visualization_dialog import TargetVolume3DDialog

        dialog = TargetVolume3DDialog(
            target,
            mask,
            axis=axis,
            index_lo=index_lo,
            voxel_spacing=tuple(float(v) for v in stats["voxel_spacing"]),
            color=_target_mask_color(target.type),
            parent=self,
        )
        self._show_visualization_window(dialog)
        self._update_volume_cell(target.id)

    def _show_visualization_window(self, dialog: QDialog) -> None:
        self._visualization_windows.append(dialog)

        def _forget(*_args) -> None:
            if dialog in self._visualization_windows:
                self._visualization_windows.remove(dialog)

        dialog.destroyed.connect(_forget)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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
            target_set = self._target_store.load_targets(include_deleted=False, stage=self._stage)
            self._targets = dict(target_set.targets)
            dialog = _ReviewQueueDialog(self._target_store, _review_rows(target_set), self)
            dialog.status_changed.connect(self.refresh)
            dialog.exec()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "审校", str(exc))

    def _update_volume_cell(self, target_id: str) -> None:
        text = _volume_text(self._targets.get(target_id), self._volume_stats)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole)) == target_id:
                self._table.setItem(row, 7, QTableWidgetItem(text))
                return


def _first_frame(target: GeoTarget) -> TargetFrame | None:
    keys = target.trajectory or sorted(target.frames)
    if not keys:
        return None
    return target.frames.get(keys[0])


def _desktop_axis(axis: str) -> str:
    return {"crossline": "xline", "timeslice": "z"}.get(axis, axis)


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
    from yj_studio_core.targets import target_type_color

    return target_type_color(target_type or "unknown", alpha=0.46)


def _target_type_text(target_type: str) -> str:
    return {
        "trap": "圈闭",
        "turbidite": "浊积体",
        "fault": "断层",
        "sandbody": "砂体",
        "unknown": "未分类",
    }.get(target_type, target_type or "未分类")


def _target_status_text(status: str) -> str:
    return {
        "active": "活动",
        "to_review": "待审校",
        "lost": "追踪丢失",
        "merged": "已合并",
        "split": "已拆分",
        "confirmed": "已确认",
        "rejected": "已打回",
        "deleted": "已删除",
    }.get(status, status)


def _result_kind_text(target: GeoTarget | None) -> str:
    if target is None:
        return ""
    if target.source == "sam3_video":
        return "多帧追踪" if target.frame_count > 1 else "追踪仅1帧"
    if target.frame_count > 1:
        return "多帧目标"
    return "单帧分割"


def _mask3d_metadata(target: GeoTarget, mask: np.ndarray, result: Mask3DResult) -> dict[str, object]:
    spacing = result.voxel_spacing or DEFAULT_VOXEL_SPACING
    stats = mask_volume_stats(mask, spacing)
    voxel_count = int(result.voxel_count) if result.voxel_count is not None else int(stats["voxel_count"])
    volume_m3 = float(result.volume_m3) if result.volume_m3 is not None else float(stats["volume_m3"])
    voxel_spacing = tuple(float(v) for v in (result.voxel_spacing or stats["voxel_spacing"]))
    return {
        "mask3d_shape": [int(v) for v in mask.shape],
        "voxel_count": voxel_count,
        "voxel_spacing": list(voxel_spacing),
        "voxel_spacing_source": result.voxel_spacing_source or "default",
        "voxel_volume_m3": float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]),
        "volume_m3": volume_m3,
        "volume_source": "mask3d",
        "target_id": target.id,
    }


def _volume_text(target: GeoTarget | None, cache: dict[str, dict[str, object]]) -> str:
    metadata: dict[str, object] = {}
    if target is not None:
        metadata.update(target.metadata)
        metadata.update(cache.get(target.id, {}))
    value = metadata.get("volume_m3")
    if value is None:
        return ""
    try:
        return _format_volume(float(value))
    except (TypeError, ValueError):
        return ""


def _format_volume(value_m3: float) -> str:
    if abs(value_m3) >= 1_000_000.0:
        return f"{value_m3 / 1_000_000.0:.3f} Mm3"
    if abs(value_m3) >= 1_000.0:
        return f"{value_m3:,.0f} m3"
    return f"{value_m3:.3g} m3"


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
