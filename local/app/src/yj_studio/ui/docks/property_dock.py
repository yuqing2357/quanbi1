from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QUndoStack
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from yj_studio.data import VolumeStore, estimate_volume_clim
from yj_studio.scene.layer import Layer
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import VolumeLayer
from yj_studio.scene.undo_commands import (
    RenameLayerCommand,
    SetColorCommand,
    SetLayerFieldCommand,
    SetOpacityCommand,
    SetVisibleCommand,
)
from yj_studio.ui.text import layer_kind_label


_VOLUME_CMAPS = ["Petrel", "gray", "seismic", "viridis", "plasma", "magma", "jet", "RdBu_r"]


class _ColorSwatch(QPushButton):
    color_changed = pyqtSignal(tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(48, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
        self.clicked.connect(self._open_dialog)

    def set_color(self, color: tuple[float, float, float, float]) -> None:
        self._color = color
        r, g, b, _a = color
        self.setStyleSheet(
            f"background-color: rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)}); "
            "border: 1px solid #888;"
        )

    def color(self) -> tuple[float, float, float, float]:
        return self._color

    def _open_dialog(self) -> None:
        r, g, b, a = self._color
        initial = QColor.fromRgbF(r, g, b, a)
        result = QColorDialog.getColor(
            initial,
            self,
            self.tr("选择颜色"),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not result.isValid():
            return
        new_color = (
            float(result.redF()),
            float(result.greenF()),
            float(result.blueF()),
            float(result.alphaF()),
        )
        self.set_color(new_color)
        self.color_changed.emit(new_color)


class PropertyDock(QDockWidget):
    """Edit the currently selected layer's common and type-specific properties.

    All edits go through ``QUndoStack`` so they are reversible.
    """

    def __init__(
        self,
        layer_store: LayerStore,
        volume_store: VolumeStore,
        undo_stack: QUndoStack,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("属性", parent)
        self._layer_store = layer_store
        self._volume_store = volume_store
        self._undo_stack = undo_stack
        self._current_layer_id: str | None = None
        self._updating = False

        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(8, 8, 8, 8)

        self._header_label = QLabel(self.tr("未选择图层"), body)
        outer.addWidget(self._header_label)

        self._form = QFormLayout()
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        outer.addLayout(self._form)

        # Common fields
        self._name_edit = QLineEdit(body)
        self._name_edit.editingFinished.connect(self._commit_name)
        self._form.addRow(self.tr("名称"), self._name_edit)

        self._visible_check = QCheckBox(body)
        self._visible_check.toggled.connect(self._commit_visible)
        self._form.addRow(self.tr("可见"), self._visible_check)

        self._color_swatch = _ColorSwatch(body)
        self._color_swatch.color_changed.connect(self._commit_color)
        self._form.addRow(self.tr("颜色"), self._color_swatch)

        opacity_row = QWidget(body)
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal, opacity_row)
        self._opacity_slider.setRange(0, 100)
        self._opacity_value = QLabel("100%", opacity_row)
        self._opacity_value.setFixedWidth(40)
        opacity_layout.addWidget(self._opacity_slider, 1)
        opacity_layout.addWidget(self._opacity_value)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider)
        self._form.addRow(self.tr("不透明度"), opacity_row)

        # Volume-specific
        self._cmap_label = QLabel(self.tr("色图"), body)
        self._cmap_combo = QComboBox(body)
        self._cmap_combo.setEditable(True)
        self._cmap_combo.addItems(_VOLUME_CMAPS)
        self._cmap_combo.activated.connect(self._commit_cmap)
        self._cmap_combo.lineEdit().editingFinished.connect(self._commit_cmap_text)
        self._form.addRow(self._cmap_label, self._cmap_combo)

        self._clim_min = QDoubleSpinBox(body)
        self._clim_min.setRange(-1e9, 1e9)
        self._clim_min.setDecimals(4)
        self._clim_max = QDoubleSpinBox(body)
        self._clim_max.setRange(-1e9, 1e9)
        self._clim_max.setDecimals(4)
        self._clim_min.editingFinished.connect(self._commit_clim)
        self._clim_max.editingFinished.connect(self._commit_clim)
        clim_row = QWidget(body)
        clim_layout = QHBoxLayout(clim_row)
        clim_layout.setContentsMargins(0, 0, 0, 0)
        clim_layout.addWidget(self._clim_min)
        clim_layout.addWidget(QLabel("–", clim_row))
        clim_layout.addWidget(self._clim_max)
        reset_clim = QPushButton(self.tr("重置"), clim_row)
        reset_clim.clicked.connect(self._reset_clim_from_volume)
        clim_layout.addWidget(reset_clim)
        self._clim_label = QLabel(self.tr("范围"), body)
        self._form.addRow(self._clim_label, clim_row)

        outer.addStretch(1)
        self.setWidget(body)

        layer_store.selection_changed.connect(self._on_selection_changed)
        layer_store.layer_changed.connect(self._on_layer_changed)
        layer_store.layer_removed.connect(self._on_layer_removed)

        self._show_volume_fields(False)
        self._set_form_enabled(False)

    # ------------------------------------------------------------------ helpers

    def _current_layer(self) -> Layer | None:
        if self._current_layer_id is None:
            return None
        try:
            return self._layer_store.get(self._current_layer_id)
        except KeyError:
            return None

    def _set_form_enabled(self, enabled: bool) -> None:
        for widget in (
            self._name_edit,
            self._visible_check,
            self._color_swatch,
            self._opacity_slider,
            self._cmap_combo,
            self._clim_min,
            self._clim_max,
        ):
            widget.setEnabled(enabled)

    def _show_volume_fields(self, show: bool) -> None:
        for widget in (
            self._cmap_label,
            self._cmap_combo,
            self._clim_label,
            self._clim_min,
            self._clim_max,
        ):
            widget.setVisible(show)

    # ------------------------------------------------------------------ slots

    def _on_selection_changed(self, layer_ids: list[str]) -> None:
        layer_id = layer_ids[0] if layer_ids else None
        self._current_layer_id = layer_id
        self._refresh()

    def _on_layer_changed(self, layer_id: str, _field: str) -> None:
        if layer_id == self._current_layer_id:
            self._refresh()

    def _on_layer_removed(self, layer_id: str) -> None:
        if layer_id == self._current_layer_id:
            self._current_layer_id = None
            self._refresh()

    # ------------------------------------------------------------------ refresh

    def _refresh(self) -> None:
        layer = self._current_layer()
        if layer is None:
            self._header_label.setText(self.tr("未选择图层"))
            self._set_form_enabled(False)
            self._show_volume_fields(False)
            self._updating = True
            try:
                self._name_edit.setText("")
                self._visible_check.setChecked(False)
                self._color_swatch.set_color((1.0, 1.0, 1.0, 1.0))
                self._opacity_slider.setValue(100)
                self._opacity_value.setText("100%")
            finally:
                self._updating = False
            return

        self._header_label.setText(
            self.tr("{kind}：{name}").format(kind=layer_kind_label(layer.kind), name=layer.name)
        )
        self._set_form_enabled(not layer.locked)
        self._updating = True
        try:
            self._name_edit.setText(layer.name)
            self._visible_check.setChecked(bool(layer.visible))
            self._color_swatch.set_color(tuple(float(c) for c in layer.color))
            opacity_pct = int(round(float(layer.opacity) * 100))
            self._opacity_slider.setValue(opacity_pct)
            self._opacity_value.setText(f"{opacity_pct}%")
        finally:
            self._updating = False

        is_volume = isinstance(layer, VolumeLayer)
        self._show_volume_fields(is_volume)
        if is_volume:
            self._refresh_volume_fields(layer)

    def _refresh_volume_fields(self, layer: VolumeLayer) -> None:
        self._updating = True
        try:
            if layer.cmap and self._cmap_combo.findText(layer.cmap) < 0:
                self._cmap_combo.addItem(layer.cmap)
            self._cmap_combo.setCurrentText(layer.cmap or "")
            clim = layer.clim or (0.0, 1.0)
            self._clim_min.setValue(float(clim[0]))
            self._clim_max.setValue(float(clim[1]))
        finally:
            self._updating = False

    # ------------------------------------------------------------------ commits

    def _push(self, command: Any) -> None:
        self._undo_stack.push(command)

    def _commit_name(self) -> None:
        if self._updating:
            return
        layer = self._current_layer()
        if layer is None:
            return
        new_name = self._name_edit.text().strip()
        if not new_name or new_name == layer.name:
            return
        self._push(RenameLayerCommand(self._layer_store, layer.id, new_name))

    def _commit_visible(self, checked: bool) -> None:
        if self._updating:
            return
        layer = self._current_layer()
        if layer is None or bool(layer.visible) == bool(checked):
            return
        self._push(SetVisibleCommand(self._layer_store, layer.id, bool(checked)))

    def _commit_color(self, color: tuple[float, float, float, float]) -> None:
        if self._updating:
            return
        layer = self._current_layer()
        if layer is None:
            return
        current = tuple(float(c) for c in layer.color)
        if current == color:
            return
        self._push(SetColorCommand(self._layer_store, layer.id, color))

    def _on_opacity_slider(self, value: int) -> None:
        self._opacity_value.setText(f"{value}%")
        if self._updating:
            return
        layer = self._current_layer()
        if layer is None:
            return
        new_opacity = float(value) / 100.0
        if abs(new_opacity - float(layer.opacity)) < 1e-6:
            return
        self._push(SetOpacityCommand(self._layer_store, layer.id, new_opacity))

    def _commit_cmap(self, _index: int) -> None:
        self._commit_cmap_text()

    def _commit_cmap_text(self) -> None:
        if self._updating:
            return
        layer = self._current_layer()
        if not isinstance(layer, VolumeLayer):
            return
        new_cmap = self._cmap_combo.currentText().strip()
        if not new_cmap or new_cmap == layer.cmap:
            return
        self._push(
            SetLayerFieldCommand(
                self._layer_store, layer.id, "cmap", new_cmap, text="修改色图"
            )
        )

    def _commit_clim(self) -> None:
        if self._updating:
            return
        layer = self._current_layer()
        if not isinstance(layer, VolumeLayer):
            return
        new_clim = (float(self._clim_min.value()), float(self._clim_max.value()))
        if new_clim[0] >= new_clim[1]:
            return
        if layer.clim is not None and tuple(float(v) for v in layer.clim) == new_clim:
            return
        self._push(
            SetLayerFieldCommand(
                self._layer_store, layer.id, "clim", new_clim, text="修改显示范围"
            )
        )

    def _reset_clim_from_volume(self) -> None:
        layer = self._current_layer()
        if not isinstance(layer, VolumeLayer) or layer.shape is None:
            return
        try:
            volume = self._volume_store.get_volume(layer.volume_id)
        except Exception:
            return
        slice_indices = layer.slice_indices
        vmin, vmax = estimate_volume_clim(
            layer.volume_id,
            volume,
            slice_indices.get("inline", layer.shape[0] // 2),
            slice_indices.get("xline", layer.shape[1] // 2),
            slice_indices.get("z", layer.shape[2] // 2),
        )
        new_clim = (float(vmin), float(vmax))
        if layer.clim is not None and tuple(float(v) for v in layer.clim) == new_clim:
            return
        self._push(
            SetLayerFieldCommand(
                self._layer_store, layer.id, "clim", new_clim, text="重置显示范围"
            )
        )
