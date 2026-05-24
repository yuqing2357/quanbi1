from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yj_studio.io.readers.volume_npy import VolumeSpec
from yj_studio.ui.text import section_axis_label


class SliceControlsDock(QDockWidget):
    volume_changed = pyqtSignal(str)
    slice_changed = pyqtSignal(str, int)
    clim_changed = pyqtSignal(object)
    cmap_changed = pyqtSignal(str)
    roi_changed = pyqtSignal(object)  # tuple[int]*6 or None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("剖面控制", parent)
        self._sliders: dict[str, QSlider] = {}
        self._labels: dict[str, QLabel] = {}
        self._volume_specs: dict[str, VolumeSpec] = {}
        self._build()

    def set_volume_specs(self, specs: Mapping[str, VolumeSpec]) -> None:
        self._volume_specs = dict(specs)
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for key, spec in self._volume_specs.items():
            self.volume_combo.addItem(spec.label, key)
        self.volume_combo.blockSignals(False)

    def set_current_volume(self, volume_id: str) -> None:
        index = self.volume_combo.findData(volume_id)
        if index >= 0:
            self.volume_combo.blockSignals(True)
            self.volume_combo.setCurrentIndex(index)
            self.volume_combo.blockSignals(False)

    def set_shape(self, shape: tuple[int, int, int], indices: dict[str, int]) -> None:
        limits = {"inline": shape[0], "xline": shape[1], "z": shape[2]}
        for axis, limit in limits.items():
            slider = self._sliders[axis]
            slider.blockSignals(True)
            slider.setMinimum(0)
            slider.setMaximum(max(0, limit - 1))
            value = int(indices.get(axis, max(0, limit // 2)))
            slider.setValue(max(0, min(limit - 1, value)))
            slider.blockSignals(False)
            self._labels[axis].setText(str(slider.value()))
        # Update ROI spin box limits to match the new shape.
        roi_limits = {
            "i0": shape[0] - 1,
            "i1": shape[0] - 1,
            "j0": shape[1] - 1,
            "j1": shape[1] - 1,
            "k0": shape[2] - 1,
            "k1": shape[2] - 1,
        }
        for key, limit in roi_limits.items():
            spin = self._roi_spins[key]
            spin.blockSignals(True)
            spin.setMaximum(max(0, limit))
            spin.blockSignals(False)
        # Default ROI spans the whole volume.
        defaults = {"i0": 0, "i1": shape[0] - 1, "j0": 0, "j1": shape[1] - 1, "k0": 0, "k1": shape[2] - 1}
        for key, value in defaults.items():
            spin = self._roi_spins[key]
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)

    def set_roi(self, roi: tuple[int, int, int, int, int, int] | None) -> None:
        enabled = roi is not None
        self._roi_enable.blockSignals(True)
        self._roi_enable.setChecked(enabled)
        self._roi_enable.blockSignals(False)
        if roi is not None:
            for key, value in zip(("i0", "i1", "j0", "j1", "k0", "k1"), roi):
                spin = self._roi_spins[key]
                spin.blockSignals(True)
                spin.setValue(int(value))
                spin.blockSignals(False)
        for spin in self._roi_spins.values():
            spin.setEnabled(enabled)

    def set_slice_index(self, axis: str, index: int) -> None:
        slider = self._sliders[axis]
        value = max(slider.minimum(), min(slider.maximum(), int(index)))
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)
        self._labels[axis].setText(str(value))

    def set_clim(self, clim: tuple[float, float] | None) -> None:
        if clim is None:
            return
        self.clim_min.blockSignals(True)
        self.clim_max.blockSignals(True)
        self.clim_min.setValue(float(clim[0]))
        self.clim_max.setValue(float(clim[1]))
        self.clim_min.blockSignals(False)
        self.clim_max.blockSignals(False)

    def set_cmap(self, cmap: str) -> None:
        index = self.cmap_combo.findText(cmap)
        if index < 0:
            self.cmap_combo.addItem(cmap)
            index = self.cmap_combo.findText(cmap)
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.setCurrentIndex(index)
        self.cmap_combo.blockSignals(False)

    def _build(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        form = QFormLayout()
        self.volume_combo = QComboBox(root)
        self.volume_combo.currentIndexChanged.connect(self._emit_volume_changed)
        form.addRow("体数据", self.volume_combo)

        self.cmap_combo = QComboBox(root)
        self.cmap_combo.addItems(["Petrel", "gray", "seismic", "viridis", "turbo", "hsv", "RdBu", "tab10"])
        self.cmap_combo.currentTextChanged.connect(self.cmap_changed.emit)
        form.addRow("色图", self.cmap_combo)

        self.clim_min = _clim_spinbox(root)
        self.clim_max = _clim_spinbox(root)
        self.clim_min.valueChanged.connect(self._emit_clim_changed)
        self.clim_max.valueChanged.connect(self._emit_clim_changed)
        form.addRow("最小", self.clim_min)
        form.addRow("最大", self.clim_max)
        layout.addLayout(form)

        group = QGroupBox("剖面位置", root)
        group_layout = QFormLayout(group)
        for axis in ("inline", "xline", "z"):
            slider = QSlider(Qt.Orientation.Horizontal, group)
            value_label = QLabel("0", group)
            slider.valueChanged.connect(lambda value, current_axis=axis: self._emit_slice_changed(current_axis, value))
            row = QWidget(group)
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(slider)
            row_layout.addWidget(value_label)
            self._sliders[axis] = slider
            self._labels[axis] = value_label
            group_layout.addRow(section_axis_label(axis), row)
        layout.addWidget(group)

        roi_group = QGroupBox("ROI 裁剪", root)
        roi_layout = QGridLayout(roi_group)
        self._roi_enable = QCheckBox("启用", roi_group)
        self._roi_enable.toggled.connect(self._on_roi_toggled)
        roi_layout.addWidget(self._roi_enable, 0, 0, 1, 3)
        self._roi_spins: dict[str, QSpinBox] = {}
        labels_grid = [
            ("纵向", "i0", "i1", 1),
            ("横向", "j0", "j1", 2),
            ("Z向", "k0", "k1", 3),
        ]
        for label_text, lo_key, hi_key, row in labels_grid:
            roi_layout.addWidget(QLabel(label_text, roi_group), row, 0)
            lo = QSpinBox(roi_group)
            hi = QSpinBox(roi_group)
            for spin, key in ((lo, lo_key), (hi, hi_key)):
                spin.setRange(0, 0)
                spin.setEnabled(False)
                spin.valueChanged.connect(self._emit_roi_changed)
                self._roi_spins[key] = spin
            roi_layout.addWidget(lo, row, 1)
            roi_layout.addWidget(hi, row, 2)
        reset_btn = QPushButton("重置 ROI", roi_group)
        reset_btn.clicked.connect(self._reset_roi)
        roi_layout.addWidget(reset_btn, 4, 0, 1, 3)
        layout.addWidget(roi_group)

        layout.addStretch(1)
        self.setWidget(root)

    def _emit_volume_changed(self) -> None:
        volume_id = self.volume_combo.currentData()
        if volume_id:
            self.volume_changed.emit(str(volume_id))

    def _emit_slice_changed(self, axis: str, value: int) -> None:
        self._labels[axis].setText(str(value))
        self.slice_changed.emit(axis, int(value))

    def _emit_clim_changed(self) -> None:
        self.clim_changed.emit((float(self.clim_min.value()), float(self.clim_max.value())))

    def _on_roi_toggled(self, checked: bool) -> None:
        for spin in self._roi_spins.values():
            spin.setEnabled(checked)
        if checked:
            self._emit_roi_changed()
        else:
            self.roi_changed.emit(None)

    def _emit_roi_changed(self) -> None:
        if not self._roi_enable.isChecked():
            return
        roi = tuple(int(self._roi_spins[key].value()) for key in ("i0", "i1", "j0", "j1", "k0", "k1"))
        i0, i1, j0, j1, k0, k1 = roi
        if i1 < i0 or j1 < j0 or k1 < k0:
            return
        self.roi_changed.emit(roi)

    def _reset_roi(self) -> None:
        self._roi_enable.setChecked(False)


def _clim_spinbox(parent: QWidget) -> QDoubleSpinBox:
    spinbox = QDoubleSpinBox(parent)
    spinbox.setDecimals(4)
    spinbox.setRange(-1_000_000.0, 1_000_000.0)
    spinbox.setSingleStep(0.1)
    return spinbox
