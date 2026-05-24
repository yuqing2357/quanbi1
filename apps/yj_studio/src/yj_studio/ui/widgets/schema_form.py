"""Render a pydantic v2 model + a layer-input spec as a Qt form widget.

The form is regenerated from the model each time ``set_algorithm`` is called,
so the AlgorithmDock can swap forms whenever the user picks a different
algorithm. ``collect()`` returns ``{"params": {...}, "layers": {role: layer_id}}``;
``validate()`` runs the pydantic model and surfaces the error string, or
returns ``None`` on success.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, get_args, get_origin

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from yj_studio.scene.layer_store import LayerStore
from yj_studio.ui.text import layer_kind_label, parameter_name_label, parameter_role_label


class SchemaForm(QWidget):
    """A Qt form rendered from a pydantic v2 model and a layer-input spec."""

    changed = pyqtSignal()

    def __init__(self, layer_store: LayerStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_store = layer_store
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout = QFormLayout()
        self._form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._layout.addLayout(self._form_layout)
        self._layout.addStretch(1)

        self._param_widgets: dict[str, QWidget] = {}
        self._layer_widgets: dict[str, QComboBox] = {}
        self._current_model: type[BaseModel] | None = None
        self._current_layer_spec: dict[str, str] = {}

    # ------------------------------------------------------------------ API

    def set_algorithm(
        self,
        model: type[BaseModel] | None,
        layer_inputs: dict[str, str] | None,
    ) -> None:
        """Clear the form, then rebuild it for ``model`` + ``layer_inputs``.

        ``layer_inputs`` maps role name → ``"kind1|kind2"`` filter; a combo
        will be added per role listing matching layers from the store.
        """

        self._clear()
        self._current_model = model
        self._current_layer_spec = dict(layer_inputs or {})

        for role, kind_filter in self._current_layer_spec.items():
            combo = self._build_layer_combo(role, kind_filter)
            self._layer_widgets[role] = combo
            self._form_layout.addRow(parameter_role_label(role), combo)

        if model is not None:
            for name, field in model.model_fields.items():
                widget = self._build_param_widget(name, field)
                if widget is None:
                    continue
                self._param_widgets[name] = widget
                self._form_layout.addRow(parameter_name_label(name), widget)
        self.changed.emit()

    def collect(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self._current_model is not None:
            for name, widget in self._param_widgets.items():
                params[name] = _widget_value(widget)
        layers: dict[str, str] = {}
        for role, combo in self._layer_widgets.items():
            layer_id = combo.currentData()
            if layer_id:
                layers[role] = str(layer_id)
        return {"params": params, "layers": layers}

    def validate(self) -> str | None:
        if self._current_model is None:
            return None
        collected = self.collect()
        try:
            self._current_model.model_validate(collected["params"])
        except ValidationError as exc:
            return str(exc)
        for role in self._current_layer_spec:
            if role not in collected["layers"]:
                return f"缺少图层输入：{parameter_role_label(role)}"
        return None

    def refresh_layer_choices(self) -> None:
        """Re-populate every LayerRef combo. Call when LayerStore changes."""

        for role, combo in self._layer_widgets.items():
            kind_filter = self._current_layer_spec.get(role, "")
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for label, layer_id in self._iter_matching_layers(kind_filter):
                combo.addItem(label, layer_id)
            idx = combo.findData(previous)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self.changed.emit()

    # ------------------------------------------------------------------ internals

    def _clear(self) -> None:
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(0)
        self._param_widgets.clear()
        self._layer_widgets.clear()

    def _build_layer_combo(self, role: str, kind_filter: str) -> QComboBox:
        combo = QComboBox(self)
        human_filter = "、".join(layer_kind_label(token.strip()) for token in kind_filter.split("|") if token.strip())
        combo.setToolTip(f"{parameter_role_label(role)}：可选 {human_filter or '全部类型'}")
        combo.addItem("— 请选择 —", "")
        for label, layer_id in self._iter_matching_layers(kind_filter):
            combo.addItem(label, layer_id)
        combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        return combo

    def _iter_matching_layers(self, kind_filter: str) -> Iterable[tuple[str, str]]:
        wanted = {token.strip() for token in kind_filter.split("|") if token.strip()}
        for layer in self._layer_store.iter_layers():
            if wanted and layer.kind not in wanted:
                continue
            yield f"{layer.name} [{layer_kind_label(layer.kind)}]", layer.id

    def _build_param_widget(self, name: str, field: FieldInfo) -> QWidget | None:
        annotation = field.annotation
        default = field.default

        # Literal[...] / Enum → combobox
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            if origin.__name__ == "Literal" or all(isinstance(a, str | int) for a in args):
                combo = QComboBox(self)
                for value in args:
                    combo.addItem(str(value), value)
                if default is not None:
                    idx = combo.findData(default)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
                _apply_description(combo, field.description)
                return combo

        if annotation is bool:
            check = QCheckBox(self)
            check.setChecked(bool(default) if default is not None else False)
            check.toggled.connect(lambda _b: self.changed.emit())
            _apply_description(check, field.description)
            return check

        if annotation is int:
            spin = QSpinBox(self)
            lo, hi = _numeric_range(field, default_int=True)
            spin.setRange(int(lo), int(hi))
            if default is not None:
                spin.setValue(int(default))
            spin.valueChanged.connect(lambda _v: self.changed.emit())
            _apply_description(spin, field.description)
            return spin

        if annotation is float:
            spin = QDoubleSpinBox(self)
            spin.setDecimals(4)
            lo, hi = _numeric_range(field, default_int=False)
            spin.setRange(float(lo), float(hi))
            if default is not None:
                spin.setValue(float(default))
            spin.valueChanged.connect(lambda _v: self.changed.emit())
            _apply_description(spin, field.description)
            return spin

        if annotation is str:
            line = QLineEdit(self)
            if default is not None:
                line.setText(str(default))
            line.textChanged.connect(lambda _t: self.changed.emit())
            _apply_description(line, field.description)
            return line

        # Unsupported types are skipped silently — log and rely on the
        # algorithm having sensible defaults.
        return None


def _widget_value(widget: QWidget) -> Any:
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return data if data is not None else widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    return None


def _numeric_range(field: FieldInfo, *, default_int: bool) -> tuple[float, float]:
    lo = -1e9 if default_int else -1e12
    hi = 1e9 if default_int else 1e12
    for meta in field.metadata or ():
        for attr_name, op in (("ge", "ge"), ("gt", "ge"), ("le", "le"), ("lt", "le")):
            value = getattr(meta, attr_name, None)
            if value is None:
                continue
            if op == "ge":
                lo = max(lo, float(value))
            else:
                hi = min(hi, float(value))
    return lo, hi


def _apply_description(widget: QWidget, description: str | None) -> None:
    if description:
        widget.setToolTip(description)
