from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.scene import LayerStore
from yj_studio.scene.layers import HorizonLayer, PolygonLayer
from yj_studio.ui.widgets.schema_form import SchemaForm


class _Params(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iterations: int = Field(default=10, ge=1, le=100)
    enabled: bool = Field(default=True)
    label: str = Field(default="run")


def test_schema_form_renders_pydantic_fields(qapp) -> None:
    store = LayerStore()
    form = SchemaForm(store)
    form.set_algorithm(_Params, layer_inputs=None)

    collected = form.collect()
    assert collected["params"] == {
        "threshold": 0.5,
        "iterations": 10,
        "enabled": True,
        "label": "run",
    }
    assert form.validate() is None


def test_schema_form_lists_matching_layers(qapp) -> None:
    store = LayerStore()
    horizon = HorizonLayer(name="T1", sample=np.zeros((4, 4), dtype=np.float32))
    polygon = PolygonLayer(name="P1", vertices=np.zeros((3, 3), dtype=np.float32))
    horizon_id = store.add(horizon)
    store.add(polygon)

    form = SchemaForm(store)
    form.set_algorithm(_Params, layer_inputs={"top": "horizon"})

    combo = form._layer_widgets["top"]
    # The placeholder counts as one entry, plus the single matching horizon.
    assert combo.count() == 2
    assert combo.itemData(1) == horizon_id
    # Validation should fail until the user picks a layer.
    assert form.validate() == "Missing layer input: top"
    combo.setCurrentIndex(1)
    assert form.validate() is None
    collected = form.collect()
    assert collected["layers"] == {"top": horizon_id}
