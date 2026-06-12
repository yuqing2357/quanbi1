from __future__ import annotations

import numpy as np

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import WellLayer, WellLogLayer
from yj_studio.view.highlight import HIGHLIGHT_COLOR, highlight_color, is_layer_highlighted, selected_well_names


def test_selected_well_names_tracks_well_and_log_selection() -> None:
    store = LayerStore()
    well = WellLayer(name="W1", well_name="W1", head_position=(1.0, 2.0, 3.0))
    log = WellLogLayer(name="W1 POR", well_name="W1", mode="por", samples=np.zeros((1, 4), dtype=np.float32))
    store.add(well)
    store.add(log)
    store.select([log.id])

    assert selected_well_names(store) == {"W1"}
    assert is_layer_highlighted(well, {log.id}, {"W1"})
    assert highlight_color((0.1, 0.2, 0.3, 1.0), True) == HIGHLIGHT_COLOR[:3]
