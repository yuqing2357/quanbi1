from __future__ import annotations

import numpy as np

from yj_studio.scene.layers import PolygonLayer
from yj_studio.scene.manual_geometry import manual_geometry_points, project_points_to_section
from yj_studio.tools.point_pick_tool import _layer_distance
from yj_studio.tools.box_pick_tool import _event_xy
from yj_studio.view.renderers.manual_geometry_renderer import ManualGeometryRenderer
from yj_studio.scene.layers import WellLogLayer


def test_manual_geometry_projects_points_to_orthogonal_section() -> None:
    layer = PolygonLayer(
        name="P1",
        vertices=np.asarray([[10.0, 1.0, 2.0], [10.0, 3.0, 4.0]], dtype=np.float32),
    )

    points = manual_geometry_points(layer)
    projected = project_points_to_section(points, "inline", 10)

    assert projected is not None
    x, y = projected
    np.testing.assert_array_equal(x, np.asarray([1.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(y, np.asarray([2.0, 4.0], dtype=np.float32))


def test_box_pick_uses_matplotlib_data_coordinates() -> None:
    class Event:
        xdata = 3.5
        ydata = 7.25
        inaxes = object()
        x = 100
        y = 200

    assert _event_xy(object(), Event()) == (3.5, 7.25)


def test_manual_geometry_renderer_uses_stable_actor_names() -> None:
    class Plotter:
        def __init__(self) -> None:
            self.added: list[str] = []
            self.removed: list[str] = []
            self.render_count = 0

        def add_mesh(self, _mesh, *, name: str, **_kwargs) -> None:
            self.added.append(name)

        def remove_actor(self, name: str, **_kwargs) -> None:
            self.removed.append(name)

        def render(self) -> None:
            self.render_count += 1

    layer = PolygonLayer(
        id="poly-1",
        name="P1",
        vertices=np.asarray([[10.0, 1.0, 2.0], [10.0, 3.0, 4.0]], dtype=np.float32),
    )
    plotter = Plotter()

    ManualGeometryRenderer(plotter).render(layer)

    assert "polygon-poly-1-line" in plotter.added
    assert "polygon-poly-1-points" in plotter.added
    assert plotter.render_count == 1


def test_manual_geometry_renderer_uses_display_z_when_count_is_known() -> None:
    class Plotter:
        def __init__(self) -> None:
            self.meshes = {}

        def add_mesh(self, mesh, *, name: str, **_kwargs) -> None:
            self.meshes[name] = mesh

        def remove_actor(self, _name: str, **_kwargs) -> None:
            return None

        def render(self) -> None:
            return None

    layer = PolygonLayer(
        id="poly-z",
        name="PZ",
        vertices=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 3.0]], dtype=np.float32),
    )
    plotter = Plotter()

    ManualGeometryRenderer(plotter).render(layer, z_count=5)

    mesh = plotter.meshes["polygon-poly-z-points"]
    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([3.0, 1.0], dtype=np.float32))


def test_point_pick_distance_handles_well_log_points_in_3d() -> None:
    layer = WellLogLayer(
        name="Log",
        samples=np.asarray([[1.0, 2.0, 3.0, 0.2], [5.0, 6.0, 7.0, 0.3]], dtype=np.float32),
    )

    assert _layer_distance(object(), layer, (1.0, 2.0, 3.0), axis=None, index=0) == 0.0
