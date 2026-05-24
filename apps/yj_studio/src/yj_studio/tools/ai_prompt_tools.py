"""Tools that feed prompts into the SAM3 pipeline.

These behave very differently from the regular ``BoxPickTool`` /
``PointPickTool`` — instead of selecting layers in the store, they capture
the *coordinates* of a click / drag on a 2D section view and forward them
to the active ``AIService``. The AI Dock listens for ``box_prompt_added`` /
``point_prompt_added`` signals and appends them to its prompt collection.

Both tools only fire when the view exposes ``axis`` + ``index`` attributes
(i.e. a 2D inline / xline / Z section), since SAM3 needs to know which
slice the prompt belongs to. Activating them on the 3D view is a no-op.

The tools also paint a lightweight rubber-band preview on the matplotlib
canvas so the user can actually see where they are dragging. Without this
the user only finds out the box location after release, which is bad UX.
"""

from __future__ import annotations

from typing import Any

from yj_studio.tools._helpers import (
    event_left_button,
    tool_notify,
    view_rect_from_events,
)
from yj_studio.tools.tool import InteractionTool


def _ai_service(view: Any):
    manager = getattr(view, "tool_manager", None)
    if manager is None:
        return None
    service = manager.service("ai_service") if hasattr(manager, "service") else None
    return service


def _slice_axis(view: Any) -> str | None:
    axis = getattr(view, "axis", None)
    return axis if axis in {"inline", "xline", "z"} else None


def _slice_index(view: Any) -> int:
    return int(getattr(view, "index", 0) or 0)


def _event_xy(view: Any, event: Any) -> tuple[float, float] | None:
    xdata = getattr(event, "xdata", None)
    ydata = getattr(event, "ydata", None)
    if xdata is None or ydata is None:
        return None
    return float(xdata), float(ydata)


def _section_axes(view: Any):
    """Return the matplotlib Axes used by a View2DSection, or None."""

    axes = getattr(view, "_axes", None)
    return axes


def _section_canvas(view: Any):
    return getattr(view, "_canvas", None)


def _draw_idle(view: Any) -> None:
    canvas = _section_canvas(view)
    if canvas is not None:
        canvas.draw_idle()


class AIPointPromptTool(InteractionTool):
    """Single-click on a section view → emits a point prompt.

    Each click also leaves a small visible marker on the section so the user
    can see where they have already prompted.
    """

    def __init__(self) -> None:
        super().__init__(
            id="ai_point_prompt",
            label="AI Point Prompt",
            icon="target",
            cursor="crosshair",
        )
        # markers keyed by view → list of matplotlib artists
        self._markers: dict[int, list] = {}

    def deactivate(self, view: Any) -> None:
        self._clear_markers(view)

    def on_mouse_press(self, view: Any, event: Any) -> bool:
        axis = _slice_axis(view)
        if axis is None:
            tool_notify(view, "Open a 2D section before collecting AI point prompts")
            return False
        if not event_left_button(event):
            return False
        xy = _event_xy(view, event)
        if xy is None:
            return False
        service = _ai_service(view)
        if service is None:
            tool_notify(view, "AI service not available")
            return False
        service.emit_point_prompt(axis, _slice_index(view), xy[0], xy[1])
        self._paint_marker(view, xy)
        tool_notify(view, f"Point prompt @ ({xy[0]:.0f}, {xy[1]:.0f})")
        return True

    def _paint_marker(self, view: Any, xy: tuple[float, float]) -> None:
        axes = _section_axes(view)
        if axes is None:
            return
        artists = []
        # A small cross + outline circle is easier to spot than a single dot
        # on top of a noisy seismic background.
        artists.append(
            axes.plot(
                [xy[0]],
                [xy[1]],
                marker="x",
                color="#ffeb3b",
                markersize=10,
                markeredgewidth=2,
                linestyle="none",
                zorder=20,
            )[0]
        )
        artists.append(
            axes.scatter(
                [xy[0]],
                [xy[1]],
                s=120,
                facecolors="none",
                edgecolors="#ffeb3b",
                linewidths=1.4,
                zorder=20,
            )
        )
        self._markers.setdefault(id(view), []).extend(artists)
        _draw_idle(view)

    def _clear_markers(self, view: Any) -> None:
        artists = self._markers.pop(id(view), [])
        for artist in artists:
            try:
                artist.remove()
            except Exception:  # noqa: BLE001
                pass
        if artists:
            _draw_idle(view)


class AIBoxPromptTool(InteractionTool):
    """Drag a rectangle on a section view → emits a box prompt.

    Shows a live rubber-band rectangle while the user drags, plus a thin
    outline for each completed box so the user can recall what has been
    submitted without checking the dock list.
    """

    def __init__(self) -> None:
        super().__init__(
            id="ai_box_prompt",
            label="AI Box Prompt",
            icon="square",
            cursor="crosshair",
        )
        self._start: tuple[float, float] | None = None
        self._preview_artists: dict[int, list] = {}
        self._final_artists: dict[int, list] = {}

    def deactivate(self, view: Any) -> None:
        self._start = None
        self._clear_preview(view)
        self._clear_final(view)

    def on_mouse_press(self, view: Any, event: Any) -> bool:
        if _slice_axis(view) is None or not event_left_button(event):
            return False
        xy = _event_xy(view, event)
        if xy is None:
            return False
        self._start = xy
        self._clear_preview(view)
        return True

    def on_mouse_move(self, view: Any, event: Any) -> bool:
        if self._start is None:
            return False
        xy = _event_xy(view, event)
        if xy is None:
            return False
        self._update_preview(view, self._start, xy)
        return True

    def on_mouse_release(self, view: Any, event: Any) -> bool:
        if self._start is None:
            return False
        axis = _slice_axis(view)
        end = _event_xy(view, event)
        start, self._start = self._start, None
        self._clear_preview(view)
        if end is None or axis is None:
            return False
        service = _ai_service(view)
        if service is None:
            tool_notify(view, "AI service not available")
            return False
        x_min, y_min, x_max, y_max = view_rect_from_events(start, end)
        if x_max - x_min < 1.0 or y_max - y_min < 1.0:
            tool_notify(view, "AI box prompt ignored (too small)")
            return False
        service.emit_box_prompt(axis, _slice_index(view), x_min, y_min, x_max, y_max)
        self._paint_final(view, x_min, y_min, x_max, y_max)
        tool_notify(
            view,
            f"Box prompt [{x_min:.0f},{y_min:.0f} → {x_max:.0f},{y_max:.0f}]",
        )
        return True

    # ------------------------------------------------------------------ painting

    def _update_preview(
        self,
        view: Any,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        axes = _section_axes(view)
        if axes is None:
            return
        from matplotlib.patches import Rectangle

        self._clear_preview(view)
        x0, y0, x1, y1 = view_rect_from_events(start, end)
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.8,
            edgecolor="#ffeb3b",
            facecolor="#ffeb3b22",  # translucent fill
            linestyle="--",
            zorder=21,
        )
        axes.add_patch(rect)
        self._preview_artists.setdefault(id(view), []).append(rect)
        _draw_idle(view)

    def _clear_preview(self, view: Any) -> None:
        artists = self._preview_artists.pop(id(view), [])
        for artist in artists:
            try:
                artist.remove()
            except Exception:  # noqa: BLE001
                pass
        if artists:
            _draw_idle(view)

    def _paint_final(
        self,
        view: Any,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        axes = _section_axes(view)
        if axes is None:
            return
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.4,
            edgecolor="#4caf50",
            facecolor="none",
            zorder=20,
        )
        axes.add_patch(rect)
        self._final_artists.setdefault(id(view), []).append(rect)
        _draw_idle(view)

    def _clear_final(self, view: Any) -> None:
        artists = self._final_artists.pop(id(view), [])
        for artist in artists:
            try:
                artist.remove()
            except Exception:  # noqa: BLE001
                pass
        if artists:
            _draw_idle(view)
