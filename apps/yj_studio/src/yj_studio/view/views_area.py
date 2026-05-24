from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QTabBar, QTabWidget, QWidget

from yj_studio.data.volume_store import VolumeStore
from yj_studio.scene.layer_store import LayerStore
from yj_studio.services.section_service import SectionAxis
from yj_studio.services.view_sync_service import ViewSyncService
from yj_studio.tools import ToolManager

from .view_arbitrary_section import ViewArbitrarySection
from .view_2d_section import View2DSection


class ViewsArea(QTabWidget):
    """Central tab area containing the 3D view and any opened 2D sections."""

    section_added = pyqtSignal(str, str, str, int)
    section_removed = pyqtSignal(str)
    section_updated = pyqtSignal(str, str, int)
    current_section_changed = pyqtSignal(str)

    def __init__(
        self,
        layer_store: LayerStore,
        volume_store: VolumeStore,
        sync_service: ViewSyncService,
        tool_manager: ToolManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._layer_store = layer_store
        self._volume_store = volume_store
        self._sync_service = sync_service
        self._tool_manager = tool_manager
        self._section_ids: set[str] = set()
        self._internal_section_meta: dict[str, tuple[str, int]] = {}
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self._on_tab_close_requested)
        self.currentChanged.connect(self._on_current_changed)

    def add_primary_view(self, widget: QWidget, title: str) -> None:
        index = self.addTab(widget, title)
        self.tabBar().setTabData(index, "")
        self.tabBar().setTabButton(index, QTabBar.ButtonPosition.RightSide, None)
        if self._tool_manager is not None and not isinstance(widget, QLabel):
            self._tool_manager.attach_view(widget)

    def add_orthogonal_section(
        self,
        *,
        volume_layer_id: str,
        axis: SectionAxis,
        index: int,
    ) -> View2DSection:
        view = View2DSection(
            self._layer_store,
            self._volume_store,
            self._sync_service,
            volume_layer_id=volume_layer_id,
            axis=axis,
            index=index,
            parent=self,
        )
        view.layer_store = self._layer_store
        view.volume_store = self._volume_store
        view.view_sync = self._sync_service
        if self._tool_manager is not None:
            self._tool_manager.attach_view(view)
        view.section_updated.connect(self._on_section_updated)
        tab_index = self.addTab(view, view.title)
        self.tabBar().setTabData(tab_index, view.section_id)
        self.setCurrentIndex(tab_index)
        self._section_ids.add(view.section_id)
        self.section_added.emit(view.section_id, view.title, view.axis, view.index)
        return view

    def add_internal_section(
        self,
        widget: QWidget,
        *,
        title: str,
        axis: str,
        index: int = 0,
    ) -> str:
        section_id = str(id(widget))
        if self._tool_manager is not None:
            self._tool_manager.attach_view(widget)
        tab_index = self.addTab(widget, title)
        self.tabBar().setTabData(tab_index, section_id)
        self.setCurrentIndex(tab_index)
        self._section_ids.add(section_id)
        self._internal_section_meta[section_id] = (axis, int(index))
        self.section_added.emit(section_id, title, axis, index)
        return section_id

    def add_arbitrary_section(self, layer_id: str) -> ViewArbitrarySection:
        view = ViewArbitrarySection(self._layer_store, layer_id, parent=self)
        view.layer_store = self._layer_store
        view.volume_store = self._volume_store
        view.view_sync = self._sync_service
        if self._tool_manager is not None:
            self._tool_manager.attach_view(view)
        view.section_updated.connect(self._on_section_updated)
        tab_index = self.addTab(view, view.title)
        self.tabBar().setTabData(tab_index, view.section_id)
        self.setCurrentIndex(tab_index)
        self._section_ids.add(view.section_id)
        self.section_added.emit(view.section_id, view.title, view.axis, view.index)
        return view

    def _on_section_updated(self, section_id: str, title: str, index_value: int) -> None:
        tab_index = self._tab_index_for_section(section_id)
        if tab_index >= 0:
            self.setTabText(tab_index, title)
        self.section_updated.emit(section_id, title, index_value)

    def close_section(self, section_id: str) -> None:
        tab_index = self._tab_index_for_section(section_id)
        if tab_index < 0:
            return
        widget = self.widget(tab_index)
        if self._tool_manager is not None:
            self._tool_manager.detach_view(widget)
        self.removeTab(tab_index)
        widget.deleteLater()
        self._section_ids.discard(section_id)
        self._internal_section_meta.pop(section_id, None)
        self.section_removed.emit(section_id)

    def activate_section(self, section_id: str) -> None:
        tab_index = self._tab_index_for_section(section_id)
        if tab_index >= 0:
            self.setCurrentIndex(tab_index)

    def iter_sections(self) -> tuple[tuple[str, str, str, int], ...]:
        sections: list[tuple[str, str, str, int]] = []
        for index in range(self.count()):
            section_id = str(self.tabBar().tabData(index) or "")
            if not section_id:
                continue
            widget = self.widget(index)
            if isinstance(widget, View2DSection):
                sections.append((section_id, widget.title, widget.axis, widget.index))
            elif isinstance(widget, ViewArbitrarySection):
                sections.append((section_id, widget.title, widget.axis, widget.index))
            else:
                axis, index_value = self._internal_section_meta.get(section_id, ("internal", 0))
                sections.append((section_id, self.tabText(index), axis, index_value))
        return tuple(sections)

    def _on_tab_close_requested(self, index: int) -> None:
        section_id = str(self.tabBar().tabData(index) or "")
        if section_id:
            self.close_section(section_id)

    def _on_current_changed(self, index: int) -> None:
        if index < 0:
            return
        section_id = str(self.tabBar().tabData(index) or "")
        if section_id:
            self.current_section_changed.emit(section_id)

    def _tab_index_for_section(self, section_id: str) -> int:
        for index in range(self.count()):
            if str(self.tabBar().tabData(index) or "") == section_id:
                return index
        return -1
