from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QDockWidget

from yj_studio.ui.text import section_axis_label


class SectionNavigatorDock(QDockWidget):
    """List opened 2D section views."""

    section_activated = pyqtSignal(str)
    section_close_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("剖面", parent)
        self._items: dict[str, QTreeWidgetItem] = {}

        content = QWidget(self)
        layout = QVBoxLayout(content)
        self.tree = QTreeWidget(content)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["名称", "轴", "索引"])
        self.tree.itemDoubleClicked.connect(self._activate_item)
        layout.addWidget(self.tree)

        close_button = QPushButton("关闭", content)
        close_button.clicked.connect(self._close_current)
        layout.addWidget(close_button)
        self.setWidget(content)

    def add_section(self, section_id: str, title: str, axis: str, index: int) -> None:
        item = QTreeWidgetItem([title, section_axis_label(axis), str(index)])
        item.setData(0, Qt.ItemDataRole.UserRole, section_id)
        self._items[section_id] = item
        self.tree.addTopLevelItem(item)
        self.tree.resizeColumnToContents(0)

    def remove_section(self, section_id: str) -> None:
        item = self._items.pop(section_id, None)
        if item is None:
            return
        index = self.tree.indexOfTopLevelItem(item)
        if index >= 0:
            self.tree.takeTopLevelItem(index)

    def update_section(self, section_id: str, title: str, index: int) -> None:
        item = self._items.get(section_id)
        if item is None:
            return
        item.setText(0, title)
        item.setText(2, str(index))

    def activate_section(self, section_id: str) -> None:
        item = self._items.get(section_id)
        if item is not None:
            self.tree.setCurrentItem(item)

    def _activate_item(self, item: QTreeWidgetItem) -> None:
        section_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        self.section_activated.emit(section_id)

    def _close_current(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        section_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        self.section_close_requested.emit(section_id)
