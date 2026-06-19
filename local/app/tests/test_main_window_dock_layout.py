from __future__ import annotations

from PyQt6.QtWidgets import QDockWidget, QMenu


def _dock_titles(window, group: str) -> list[str]:
    return [dock.windowTitle() for dock in window._dock_groups[group]]


def _panel_menu_titles(window) -> dict[str, list[str]]:
    titles: dict[str, list[str]] = {}
    assert window._panels_menu is not None
    for action in window._panels_menu.actions():
        menu = action.menu()
        if not isinstance(menu, QMenu):
            continue
        titles[menu.title()] = [child.text() for child in menu.actions()]
    return titles


def test_main_window_groups_docks_by_workflow(qapp) -> None:
    from yj_studio.ui.main_window import MainWindow

    window = MainWindow(auto_load=False, enable_3d=False)

    assert _dock_titles(window, "data") == ["图层", "层位", "断层", "井", "属性"]
    assert _dock_titles(window, "annotation") == ["工具", "AI 助手", "目标"]
    assert _dock_titles(window, "result") == ["剖面", "井剖面", "算法"]

    menu_titles = _panel_menu_titles(window)
    assert menu_titles["数据"] == ["图层", "层位", "断层", "井", "属性"]
    assert menu_titles["标注"] == ["工具", "AI 助手", "目标"]
    assert menu_titles["结果"] == ["剖面", "井剖面", "算法"]

    for group, docks in window._dock_groups.items():
        for dock in docks:
            assert dock.property("dock_group") == group

    window.close()


def test_main_window_default_core_docks_are_visible(qapp) -> None:
    from yj_studio.ui.main_window import MainWindow

    window = MainWindow(auto_load=False, enable_3d=False)
    docks = {dock.windowTitle(): dock for dock in window.findChildren(QDockWidget)}

    assert {
        title
        for title, dock in docks.items()
        if bool(dock.property("default_visible"))
    } == {"图层", "属性", "AI 助手", "目标"}
    assert not docks["图层"].isHidden()
    assert not docks["属性"].isHidden()
    assert not docks["AI 助手"].isHidden()
    assert not docks["目标"].isHidden()

    for title in ("工具", "层位", "断层", "井", "剖面", "井剖面", "算法"):
        assert docks[title].isHidden()

    window.close()


def test_slice_controls_are_core_data_dock_when_available(qapp) -> None:
    from yj_studio.ui.main_window import MainWindow

    window = MainWindow(auto_load=False, enable_3d=False)
    window._build_slice_controls()
    window._populate_panel_menu()
    window._apply_default_dock_visibility()

    assert _dock_titles(window, "data") == ["图层", "层位", "断层", "井", "属性", "剖面控制"]
    assert _panel_menu_titles(window)["数据"] == ["图层", "层位", "断层", "井", "属性", "剖面控制"]
    assert window._slice_controls is not None
    assert window._slice_controls.property("dock_group") == "data"
    assert window._slice_controls.property("default_visible")
    assert not window._slice_controls.isHidden()

    window.close()


def test_main_window_view_menu_has_no_removed_entries(qapp) -> None:
    from yj_studio.ui.main_window import MainWindow

    window = MainWindow(auto_load=False, enable_3d=False)
    menu_text = " ".join(
        action.text()
        for menu in window.menuBar().findChildren(QMenu)
        for action in menu.actions()
    )
    dock_titles = {dock.windowTitle() for dock in window.findChildren(QDockWidget)}

    assert "打开储层剖面" not in menu_text
    assert "测量" not in dock_titles

    window.close()
