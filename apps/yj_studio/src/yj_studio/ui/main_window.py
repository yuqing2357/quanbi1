from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QUndoStack
from PyQt6.QtWidgets import QDialog, QFileDialog, QLabel, QMainWindow, QMessageBox

from yj_studio import __version__
from yj_studio.config.defaults import DEFAULT_Z_WINDOW_START
from yj_studio.config.paths import (
    DEFAULT_LITH_POR_MODEL_ROOT,
    DEFAULT_SEISMIC_NPY,
    existing_processed_root,
)
from yj_studio.config.styles import LITH_BODY_STYLE, PALETTE
from yj_studio.data import VolumeStore, WellRepository, estimate_volume_clim
from yj_studio.data.arbitrary_section import sample_arbitrary_section
from yj_studio.io.readers.fault_mesh import discover_fault_mesh_summaries
from yj_studio.io.readers.layers_npz import discover_layer_summaries
from yj_studio.io.readers.lith_body import discover_lith_body_mesh_summaries
from yj_studio.io.readers.volume_npy import VolumeSpec, load_available_volume_specs
from yj_studio.io.readers.well_logs import load_log_samples
from yj_studio.scene import LayerStore
from yj_studio.scene.layers import (
    ArbitrarySectionLayer,
    FaultSurfaceLayer,
    HorizonLayer,
    LithBodyLayer,
    VolumeLayer,
    WellLayer,
    WellLogLayer,
)
from yj_studio.scene.manual_geometry import is_manual_geometry_layer, manual_geometry_points
from yj_studio.services import (
    SectionAxis,
    ViewSyncService,
    build_structure_map,
    build_well_section_data,
    find_horizon_high_point,
    sample_volume_along_horizon,
)
from yj_studio.tools import ToolManager, build_default_tools
from yj_studio.ui.dialogs.arbitrary_section_dialog import ArbitrarySectionDialog, WellMapPoint
from yj_studio.ui.docks.fault_dock import FaultDock
from yj_studio.ui.docks.horizon_dock import HorizonDock
from yj_studio.ui.docks.layer_tree_dock import LayerTreeDock
from yj_studio.ai import AIService, SAM3Config
from yj_studio.algorithms import AlgorithmRunner, registry as algorithm_registry
from yj_studio.algorithms import builtin as _algorithm_builtin  # noqa: F401 — registers algorithms
from yj_studio.ui.docks.ai_dock import AIDock
from yj_studio.ui.docks.algorithm_dock import AlgorithmDock
from yj_studio.ui.docks.measurement_dock import MeasurementDock
from yj_studio.ui.docks.property_dock import PropertyDock
from yj_studio.ui.docks.section_navigator_dock import SectionNavigatorDock
from yj_studio.ui.docks.slice_controls_dock import SliceControlsDock
from yj_studio.ui.docks.tool_palette_dock import ToolPaletteDock
from yj_studio.ui.docks.well_section_dock import WellSectionDock
from yj_studio.ui.docks.wells_dock import WellsDock
from yj_studio.view.views_area import ViewsArea
from yj_studio.view.view_horizon_map import ViewHorizonMap
from yj_studio.view.view_well_section import ViewWellSection

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window for the YJ Studio desktop app."""

    def __init__(self, *, auto_load: bool = True, enable_3d: bool | None = None) -> None:
        super().__init__()
        self.layer_store = LayerStore()
        self.volume_store = VolumeStore()
        self.view_sync = ViewSyncService()
        self.tool_manager = ToolManager()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(100)
        self.algorithm_runner = AlgorithmRunner(layer_store=self.layer_store, parent=self)
        self.ai_service = AIService(SAM3Config(), parent=self)
        # In-process algorithms (e.g. SAM3) reach the service via
        # ``ctx.services['ai_service']``; volume_store gives them slice access.
        self.algorithm_runner.register_service("ai_service", self.ai_service)
        self.algorithm_runner.register_service("volume_store", self.volume_store)
        # Interactive AI prompt tools talk to the same service.
        self.tool_manager.register_service("ai_service", self.ai_service)
        self.volume_specs: dict[str, VolumeSpec] = {}
        self._active_volume_layer_id: str | None = None
        self._loaded_horizon_paths: set[str] = set()
        self._loaded_fault_paths: set[str] = set()
        self._loaded_lith_body_paths: set[str] = set()
        self._loaded_well_names: set[str] = set()
        self._loaded_well_log_keys: set[tuple[str, str, str]] = set()
        self._scene_controller = None
        self._slice_controls: SliceControlsDock | None = None
        self._layer_tree: LayerTreeDock | None = None
        self._tool_palette: ToolPaletteDock | None = None
        self._section_navigator: SectionNavigatorDock | None = None
        self._wells_dock: WellsDock | None = None
        self._well_section_dock: WellSectionDock | None = None
        self._measurement_dock: MeasurementDock | None = None
        self._property_dock: PropertyDock | None = None
        self._algorithm_dock: AlgorithmDock | None = None
        self._ai_dock: AIDock | None = None
        self._views_area: ViewsArea | None = None
        self._view_3d = None
        self._well_display_mode = "none"
        if enable_3d is None:
            enable_3d = os.environ.get("YJ_STUDIO_DISABLE_3D") != "1"

        self.setWindowTitle(f"YJ Studio v{__version__}")
        self.resize(1280, 800)
        self._register_tools()
        self._build_central_area(enable_3d=enable_3d)
        self._build_menus()
        self._build_layer_tree()
        self._build_tool_palette()
        self._build_manager_docks()
        self._build_property_dock()
        if enable_3d:
            self._build_slice_controls()
        self._build_section_navigator()
        self._build_measurement_dock()
        self._build_well_section_dock()
        self._build_algorithm_dock()
        self._build_ai_dock()
        self._constrain_dock_sizes()
        # Tab bars for tabified docks are created lazily by Qt, so deferring
        # configuration until the event loop turns once is the most reliable
        # way to actually catch them all (a synchronous call here misses the
        # left-side tab group on the first show).
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, self._configure_dock_tabs)
        # Make sure the right-side tab group surfaces a sensible default tab
        # (otherwise Qt picks whichever dock was added last, which would
        # leave the user staring at the AI dock at every launch).
        if self._property_dock is not None:
            self._property_dock.raise_()
        if enable_3d:
            self._discover_default_volumes()
            if auto_load:
                self.load_default_volume()
                self.load_default_horizons()
                self.load_default_faults()
                self.load_default_lith_bodies()
                self.load_default_wells()
        self.statusBar().showMessage(self.tr("Ready"))

    def _build_central_area(self, *, enable_3d: bool) -> None:
        self._views_area = ViewsArea(
            self.layer_store,
            self.volume_store,
            self.view_sync,
            tool_manager=self.tool_manager,
            parent=self,
        )
        self.setCentralWidget(self._views_area)
        if enable_3d:
            from yj_studio.view.scene_controller import SceneController
            from yj_studio.view.view_3d import View3D

            self._view_3d = View3D(self)
            self._view_3d.layer_store = self.layer_store
            self._view_3d.volume_store = self.volume_store
            self._view_3d.view_sync = self.view_sync
            self._views_area.add_primary_view(self._view_3d, self.tr("3D"))
            self._scene_controller = SceneController(self.layer_store, self.volume_store, self._view_3d)
            return

        label = QLabel(self.tr("YJ Studio"))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("emptyWorkspaceLabel")
        self._views_area.add_primary_view(label, self.tr("Workspace"))

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu(self.tr("File"))
        open_volume_action = file_menu.addAction(self.tr("Open Volume..."))
        open_volume_action.triggered.connect(self._open_volume_dialog)
        file_menu.addSeparator()
        file_menu.addAction(self.tr("Exit"), self.close)

        edit_menu = self.menuBar().addMenu(self.tr("Edit"))
        undo_action = self.undo_stack.createUndoAction(self, self.tr("Undo"))
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        redo_action = self.undo_stack.createRedoAction(self, self.tr("Redo"))
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)

        view_menu = self.menuBar().addMenu(self.tr("View"))
        inline_action = view_menu.addAction(self.tr("New Inline Section"))
        inline_action.triggered.connect(lambda: self._open_section("inline"))
        xline_action = view_menu.addAction(self.tr("New Xline Section"))
        xline_action.triggered.connect(lambda: self._open_section("xline"))
        z_action = view_menu.addAction(self.tr("New Z Section"))
        z_action.triggered.connect(lambda: self._open_section("z"))
        arbitrary_action = view_menu.addAction(self.tr("New Arbitrary Section..."))
        arbitrary_action.triggered.connect(self._open_arbitrary_section_dialog)

        help_menu = self.menuBar().addMenu(self.tr("Help"))
        help_menu.addAction(self.tr("About"), self._show_about)

    def _build_layer_tree(self) -> None:
        dock = LayerTreeDock(self.layer_store, self, undo_stack=self.undo_stack)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._layer_tree = dock

    def _build_property_dock(self) -> None:
        dock = PropertyDock(
            self.layer_store,
            self.volume_store,
            self.undo_stack,
            self,
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # PropertyDock is the anchor of the right-side tab group; every other
        # right-side dock built later tabifies onto it so we end up with a
        # single horizontal row of tabs instead of multiple stacked groups.
        self._property_dock = dock

    def _build_tool_palette(self) -> None:
        dock = ToolPaletteDock(self.tool_manager, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        if self._layer_tree is not None:
            self.tabifyDockWidget(self._layer_tree, dock)
        self._tool_palette = dock

    def _build_manager_docks(self) -> None:
        horizon_dock = HorizonDock(self.layer_store, self)
        fault_dock = FaultDock(self.layer_store, self)
        wells_dock = WellsDock(self.layer_store, self)
        horizon_dock.structure_map_requested.connect(self._open_horizon_structure_map)
        horizon_dock.high_point_requested.connect(self._jump_to_horizon_high_point)
        horizon_dock.along_horizon_requested.connect(self._open_along_horizon_map)
        wells_dock.display_mode_changed.connect(
            lambda mode: self._set_well_display_mode(mode, sync_dock=False)
        )
        for dock in (horizon_dock, fault_dock, wells_dock):
            dock.layer_activated.connect(self._activate_layer_from_dock)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
            if self._layer_tree is not None:
                self.tabifyDockWidget(self._layer_tree, dock)
        if self._layer_tree is not None:
            self._layer_tree.raise_()
        self._wells_dock = wells_dock

    def _build_section_navigator(self) -> None:
        dock = SectionNavigatorDock(self)
        if self._views_area is not None:
            self._views_area.section_added.connect(dock.add_section)
            self._views_area.section_removed.connect(dock.remove_section)
            self._views_area.section_updated.connect(dock.update_section)
            self._views_area.current_section_changed.connect(dock.activate_section)
            dock.section_activated.connect(self._views_area.activate_section)
            dock.section_close_requested.connect(self._views_area.close_section)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_right(dock)
        self._section_navigator = dock

    def _build_measurement_dock(self) -> None:
        dock = MeasurementDock(self.layer_store, self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_right(dock)
        self._measurement_dock = dock

    def _build_well_section_dock(self) -> None:
        dock = WellSectionDock(self.layer_store, self)
        dock.build_requested.connect(self._open_connected_well_section)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_right(dock)
        self._well_section_dock = dock

    def _build_algorithm_dock(self) -> None:
        dock = AlgorithmDock(
            self.layer_store,
            algorithm_registry,
            self.algorithm_runner,
            self,
            undo_stack=self.undo_stack,
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_right(dock)
        self._algorithm_dock = dock

    def _build_ai_dock(self) -> None:
        dock = AIDock(
            self.layer_store,
            self.ai_service,
            self.algorithm_runner,
            self.tool_manager,
            undo_stack=self.undo_stack,
            parent=self,
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_right(dock)
        self._ai_dock = dock

    def _tabify_right(self, dock) -> None:
        """Attach ``dock`` to the right-side tab group anchored on
        ``_property_dock`` so the right column stays a single row of tabs
        instead of splitting into multiple stacked sub-areas.
        """

        if self._property_dock is None or dock is self._property_dock:
            return
        self.tabifyDockWidget(self._property_dock, dock)

    def _configure_dock_tabs(self) -> None:
        """Stop Qt from eliding dock tab labels (``"P..." "Sli..."``).

        Qt's auto-generated QTabBar for tabified docks defaults to
        ``ElideRight`` so it can squeeze many tabs into a tiny strip. Our
        dock names are short and meaningful, so we prefer scrolling +
        readable labels over heavy truncation. This also forces a slightly
        denser font so a few extra tabs fit before scroll arrows appear.
        """

        from PyQt6.QtCore import Qt as _Qt
        from PyQt6.QtWidgets import QTabBar

        for tab_bar in self.findChildren(QTabBar):
            tab_bar.setElideMode(_Qt.TextElideMode.ElideNone)
            tab_bar.setUsesScrollButtons(True)
            tab_bar.setExpanding(False)
            # Keep the font slightly smaller than default to fit more tabs
            # without needing the scroll arrows on most screens.
            font = tab_bar.font()
            font.setPointSizeF(max(8.5, font.pointSizeF() - 0.5))
            tab_bar.setFont(font)

    def _constrain_dock_sizes(self) -> None:
        """Give each dock a sensible default width and a small minimum so
        the central viewport keeps a sensible amount of room and the main
        window's overall minimum size doesn't balloon to the sum of every
        dock's sizeHint.

        We use ``resizeDocks`` for the initial column width (one shot) and
        only set a *minimum* on the inner widget — the user can still drag
        the splitter wider afterwards if they want a roomier form.

        Without this, ``AlgorithmDock`` / ``AIDock`` (which contain wide
        forms + list widgets) force the main window to require ~1300 px
        just for the right column, so a 1920x1080 monitor ends up with a
        tiny central area and the user perceives the layout as 'broken' on
        resize.
        """

        from PyQt6.QtCore import Qt as _Qt

        left_docks = [d for d in (self._layer_tree, self._tool_palette) if d is not None]
        right_docks = [
            d
            for d in (
                self._slice_controls,
                self._property_dock,
                self._section_navigator,
                self._measurement_dock,
                self._well_section_dock,
                self._algorithm_dock,
                self._ai_dock,
            )
            if d is not None
        ]

        # Allow the user to shrink each dock down to ~200 px without
        # accidentally crushing controls; the dock itself can still grow.
        for dock in left_docks + right_docks:
            inner = dock.widget()
            if inner is not None:
                inner.setMinimumWidth(200)

        # Suggest an initial column width via resizeDocks (Qt 5.6+). The
        # right column needs more room because algorithm / AI forms have
        # labels + spin boxes side-by-side; the left column only hosts a
        # tool palette / layer tree.
        if left_docks:
            self.resizeDocks(left_docks, [240] * len(left_docks), _Qt.Orientation.Horizontal)
        if right_docks:
            self.resizeDocks(right_docks, [360] * len(right_docks), _Qt.Orientation.Horizontal)

    def _register_tools(self) -> None:
        for tool in build_default_tools():
            self.tool_manager.register(tool)
        self.tool_manager.active_tool_changed.connect(self._on_active_tool_changed)
        self.tool_manager.message_requested.connect(self.statusBar().showMessage)
        active = self.tool_manager.active_tool
        if active is not None:
            self._on_active_tool_changed(active.id)

    def _on_active_tool_changed(self, tool_id: str) -> None:
        tool = self.tool_manager.get(tool_id)
        self.statusBar().showMessage(self.tr("Tool: {label}").format(label=tool.label))

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            self.tr("About YJ Studio"),
            self.tr("YJ Studio v{version}\nSeismic interpretation desktop system.").format(
                version=__version__
            ),
        )

    def _build_slice_controls(self) -> None:
        dock = SliceControlsDock(self)
        dock.volume_changed.connect(self.load_volume)
        dock.slice_changed.connect(self._set_slice_index)
        dock.clim_changed.connect(self._set_clim)
        dock.cmap_changed.connect(self._set_cmap)
        dock.roi_changed.connect(self._set_roi)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_right(dock)
        self._slice_controls = dock

    def _discover_default_volumes(self) -> None:
        processed_root = existing_processed_root()
        specs, notes = load_available_volume_specs(
            DEFAULT_SEISMIC_NPY,
            processed_root / "地震属性",
            DEFAULT_LITH_POR_MODEL_ROOT,
        )
        self.volume_specs = dict(specs)
        for spec in specs.values():
            self.volume_store.register(spec)
        if self._slice_controls is not None:
            self._slice_controls.set_volume_specs(self.volume_specs)
        for note in notes:
            logger.info("Volume discovery: %s", note)

    def load_default_horizons(self) -> None:
        horizon_dir = existing_processed_root() / "层位"
        if not horizon_dir.exists():
            logger.info("Horizon directory not found: %s", horizon_dir)
            return
        try:
            summaries = discover_layer_summaries(horizon_dir)
        except FileNotFoundError:
            return
        for summary in summaries:
            path_text = str(summary.path)
            if path_text in self._loaded_horizon_paths:
                continue
            layer = HorizonLayer(
                name=summary.name,
                data_path=path_text,
                color=(0.302, 0.686, 0.290, 0.88),
                opacity=0.88,
                visible=False,
                metadata={**summary.metadata, "path": path_text, "lazy": True},
                provenance={"source": "processed.horizon"},
            )
            self.layer_store.add(layer)
            self._loaded_horizon_paths.add(path_text)
        if summaries:
            self.statusBar().showMessage(
                self.tr("Loaded {count} horizon entries").format(count=len(summaries))
            )

    def load_default_faults(self) -> None:
        fault_dir = existing_processed_root() / "断层"
        if not fault_dir.exists():
            logger.info("Fault directory not found: %s", fault_dir)
            return
        try:
            summaries = discover_fault_mesh_summaries(fault_dir)
        except FileNotFoundError:
            return
        for idx, summary in enumerate(summaries):
            path_text = str(summary.path)
            if path_text in self._loaded_fault_paths:
                continue
            color = PALETTE[idx % len(PALETTE)]
            layer = FaultSurfaceLayer(
                name=summary.name,
                data_path=path_text,
                color=(color[0], color[1], color[2], 0.35),
                opacity=0.35,
                visible=False,
                metadata={**summary.metadata, "path": path_text, "lazy": True},
                provenance={"source": "processed.fault_mesh"},
            )
            self.layer_store.add(layer)
            self._loaded_fault_paths.add(path_text)
        if summaries:
            self.statusBar().showMessage(
                self.tr("Loaded {count} fault entries").format(count=len(summaries))
            )

    def load_default_lith_bodies(self) -> None:
        if not DEFAULT_LITH_POR_MODEL_ROOT.exists():
            logger.info("Lithology model directory not found: %s", DEFAULT_LITH_POR_MODEL_ROOT)
            return
        try:
            summaries = discover_lith_body_mesh_summaries(DEFAULT_LITH_POR_MODEL_ROOT)
        except FileNotFoundError:
            return
        for summary in summaries:
            path_text = str(summary.path)
            if path_text in self._loaded_lith_body_paths:
                continue
            rgba = _lith_body_rgba(summary.class_value, summary.metadata)
            layer = LithBodyLayer(
                name=f"{summary.class_value}_{summary.class_name}",
                class_value=summary.class_value,
                class_name=summary.class_name,
                data_path=path_text,
                color=rgba,
                opacity=rgba[3],
                visible=False,
                metadata={**summary.metadata, "path": path_text, "lazy": True},
                provenance={"source": "model.lithology_body_mesh"},
            )
            self.layer_store.add(layer)
            self._loaded_lith_body_paths.add(path_text)
        if summaries:
            self.statusBar().showMessage(
                self.tr("Loaded {count} lithology body entries").format(count=len(summaries))
            )

    def load_default_wells(self) -> None:
        volume_layer = self._active_volume_layer()
        z_count = volume_layer.shape[2] if volume_layer is not None and volume_layer.shape else 654
        coords_csv = (
            existing_processed_root()
            / "测井坐标"
            / "combined_well_coordinates_inside_new_seismic_depth_0_654.csv"
        )
        if not coords_csv.exists():
            logger.info("Well coordinates file not found: %s", coords_csv)
            return
        log_roots = _default_well_log_roots()
        repository = WellRepository.from_coordinates_csv(
            coords_csv,
            z_count=z_count,
            log_roots=log_roots,
        )
        log_specs = _default_well_log_specs()
        for idx, record in enumerate(repository.iter_records()):
            if record.name in self._loaded_well_names:
                continue
            color = PALETTE[idx % len(PALETTE)]
            layer = WellLayer(
                name=record.name,
                well_name=record.name,
                trajectory=record.trajectory,
                head_position=record.head_position,
                color=(color[0], color[1], color[2], 1.0),
                opacity=1.0,
                visible=False,
                metadata={
                    **record.metadata,
                    "matched_csv_name": record.matched_csv_name or "",
                    "path": str(coords_csv),
                    "depth_mapping": "sample_index = DEPT / 10.0 for F:/YJ-ALL-SEISMIC_depth_0_653.npy",
                },
                provenance={"source": "processed.well_coordinates"},
            )
            self.layer_store.add(layer)
            self._loaded_well_names.add(record.name)
            self._add_well_log_layers(record, z_count=z_count, log_specs=log_specs)
        if len(repository):
            self.statusBar().showMessage(
                self.tr("Loaded {count} well entries").format(count=len(repository))
            )

    def _add_well_log_layers(
        self,
        record,
        *,
        z_count: int,
        log_specs: list[dict[str, object]],
    ) -> None:
        if not record.matched_csv_name:
            return
        inline_index = float(record.head_position[0])
        xline_index = float(record.head_position[1])
        for spec in log_specs:
            root = spec["root"]
            if not isinstance(root, Path):
                continue
            log_path = root / f"{record.matched_csv_name}.csv"
            key = (record.name, str(spec["key"]), str(log_path))
            if key in self._loaded_well_log_keys or not log_path.exists():
                continue
            samples = load_log_samples(
                log_path,
                inline_index=inline_index,
                xline_index=xline_index,
                value_column=str(spec["value_column"]),
                z_count=z_count,
                z_window_start=DEFAULT_Z_WINDOW_START,
            )
            if samples.samples.size == 0:
                continue
            layer = WellLogLayer(
                name=f"{record.name} {spec['label']}",
                well_name=record.name,
                mode=spec["mode"],
                samples=samples.samples,
                value_column=samples.value_column,
                color=(1.0, 1.0, 1.0, 0.85),
                opacity=0.85,
                visible=False,
                metadata={
                    "path": str(log_path),
                    "source_path": str(log_path),
                    "cmap": str(spec["cmap"]),
                    "clim": spec["clim"],
                    "sample_count": int(samples.samples.shape[0]),
                    "depth_mapping": "sample_index = DEPT / 10.0 for F:/YJ-ALL-SEISMIC_depth_0_653.npy",
                },
                provenance={"source": "processed.well_log"},
            )
            self.layer_store.add(layer)
            self._loaded_well_log_keys.add(key)

    def load_default_volume(self) -> None:
        if not self.volume_specs:
            self.statusBar().showMessage(self.tr("No default volume found"))
            return
        volume_id = "seismic" if "seismic" in self.volume_specs else next(iter(self.volume_specs))
        self.load_volume(volume_id)

    def load_volume(self, volume_id: str) -> None:
        try:
            volume = self.volume_store.get_volume(volume_id)
            shape = tuple(int(value) for value in volume.shape)
        except Exception as exc:
            logger.exception("Failed to load volume %s", volume_id)
            QMessageBox.warning(self, self.tr("Open Volume"), str(exc))
            return

        spec = self.volume_store.spec(volume_id)
        slice_indices = _default_slice_indices(shape)
        clim = tuple(estimate_volume_clim(volume_id, volume, *slice_indices.values()))
        layer = self._active_volume_layer()
        if layer is None:
            layer = VolumeLayer(
                name=spec.label,
                volume_id=volume_id,
                shape=shape,
                clim=clim,
                cmap=spec.cmap,
                slice_indices=slice_indices,
                metadata={"path": str(spec.path)},
            )
            self._active_volume_layer_id = self.layer_store.add(layer)
        else:
            layer.name = spec.label
            layer.volume_id = volume_id
            layer.shape = shape
            layer.clim = clim
            layer.cmap = spec.cmap
            layer.slice_indices = slice_indices
            layer.metadata["path"] = str(spec.path)
            self.layer_store.layer_changed.emit(layer.id, "data")

        if self._slice_controls is not None:
            self._slice_controls.set_current_volume(volume_id)
            self._slice_controls.set_shape(shape, slice_indices)
            self._slice_controls.set_clim(clim)
            self._slice_controls.set_cmap(spec.cmap)
        if self._view_3d is not None:
            self._view_3d.reset_to_volume(shape)
        self.statusBar().showMessage(
            self.tr("Loaded {label}: {shape}").format(label=spec.label, shape=shape)
        )

    def _set_slice_index(self, axis: str, index: int) -> None:
        self._set_active_slice_index(axis, index, origin=self._slice_controls)

    def _set_active_slice_index(self, axis: str, index: int, *, origin: object | None = None) -> int | None:
        layer = self._active_volume_layer()
        if layer is None:
            return None
        if layer.shape is None:
            return None
        limit = {"inline": layer.shape[0], "xline": layer.shape[1], "z": layer.shape[2]}[axis]
        clipped = int(np.clip(int(index), 0, limit - 1))
        layer.slice_indices[axis] = clipped
        self.layer_store.layer_changed.emit(layer.id, "slice_indices")
        if self._slice_controls is not None:
            self._slice_controls.set_slice_index(axis, clipped)
        self.view_sync.publish(f"slice.{axis}_position", clipped, origin)
        return clipped

    def _set_clim(self, clim: object) -> None:
        layer = self._active_volume_layer()
        if layer is None:
            return
        vmin, vmax = clim
        layer.clim = (float(vmin), float(vmax))
        self.layer_store.layer_changed.emit(layer.id, "clim")

    def _set_cmap(self, cmap: str) -> None:
        layer = self._active_volume_layer()
        if layer is None:
            return
        layer.cmap = cmap
        self.layer_store.layer_changed.emit(layer.id, "cmap")

    def _set_roi(self, roi: object) -> None:
        from yj_studio.scene.undo_commands import SetLayerFieldCommand

        layer = self._active_volume_layer()
        if layer is None:
            return
        new_value: tuple[int, int, int, int, int, int] | None
        if roi is None:
            new_value = None
        else:
            new_value = tuple(int(v) for v in roi)  # type: ignore[assignment]
        if layer.roi == new_value:
            return
        self.undo_stack.push(
            SetLayerFieldCommand(
                self.layer_store, layer.id, "roi", new_value, text="Change ROI"
            )
        )

    def _open_section(self, axis: SectionAxis) -> None:
        layer = self._active_volume_layer()
        if layer is None or self._views_area is None:
            self.statusBar().showMessage(self.tr("Load a volume before opening a section"))
            return
        if layer.shape is None:
            return
        default_index = {"inline": layer.shape[0] // 2, "xline": layer.shape[1] // 2, "z": layer.shape[2] // 2}[axis]
        index = int(layer.slice_indices.get(axis, default_index))
        self._open_section_at(axis, index)

    def _open_section_at(self, axis: SectionAxis, index: int) -> None:
        layer = self._active_volume_layer()
        if layer is None or self._views_area is None:
            self.statusBar().showMessage(self.tr("Load a volume before opening a section"))
            return
        self._views_area.add_orthogonal_section(
            volume_layer_id=layer.id,
            axis=axis,
            index=index,
        )

    def _open_arbitrary_section_dialog(self) -> None:
        layer = self._active_volume_layer()
        if layer is None or layer.shape is None or self._views_area is None:
            self.statusBar().showMessage(self.tr("Load a volume before opening an arbitrary section"))
            return
        dialog = ArbitrarySectionDialog(
            shape=layer.shape,
            topdown_image=self._topdown_slice(layer),
            topdown_cmap=layer.cmap,
            well_points=self._well_map_points(),
            initial_polyline=self._selected_polyline_xy(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._open_arbitrary_section(
            dialog.polyline(),
            z_start=dialog.z_start.value(),
            z_end=dialog.z_end.value(),
            max_trace_count=dialog.max_traces.value(),
        )

    def _open_arbitrary_section(
        self,
        polyline_xy: np.ndarray,
        *,
        z_start: int,
        z_end: int,
        max_trace_count: int,
    ) -> None:
        volume_layer = self._active_volume_layer()
        if volume_layer is None or volume_layer.shape is None or self._views_area is None:
            self.statusBar().showMessage(self.tr("Load a volume before opening an arbitrary section"))
            return
        try:
            volume = self.volume_store.get_volume(volume_layer.volume_id)
            section = sample_arbitrary_section(
                volume,
                polyline_xy,
                z_start=z_start,
                z_end=z_end,
                max_trace_count=max_trace_count,
            )
        except Exception as exc:
            logger.exception("Failed to create arbitrary section")
            QMessageBox.warning(self, self.tr("Arbitrary Section"), str(exc))
            return
        z_mid = float(section.depths[0] + section.depths[-1]) / 2.0
        polyline_xyz = np.column_stack(
            [section.polyline_xy, np.full(section.polyline_xy.shape[0], z_mid, dtype=np.float32)]
        ).astype(np.float32)
        section_layer = ArbitrarySectionLayer(
            name=_next_layer_name(self.layer_store, "Arbitrary Section"),
            polyline=polyline_xyz,
            image=section.values,
            distances=section.distances,
            depths=section.depths,
            axis_label="Distance",
            color=(1.0, 0.55, 0.1, 0.9),
            opacity=0.9,
            visible=True,
            metadata={
                "volume_id": volume_layer.volume_id,
                "trace_count": int(section.values.shape[1]),
                "z_range": f"{int(section.depths[0])}..{int(section.depths[-1])}",
                "cmap": volume_layer.cmap,
            },
            provenance={"source": "manual.arbitrary_section"},
        )
        layer_id = self.layer_store.add(section_layer)
        try:
            self._views_area.add_arbitrary_section(layer_id)
        except Exception as exc:
            self.layer_store.remove(layer_id)
            logger.exception("Failed to open arbitrary section view")
            QMessageBox.warning(self, self.tr("Arbitrary Section"), str(exc))
            return
        self.layer_store.select([layer_id])
        self.statusBar().showMessage(
            self.tr("Opened arbitrary section: {traces} traces").format(
                traces=section.values.shape[1]
            )
        )

    def _selected_polyline_xy(self) -> np.ndarray | None:
        for layer_id in self.layer_store.selection:
            try:
                layer = self.layer_store.get(layer_id)
            except KeyError:
                continue
            if not is_manual_geometry_layer(layer):
                continue
            points = manual_geometry_points(layer)
            if points is not None and points.shape[0] >= 2:
                return points[:, :2].astype(np.float32)
        return None

    def _topdown_slice(self, layer: VolumeLayer) -> np.ndarray | None:
        if layer.shape is None:
            return None
        z_index = int(layer.slice_indices.get("z", layer.shape[2] // 2))
        z_index = int(np.clip(z_index, 0, layer.shape[2] - 1))
        try:
            return self.volume_store.get_slice(layer.volume_id, "z", z_index)
        except Exception:
            logger.exception("Failed to load top-down slice for arbitrary section")
            return None

    def _well_map_points(self) -> tuple[WellMapPoint, ...]:
        points: list[WellMapPoint] = []
        for layer in self.layer_store.iter_by_type(WellLayer):
            if layer.head_position is None:
                continue
            points.append(
                WellMapPoint(
                    name=layer.well_name or layer.name,
                    inline=float(layer.head_position[0]),
                    xline=float(layer.head_position[1]),
                )
            )
        return tuple(points)

    def _open_volume_dialog(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Volume"),
            str(DEFAULT_SEISMIC_NPY.parent),
            self.tr("NumPy volume (*.npy)"),
        )
        if not path_text:
            return
        path = Path(path_text)
        volume_id = path.stem
        spec = VolumeSpec(
            key=volume_id,
            path=path,
            label=path.stem,
            cmap="gray",
            filename=path.name,
        )
        self.volume_specs[volume_id] = spec
        self.volume_store.register(spec)
        if self._slice_controls is not None:
            self._slice_controls.set_volume_specs(self.volume_specs)
        self.load_volume(volume_id)

    def _open_horizon_structure_map(self, layer_id: str) -> None:
        if self._views_area is None:
            return
        layer = self.layer_store.get(layer_id)
        if not isinstance(layer, HorizonLayer):
            return
        try:
            data = build_structure_map(layer)
        except Exception as exc:
            logger.exception("Failed to open horizon structure map")
            QMessageBox.warning(self, self.tr("Horizon"), str(exc))
            return
        view = ViewHorizonMap(data, axis="horizon", parent=self._views_area)
        self._views_area.add_internal_section(
            view,
            title=view.title,
            axis="horizon",
            index=0,
        )
        self.statusBar().showMessage(
            self.tr("Opened structure map: {name}").format(name=layer.name)
        )

    def _open_along_horizon_map(self, layer_id: str) -> None:
        volume_layer = self._active_volume_layer()
        if volume_layer is None or self._views_area is None:
            self.statusBar().showMessage(self.tr("Load a volume before sampling along a horizon"))
            return
        layer = self.layer_store.get(layer_id)
        if not isinstance(layer, HorizonLayer):
            return
        try:
            data = sample_volume_along_horizon(self.volume_store, volume_layer, layer)
        except Exception as exc:
            logger.exception("Failed to sample along horizon")
            QMessageBox.warning(self, self.tr("Horizon"), str(exc))
            return
        view = ViewHorizonMap(data, axis="horizon", parent=self._views_area)
        self._views_area.add_internal_section(
            view,
            title=view.title,
            axis="horizon",
            index=0,
        )
        self.statusBar().showMessage(
            self.tr("Opened along-horizon map: {name}").format(name=layer.name)
        )

    def _jump_to_horizon_high_point(self, layer_id: str) -> None:
        layer = self.layer_store.get(layer_id)
        if not isinstance(layer, HorizonLayer):
            return
        try:
            point = find_horizon_high_point(layer)
        except Exception as exc:
            logger.exception("Failed to find horizon high point")
            QMessageBox.warning(self, self.tr("Horizon"), str(exc))
            return
        self.layer_store.select([layer.id])
        if not layer.visible:
            self.layer_store.update(layer.id, visible=True)
        self._set_active_slice_index("inline", point.inline, origin=self)
        self._set_active_slice_index("xline", point.xline, origin=self)
        self._set_active_slice_index("z", int(round(point.sample)), origin=self)
        self._focus_3d_on_point((float(point.inline), float(point.xline), float(point.sample)))
        self._open_section_at("inline", point.inline)
        self._open_section_at("xline", point.xline)
        self.statusBar().showMessage(
            self.tr("Jumped to high point: {name} inline={inline}, xline={xline}, sample={sample:.1f}").format(
                name=layer.name,
                inline=point.inline,
                xline=point.xline,
                sample=point.sample,
            )
        )

    def _activate_layer_from_dock(self, layer_id: str) -> None:
        self.layer_store.select([layer_id])
        layer = self.layer_store.get(layer_id)
        if isinstance(layer, WellLayer):
            self._open_well_adjacent_section(layer)
            return
        if isinstance(layer, WellLogLayer):
            well_layer = self._well_layer_for_name(layer.well_name)
            if well_layer is not None:
                self._open_well_adjacent_section(well_layer)
                return
        head_position = getattr(layer, "head_position", None)
        if head_position is not None:
            self._focus_3d_on_point(head_position)
        self.statusBar().showMessage(self.tr("Selected {name}").format(name=layer.name))

    def _open_well_adjacent_section(self, layer: WellLayer) -> None:
        if layer.head_position is None:
            return
        self._show_well_layers(layer.well_name or layer.name)
        self._focus_3d_on_point(layer.head_position)
        inline_index = int(round(float(layer.head_position[0])))
        xline_index = int(round(float(layer.head_position[1])))
        opened_inline = self._set_active_slice_index("inline", inline_index, origin=self)
        opened_xline = self._set_active_slice_index("xline", xline_index, origin=self)
        self._open_section_at("inline", opened_inline if opened_inline is not None else inline_index)
        self._open_section_at("xline", opened_xline if opened_xline is not None else xline_index)
        self.statusBar().showMessage(
            self.tr("Opened inline and xline sections for {name}").format(
                name=layer.well_name or layer.name
            )
        )

    def _set_well_display_mode(self, mode: str, *, sync_dock: bool = True) -> None:
        if mode not in {"none", "lith", "por", "perm"}:
            mode = "none"
        self._well_display_mode = mode
        if sync_dock and self._wells_dock is not None:
            self._wells_dock.set_display_mode(mode)
        for well_name in self._visible_or_selected_well_names():
            self._show_well_layers(well_name, mode=mode)
        self.statusBar().showMessage(
            self.tr("Well display mode: {mode}").format(mode=_well_display_label(mode))
        )

    def _visible_or_selected_well_names(self) -> list[str]:
        selected_ids = set(self.layer_store.selection)
        names: list[str] = []
        seen: set[str] = set()
        for layer in self.layer_store.iter_by_type(WellLayer):
            name = layer.well_name or layer.name
            if name in seen:
                continue
            if layer.visible or layer.id in selected_ids:
                names.append(name)
                seen.add(name)
        return names

    def _show_well_layers(self, well_name: str, *, mode: str | None = None) -> None:
        display_mode = self._well_display_mode if mode is None else mode
        for layer in list(self.layer_store.iter_layers()):
            if isinstance(layer, WellLayer) and (layer.well_name or layer.name) == well_name:
                if not layer.visible:
                    self.layer_store.update(layer.id, visible=True)
            elif isinstance(layer, WellLogLayer) and layer.well_name == well_name:
                visible = display_mode != "none" and layer.mode == display_mode
                if layer.visible != visible:
                    self.layer_store.update(layer.id, visible=visible)

    def _well_layer_for_name(self, well_name: str) -> WellLayer | None:
        for layer in self.layer_store.iter_by_type(WellLayer):
            if (layer.well_name or layer.name) == well_name:
                return layer
        return None

    def _open_connected_well_section(self, selected_wells: list[str], mode: str) -> None:
        if len(selected_wells) < 2:
            QMessageBox.information(
                self,
                self.tr("Well Section"),
                self.tr("Select at least two wells."),
            )
            return
        volume_layer = self._active_volume_layer()
        if volume_layer is None or self._views_area is None:
            self.statusBar().showMessage(self.tr("Load a volume before opening a well section"))
            return
        try:
            self._set_well_display_mode(mode)
            for well_name in selected_wells:
                self._show_well_layers(well_name, mode=mode)
            data = build_well_section_data(
                self.layer_store,
                self.volume_store,
                volume_layer,
                selected_wells,
                mode=mode,
            )
        except Exception as exc:
            logger.exception("Failed to open well section")
            QMessageBox.warning(self, self.tr("Well Section"), str(exc))
            return
        view = ViewWellSection(data, self.layer_store, self._views_area)
        view.layer_store = self.layer_store
        view.volume_store = self.volume_store
        view.view_sync = self.view_sync
        self._views_area.add_internal_section(
            view,
            title=view.title,
            axis="well",
            index=0,
        )
        self.statusBar().showMessage(
            self.tr("Opened well section: {names}").format(names=" -> ".join(data.names))
        )

    def _focus_3d_on_point(self, point: tuple[float, float, float]) -> None:
        if self._view_3d is None:
            return
        camera = self._view_3d.camera
        old_position = np.asarray(camera.position, dtype=np.float64)
        old_focal = np.asarray(camera.focal_point, dtype=np.float64)
        new_focal = np.asarray(point, dtype=np.float64)
        camera.focal_point = tuple(float(v) for v in new_focal)
        camera.position = tuple(float(v) for v in new_focal + old_position - old_focal)
        self._view_3d.render()

    def _active_volume_layer(self) -> VolumeLayer | None:
        if self._active_volume_layer_id is None:
            return None
        layer = self.layer_store.get(self._active_volume_layer_id)
        if isinstance(layer, VolumeLayer):
            return layer
        return None


def _default_slice_indices(shape: tuple[int, int, int]) -> dict[str, int]:
    return {"inline": shape[0] // 2, "xline": shape[1] // 2, "z": shape[2] // 2}


def _next_layer_name(layer_store: LayerStore, prefix: str) -> str:
    count = sum(1 for layer in layer_store.iter_layers() if layer.name.startswith(prefix))
    return f"{prefix} {count + 1}"


def _default_well_log_roots() -> list[Path]:
    processed_root = existing_processed_root()
    return [
        processed_root / "por",
        processed_root / "perm",
        processed_root / "lith" / "coarse",
        processed_root / "lith" / "fine",
        processed_root / "lith" / "raw",
    ]


def _default_well_log_specs() -> list[dict[str, object]]:
    processed_root = existing_processed_root()
    return [
        {
            "key": "por",
            "label": "POR",
            "mode": "por",
            "root": processed_root / "por",
            "value_column": "por",
            "cmap": "viridis",
            "clim": (0.0, 0.35),
        },
        {
            "key": "perm",
            "label": "PERM",
            "mode": "perm",
            "root": processed_root / "perm",
            "value_column": "perm",
            "cmap": "plasma",
            "clim": (0.0, 1000.0),
        },
        {
            "key": "lith_coarse",
            "label": "LITH",
            "mode": "lith",
            "root": processed_root / "lith" / "coarse",
            "value_column": "lith",
            "cmap": "tab10",
            "clim": (-0.5, 5.5),
        },
    ]


def _well_display_label(mode: str) -> str:
    return {
        "none": "Well only",
        "lith": "Lithology",
        "por": "Porosity",
        "perm": "Permeability",
    }.get(mode, "Well only")


def _lith_body_rgba(class_value: int, metadata: dict[str, object]) -> tuple[float, float, float, float]:
    raw_color = metadata.get("color_rgba_uint8")
    if isinstance(raw_color, list | tuple) and len(raw_color) >= 4:
        return tuple(float(v) / 255.0 for v in raw_color[:4])
    style = LITH_BODY_STYLE.get(class_value, {"color": (180, 180, 180)})
    r, g, b = style["color"]
    return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 0.35)
