from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
import useq
from fonticon_mdi6 import MDI6
from ndv import NDViewer
from pydantic import ValidationError
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from pymmcore_widgets.useq_widgets._well_plate_widget import (
    DATA_POSITION,
    WellPlateView,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QIcon
from qtpy.QtWidgets import (
    QAbstractGraphicsShapeItem,
    QGridLayout,
    QGroupBox,
    QMainWindow,
    QMenuBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._widgets._mda_widget._save_widget import (
    OME_ZARR,
    WRITERS,
    ZARR_TESNSORSTORE,
)
from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

from ._analysis import EVOKED, _AnalyseCalciumTraces
from ._analysis_gui import AnalysisSettingsData, ExperimentTypeData
from ._fov_table import WellInfo, _FOVTable
from ._graph_widgets import _MultilWellGraphWidget, _SingleWellGraphWidget
from ._image_viewer import _ImageViewer
from ._init_dialog import _InitDialog
from ._logger import LOGGER
from ._old_plate_model import OldPlate
from ._plate_plan_wizard import PlatePlanWizard
from ._save_as_widgets import _SaveAsCSV, _SaveAsTiff
from ._segmentation import _CellposeSegmentation
from ._to_csv import save_analysis_data_to_csv, save_trace_data_to_csv
from ._util import (
    EVENT_KEY,
    GENOTYPE_MAP,
    PLATE_PLAN,
    SETTINGS_PATH,
    TREATMENT_MAP,
    ROIData,
    _ProgressBarWidget,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator

HCS = "hcs"
UNSELECTABLE_COLOR = "#404040"
TS = WRITERS[ZARR_TESNSORSTORE][0]
ZR = WRITERS[OME_ZARR][0]
DEFAULT_PLATE_PLAN = useq.WellPlatePlan(
    plate=useq.WellPlate.from_str("coverslip-18mm-square"),
    a1_center_xy=(0.0, 0.0),
    selected_wells=((0,), (0,)),
)


class PlateViewer(QMainWindow):
    """A widget for displaying a plate preview."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        labels_directory: str | None = None,
        analysis_directory: str | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Plate Viewer")
        self.setWindowIcon(QIcon(icon(MDI6.view_comfy, color="#00FF00")))

        # add central widget
        self._central_widget = QWidget(self)
        self._central_widget_layout = QVBoxLayout(self._central_widget)
        self._central_widget_layout.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(self._central_widget)

        self._data: TensorstoreZarrReader | OMEZarrReader | None = None
        self._labels_path = labels_directory
        self._analysis_path = analysis_directory

        self._analysis_data: dict[str, dict[str, ROIData]] = {}

        self._plate_plan_wizard = PlatePlanWizard(self)
        self._plate_plan_wizard.hide()
        self._default_plate_plan: bool = False

        # add menu bar
        self.menu_bar = QMenuBar(self)
        self.file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Load Data and Set Directories...", self)
        open_action.setToolTip(
            "Load a zarr datastore and directories for labels and analysis data."
        )
        open_action.triggered.connect(self._show_data_input_dialog)
        save_as_tiff_action = QAction("Save Data as Tiff...", self)
        save_as_tiff_action.triggered.connect(self._show_save_as_tiff_dialog)
        save_as_csv_action = QAction("Save Analysis Data as CSV...", self)
        save_as_csv_action.triggered.connect(self._show_save_as_csv_dialog)
        self.file_menu.addAction(open_action)
        self.file_menu.addAction(save_as_tiff_action)
        self.file_menu.addAction(save_as_csv_action)
        self.setMenuBar(self.menu_bar)

        # scene and view for the plate map
        self._plate_view = WellPlateView()
        self._plate_view.setDragMode(WellPlateView.DragMode.NoDrag)
        self._plate_view.setSelectionMode(WellPlateView.SelectionMode.SingleSelection)

        # table for the fields of view
        self._fov_table = _FOVTable(self)
        self._fov_table.itemSelectionChanged.connect(
            self._on_fov_table_selection_changed
        )
        self._fov_table.doubleClicked.connect(self._on_fov_double_click)

        # image viewer
        self._image_viewer = _ImageViewer(self)
        self._image_viewer.valueChanged.connect(self._update_graphs_with_roi)

        # left widgets -------------------------------------------------
        left_group = QGroupBox()
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(5)
        left_layout.addWidget(self._plate_view)
        left_layout.addWidget(self._fov_table)

        # splitter for the plate map and the fov table
        self.splitter_top_left = QSplitter(
            parent=self, orientation=Qt.Orientation.Vertical
        )
        self.splitter_top_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_top_left.setChildrenCollapsible(False)
        self.splitter_top_left.addWidget(self._plate_view)
        self.splitter_top_left.addWidget(self._fov_table)
        top_left_group = QGroupBox()
        top_left_layout = QVBoxLayout(top_left_group)
        top_left_layout.setContentsMargins(10, 10, 10, 10)
        top_left_layout.addWidget(self.splitter_top_left)

        # splitter for the plate map/fov table and the image viewer
        self.splitter_bottom_left = QSplitter(
            parent=self, orientation=Qt.Orientation.Vertical
        )
        self.splitter_bottom_left.setContentsMargins(0, 0, 0, 0)
        self.splitter_bottom_left.setChildrenCollapsible(False)
        self.splitter_bottom_left.addWidget(top_left_group)
        self.splitter_bottom_left.addWidget(self._image_viewer)

        # right widgets --------------------------------------------------

        # tab widget
        self._tab = QTabWidget(self)
        self._tab.currentChanged.connect(self._on_tab_changed)

        # analysis tab
        self._analysis_tab = QWidget()
        self._tab.addTab(self._analysis_tab, "Analysis Tab")

        # Create a scroll area for the analysis tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create a widget to hold the analysis content
        analysis_content_widget = QWidget()
        scroll_area.setWidget(analysis_content_widget)

        # Set up the main layout for the analysis tab
        analysis_tab_layout = QVBoxLayout(self._analysis_tab)
        analysis_tab_layout.setContentsMargins(0, 0, 0, 0)
        analysis_tab_layout.addWidget(scroll_area)

        self._segmentation_wdg = _CellposeSegmentation(self)
        self._segmentation_wdg.segmentationFinished.connect(
            self._on_fov_table_selection_changed
        )

        self._analysis_wdg = _AnalyseCalciumTraces(self)

        # Layout for the scrollable content
        analysis_layout = QVBoxLayout(analysis_content_widget)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(15)
        analysis_layout.addWidget(self._segmentation_wdg)
        analysis_layout.addWidget(self._analysis_wdg)
        analysis_layout.addStretch(1)

        # single wells visualization tab
        self._single_well_vis_tab = QWidget()
        self._tab.addTab(self._single_well_vis_tab, "Single Wells Visualization Tab")
        single_well_vis_layout = QGridLayout(self._single_well_vis_tab)
        single_well_vis_layout.setContentsMargins(5, 5, 5, 5)
        single_well_vis_layout.setSpacing(5)

        self._single_well_graph_wdg_1 = _SingleWellGraphWidget(self)
        self._single_well_graph_wdg_2 = _SingleWellGraphWidget(self)
        self._single_well_graph_wdg_3 = _SingleWellGraphWidget(self)
        self._single_well_graph_wdg_4 = _SingleWellGraphWidget(self)
        single_well_vis_layout.addWidget(self._single_well_graph_wdg_1, 0, 0)
        single_well_vis_layout.addWidget(self._single_well_graph_wdg_2, 0, 1)
        single_well_vis_layout.addWidget(self._single_well_graph_wdg_3, 1, 0)
        single_well_vis_layout.addWidget(self._single_well_graph_wdg_4, 1, 1)

        self.SW_GRAPHS = [
            self._single_well_graph_wdg_1,
            self._single_well_graph_wdg_2,
            self._single_well_graph_wdg_3,
            self._single_well_graph_wdg_4,
        ]

        # connect the roiSelected signal from the graphs to the image viewer so we can
        # highlight the roi in the image viewer when a roi is selected in the graph
        for graph in self.SW_GRAPHS:
            graph.roiSelected.connect(self._highlight_roi)

        # multi wells visualization tab
        self._multi_well_vis_tab = QWidget()
        self._tab.addTab(self._multi_well_vis_tab, "Multi Wells Visualization Tab")
        multi_well_layout = QGridLayout(self._multi_well_vis_tab)
        multi_well_layout.setContentsMargins(5, 5, 5, 5)
        multi_well_layout.setSpacing(5)

        self._multi_well_graph_wdg_1 = _MultilWellGraphWidget(self)
        multi_well_layout.addWidget(self._multi_well_graph_wdg_1, 0, 0)

        self.MW_GRAPHS = [self._multi_well_graph_wdg_1]

        # splitter between the plate map/fov table/image viewer and the graphs
        self.main_splitter = QSplitter(self)
        self.main_splitter.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.splitter_bottom_left)
        self.main_splitter.addWidget(self._tab)

        # add widgets to central widget
        self._central_widget_layout.addWidget(self.main_splitter)

        self._plate_view.selectionChanged.connect(self._on_scene_well_changed)

        self._loading_bar = _ProgressBarWidget(self)

        self.showMaximized()

        self._set_splitter_sizes()

        # TO REMOVE, IT IS ONLY TO TEST________________________________________________
        # fmt off
        # data = "tests/test_plate_viewer/data/evoked/evk.tensorstore.zarr"
        # self._pv_labels_path = "tests/test_plate_viewer/data/evoked/evk_labels"
        # self._pv_analysis_path = "tests/test_plate_viewer/data/evoked/evk_analysis"
        # self.initialize_widget(data, self._pv_labels_path, self._pv_analysis_path)

        # data = "tests/test_plate_viewer/data/spontaneous/spont.tensorstore.zarr"
        # self._labels_path = "tests/test_plate_viewer/data/spontaneous/spont_labels"
        # self._analysis_path = "tests/test_plate_viewer/data/spontaneous/spont_analysis"  # noqa: E501
        # self.initialize_widget(data, self._labels_path, self._analysis_path)
        # fmt: on
        # ____________________________________________________________________________

    @property
    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @property
    def pv_labels_path(self) -> str | None:
        return self._labels_path

    @pv_labels_path.setter
    def pv_labels_path(self, value: str | None) -> None:
        self._labels_path = value
        self._segmentation_wdg.labels_path = value
        self._on_fov_table_selection_changed()

    @property
    def analysis_path(self) -> str | None:
        return self._analysis_path

    @analysis_path.setter
    def analysis_path(self, value: str) -> None:
        self._analysis_path = value
        self._analysis_wdg.analysis_path = value
        self._load_and_set_analysis_data(value)

    @property
    def analysis_data(self) -> dict[str, dict[str, ROIData]]:
        """Return the analysis data."""
        return self._analysis_data

    @analysis_data.setter
    def analysis_data(self, value: dict[str, dict[str, ROIData]]) -> None:
        """Set the analysis data."""
        self._analysis_data = value

    # PUBLIC METHODS-------------------------------------------------------------------

    def initialize_widget(
        self, datastore_path: str, labels_path: str = "", analysis_path: str = ""
    ) -> None:
        """Initialize the widget with given datastore, labels and analysis path."""
        # CLEARING---------------------------------------------------------------------

        self._clear_widget_before_initialization()

        # DATASTORE--------------------------------------------------------------------

        # select which reader to use for the datastore
        if datastore_path.endswith(TS):
            # read tensorstore
            self._data = TensorstoreZarrReader(datastore_path)
        elif datastore_path.endswith(ZR):
            # read ome zarr
            self._data = OMEZarrReader(datastore_path)
        else:
            self._data = None
            show_error_dialog(
                self,
                f"Unsupported file format! Only {WRITERS[ZARR_TESNSORSTORE][0]} and"
                f" {WRITERS[OME_ZARR][0]} are supported.",
            )
            return

        self._data = cast((TensorstoreZarrReader | OMEZarrReader), self._data)
        if self._data.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot use the  `PlateViewer` without "
                "the useq.MDASequence in the datastore metadata!",
            )
            return

        # LOAD ANALYSIS DATA-----------------------------------------------------------

        self._analysis_path = analysis_path
        self._labels_path = labels_path

        # load analysis json file if the analysis path is set
        if self._analysis_path:
            self._load_and_set_analysis_data(self._analysis_path)

        # LOAD PLATE-------------------------------------------------------------------
        plate = self._load_plate_plan(self._data.sequence.stage_positions)

        # UPDATE SEGMENTATION AND ANALYSIS WIDGETS-------------------------------------
        self._set_widgets_data(plate)

    # WIDGET INITIALIZATION------------------------------------------------------------

    def _set_splitter_sizes(self) -> None:
        """Set the initial sizes for the splitters."""
        splitter_and_sizes = (
            (self.splitter_top_left, [0.73, 0.27]),
            (self.splitter_bottom_left, [0.50, 0.50]),
            (self.main_splitter, [0.30, 0.70]),
        )
        for splitter, sizes in splitter_and_sizes:
            total_size = splitter.size().width()
            splitter.setSizes([int(size * total_size) for size in sizes])

    # DATA INITIALIZATION--------------------------------------------------------------

    def _show_data_input_dialog(self) -> None:
        """Show dialog to select zarr datastore, segmentation and analysis path."""
        init_dialog = _InitDialog(
            self,
            datastore_path=(str(self._data.path) if self._data is not None else None),
            labels_path=self._labels_path,
            analysis_path=self._analysis_path,
        )
        init_dialog.resize(600, init_dialog.sizeHint().height())
        if init_dialog.exec():
            self.initialize_widget(*init_dialog.value())

    def _clear_widget_before_initialization(self) -> None:
        """Clear the widget before initializing it with new data."""
        # clear the datastore
        self._data = None
        # clear fov table
        self._fov_table.clear()
        # clear scene
        self._plate_view.clear()
        # clear the image viewer cache
        self._image_viewer._viewer._contour_cache.clear()
        # clear the analysis data
        self._analysis_data.clear()
        # clear the segmentation widget
        self._segmentation_wdg.data = None
        self._segmentation_wdg.labels_path = None
        # clear the analysis widget data
        self._analysis_wdg.data = None
        self._analysis_wdg.analysis_data.clear()
        self._analysis_wdg.analysis_path = None
        self._analysis_wdg.stimulation_area_path = None
        # clear the plate map
        self._analysis_wdg._analysis_settings_gui._plate_map_wdg.clear()
        # no plate flag
        self._default_plate_plan = False

    def _load_and_set_analysis_data(self, path: str | Path) -> None:
        """Load the analysis data from the given JSON file."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            show_error_dialog(
                self, f"Error while loading the file. Path {path} does not exist!"
            )
            return
        if not path.is_dir():
            show_error_dialog(
                self, f"Error while loading the file. Path {path} is not a directory!"
            )
            return

        # start the waiting progress bar
        self._init_loading_bar("Loading Analysis Data...")

        create_worker(
            self._load_and_set_data_from_json,
            path=path,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress,
                "finished": self._on_loading_finished,
                "errored": self._on_loading_finished,
            },
        )

    def _on_loading_finished(self) -> None:
        """Called when the loading of the analysis data is finished."""
        self._loading_bar.hide()

    def _load_and_set_data_from_json(
        self, path: Path
    ) -> Generator[int | str, None, None]:
        """Load the analysis data from the given JSON file."""
        json_files = self._filter_data(list(path.glob("*.json")))
        self._loading_bar.setRange(0, len(json_files))
        try:
            # loop over the files in the directory
            for idx, f in enumerate(tqdm(json_files, desc="Loading Analysis Data")):
                LOGGER.debug(f"Loading file: {f}")
                yield idx + 1
                # get the name of the file without the extensions
                well = f.name.removesuffix(f.suffix)
                # create the dict for the well
                self._analysis_data[well] = {}
                # open the data for the well
                with open(f) as file:
                    data = {}
                    try:
                        data = cast(dict, json.load(file))
                    except json.JSONDecodeError as e:
                        msg = f"Error reading the analysis data: {e}"
                        LOGGER.error(msg)
                        yield msg  # for showing the error dialog
                        self._analysis_data = data
                    # if the data is empty, continue
                    if not data:
                        continue
                    # loop over the rois
                    for roi in data.keys():
                        if not roi.isdigit():
                            # this is the case of global data
                            # (e.g. cubic or linear global connectivity)
                            self._analysis_data[roi] = data[roi]
                            continue
                        # get the data for the roi
                        fov_data = cast(dict, data[roi])
                        # remove any key that is not in ROIData
                        for key in list(fov_data.keys()):
                            if key not in ROIData.__annotations__:
                                fov_data.pop(key)
                        # convert to a ROIData object and add store it in _analysis_data
                        self._analysis_data[well][roi] = ROIData(**fov_data)
        except Exception as e:
            msg = f"Error loading the analysis data: {e}"
            LOGGER.error(msg)
            yield msg  # for showing the error dialog
            self._analysis_data.clear()

    def _filter_data(self, path_list: list[Path]) -> list[Path]:
        filtered_paths: list[Path] = []

        # the json file names should be in the form A1_0000.json
        for f in path_list:
            if f.name in {
                GENOTYPE_MAP,
                TREATMENT_MAP,
                SETTINGS_PATH,
                f"{PLATE_PLAN}.json",  # for compatibility with old versions
            }:
                continue
            # skip hidden files
            if f.name.startswith("."):
                continue

            name_no_suffix = f.name.removesuffix(f.suffix)  # A1_0000 or A1_0000_p0
            split_name = name_no_suffix.split("_")  # ["A1","0000"]or["A1","0000","p0"]

            if len(split_name) == 2:
                well, fov = split_name
            elif len(split_name) == 3:
                well, fov, pos = split_name
            else:
                continue

            # validate well format, only letters and numbers
            if not re.match(r"^[a-zA-Z0-9]+$", well):
                continue

            # validate fov format, only numbers
            if len(split_name) == 3:
                if not fov.isdigit():
                    continue
                if not pos[1:].isdigit():
                    continue

            filtered_paths.append(f)

        return filtered_paths

    def _set_widgets_data(self, plate: useq.WellPlate | None) -> None:
        """Update the segmentation and analysis widgets data."""
        # set the segmentation widget data
        self._segmentation_wdg.data = self._data
        self._segmentation_wdg.labels_path = self._labels_path
        # set the analysis widget data
        self._analysis_wdg.data = self._data
        self._analysis_wdg.analysis_data = self._analysis_data
        self._analysis_wdg.analysis_path = self._analysis_path
        self._analysis_wdg.labels_path = self._labels_path
        self._analysis_wdg.update_widget_form_settings()
        # set the plate map
        self._analysis_wdg._load_plate_map(plate)
        # set the stimulation mask if it exists
        if self._analysis_wdg.analysis_path:
            # if a file namend "stimulation_mask.tif" exists in the analysis path
            stim_mask = Path(self._analysis_wdg.analysis_path) / "stimulation_mask.tif"
            if stim_mask.exists():
                self._analysis_wdg._analysis_settings_gui.setValue(
                    AnalysisSettingsData(
                        experiment_type_data=ExperimentTypeData(
                            experiment_type=EVOKED, stimulation_area_path=str(stim_mask)
                        )
                    )
                )

    def _load_plate_plan(
        self, plate_plan: useq.WellPlatePlan | tuple[useq.Position, ...] | None = None
    ) -> useq.WellPlate | None:
        """Load the plate from the datastore."""
        if self._data is None or plate_plan is None:
            return None

        final_plate_plan: useq.WellPlatePlan | None = None
        # if already a WellPlatePlan, use it directly
        if isinstance(plate_plan, useq.WellPlatePlan):
            final_plate_plan = plate_plan
        else:
            # try to load from various sources
            final_plate_plan = self._resolve_plate_plan()
            # if is the default plate plan, set the no_plate flag
            if final_plate_plan == DEFAULT_PLATE_PLAN:
                self._default_plate_plan = True

            # save the resolved plate plan if we have an analysis path
            if final_plate_plan and self._analysis_path:
                self._save_plate_plan_json_settings(final_plate_plan)

        if final_plate_plan is None:
            return None

        plate = final_plate_plan.plate
        self._draw_plate_with_selection(plate, final_plate_plan)
        return plate

    def _resolve_plate_plan(self) -> useq.WellPlatePlan | None:
        """Resolve plate plan from various sources in order of preference."""
        # try loading from JSON file
        if plate_plan := self._load_plate_plan_from_json_settings():
            return plate_plan

        # try loading from old metadata
        if plate_plan := self._retrieve_plate_plan_from_old_metadata():
            return plate_plan

        # try using the wizard
        if self._plate_plan_wizard.exec():
            return self._plate_plan_wizard.value()

        # if no HCSWizard was used but single position list was created,
        # fallback to a default square coverslip plate plan
        self._default_plate_plan = True
        return DEFAULT_PLATE_PLAN

    def _load_plate_plan_from_json_settings(self) -> useq.WellPlatePlan | None:
        """Load plate plan from JSON file if it exists."""
        if not self._analysis_path:
            return None

        settings_json_file = Path(self._analysis_path) / SETTINGS_PATH
        if not settings_json_file.exists():
            return None

        try:
            with open(settings_json_file) as f:
                settings = cast(dict, json.load(f))
                pp = settings.get(PLATE_PLAN)
                return useq.WellPlatePlan.model_validate(pp) if pp else None
        except (json.JSONDecodeError, ValidationError) as e:
            LOGGER.warning(f"Failed to load plate plan from {settings_json_file}: {e}")
            return None

    def _save_plate_plan_json_settings(self, plate_plan: useq.WellPlatePlan) -> None:
        """Save plate plan to JSON file."""
        if not self._analysis_path:
            return
        try:
            settings_json_file = Path(self._analysis_path) / SETTINGS_PATH

            # Read existing settings if file exists
            settings = {}
            if settings_json_file.exists():
                with open(settings_json_file) as f:
                    settings = json.load(f)

            # Update the plate plan
            settings[PLATE_PLAN] = plate_plan.model_dump()

            # Write back the complete settings
            with open(settings_json_file, "w") as f:
                json.dump(settings, f, indent=4)
        except OSError as e:
            LOGGER.error(f"Failed to save plate plan: {e}")

    def _draw_plate_with_selection(
        self, plate: useq.WellPlate, plate_plan: useq.WellPlatePlan
    ) -> None:
        """Draw the plate and disable non-selected wells."""
        self._plate_view.drawPlate(plate)

        wells = self._plate_view._well_items
        selected_indices = {
            tuple(plate_plan.selected_well_indices[i])
            for i in range(len(plate_plan.selected_well_indices))
        }

        for r, c in wells.keys():
            if (r, c) not in selected_indices:
                self._plate_view.setWellColor(r, c, UNSELECTABLE_COLOR)

    def _retrieve_plate_plan_from_old_metadata(self) -> useq.WellPlatePlan | None:
        """Retrieve the plate plan from the old metadata version."""
        if self._data is None:
            return None

        if self._data.sequence is None:
            return None

        meta = cast(dict, self._data.sequence.metadata.get(PYMMCW_METADATA_KEY, {}))

        plate_plan: useq.WellPlatePlan | None = None

        try:
            # in the old version the HCS metadata was in the root of the metadata
            if old_hcs_meta := meta.get(HCS, {}):
                old_plate = old_hcs_meta.get("plate")
                if not old_plate:
                    return None

                old_plate = (
                    old_plate
                    if isinstance(old_plate, OldPlate)
                    else OldPlate(**old_plate)
                )

                # old plate to new useq.WellPlate
                plate = useq.WellPlate(
                    name=old_plate.id,
                    rows=old_plate.rows,
                    columns=old_plate.columns,
                    well_spacing=(old_plate.well_spacing_x, old_plate.well_spacing_y),
                    well_size=(old_plate.well_size_x, old_plate.well_size_y),
                    circular_wells=old_plate.circular,
                )

                # old_meta should be like this:
                # plate: OldPlate] = None
                # wells: list[Well] = None
                # name: str
                # row: int
                # column: int
                # calibration: CalibrationData = None
                # plate: OldPlatePlate] = None
                # well_A1_center: tuple[float, float] = None
                # rotation_matrix: list[list[float]] = None
                # calibration_positions_a1: list[tuple[float, float]] = None
                # calibration_positions_an: list[tuple[float, float]] = None
                # positions: list[Position] = None

                # group the selected wells by row and column
                selected_wells = tuple(
                    zip(
                        *(
                            (well["row"], well["column"])
                            for well in old_hcs_meta.get("wells", [])
                        )
                    )
                )
                # create useq plate plan
                plate_plan = useq.WellPlatePlan(
                    plate=plate,
                    a1_center_xy=old_hcs_meta["calibration"]["well_A1_center"],
                    selected_wells=cast(
                        tuple[tuple[int, int], tuple[int, int]], selected_wells
                    ),
                )
            return plate_plan

        except Exception as e:
            show_error_dialog(self, "Cannot find the plate plan in the metadata!")
            LOGGER.error(f"Error retrieving the plate plan: {e}")
            return None

    # ---------------------WIDGETS------------------------------------

    def _init_loading_bar(self, text: str) -> None:
        """Reset the loading bar."""
        self._loading_bar.setEnabled(True)
        self._loading_bar.setText(text)
        self._loading_bar.setValue(0)
        self._loading_bar.showPercentage(True)
        self._loading_bar.show()

    def _update_graphs_with_roi(self, roi: int) -> None:
        """Update the graphs with the given roi.

        This function is called when a roi is selected in the image viewer and will
        update the graphs with the traces of the selected roi.
        """
        # get the current tab index
        idx = self._tab.currentIndex()
        if idx == 0:
            return
        for graph in self.SW_GRAPHS:
            if graph._combo.currentText() == "None":
                continue
            graph._choose_dysplayed_traces.setChecked(True)
            graph._choose_dysplayed_traces._roi_le.setText(str(roi))
            graph._choose_dysplayed_traces._update()

    def _on_tab_changed(self, idx: int) -> None:
        """Update the graph combo boxes when the tab is changed."""
        # skip if the tab is the analysis tab
        if idx == 0:
            return

        # if single wells tab is selected
        if idx == 1:
            # get the current fov
            value = self._fov_table.value() if self._fov_table.selectedItems() else None
            if value is None:
                return
            fov_data = self._get_fov_data(value)
            # update the graphs combo boxes
            self._update_single_wells_graphs_combo(combo_red=(fov_data is None))

        # if multi wells tab is selected
        elif idx == 2:
            self._update_multi_wells_graphs_combo()

    def _highlight_roi(self, roi: str | list[str]) -> None:
        """Highlight the selected roi in the image viewer."""
        if isinstance(roi, list):
            roi = ",".join(roi)
        self._image_viewer._roi_number_le.setText(roi)
        self._image_viewer._highlight_rois()

    def _show_save_as_tiff_dialog(self) -> None:
        """Show the save as tiff dialog."""
        if self._data is None or (sequence := self._data.sequence) is None:
            show_error_dialog(
                self,
                "No data to save or useq.MDASequence not found! Cannot save the data.",
            )
            return

        dialog = _SaveAsTiff(self)

        if dialog.exec():
            path, positions = dialog.value()

            if not Path(path).is_dir():
                show_error_dialog(
                    self, f"The path {path} is not a directory! Cannot save the data."
                )
                return

            # start the waiting progress bar
            self._init_loading_bar("Saving as tiff...")
            self._loading_bar.setRange(0, len(positions))

            create_worker(
                self._save_as_tiff,
                path=path,
                positions=positions,
                sequence=sequence,
                _start_thread=True,
                _connect={
                    "yielded": self._update_progress,
                    "finished": self._on_loading_finished,
                },
            )

    def _save_as_tiff(
        self, path: str, positions: list[int], sequence: useq.MDASequence
    ) -> Generator[int, None, None]:
        """Save the selected positions as tiff files."""
        # TODO: multithreading or multiprocessing
        # TODO: also save metadata
        if not self._data:
            return
        if not positions:
            positions = list(range(len(sequence.stage_positions)))
        for pos in tqdm(positions, desc="Saving as tiff"):
            data, meta = self._data.isel(p=pos, metadata=True)
            # the "Event" key was used in the old metadata format
            event_key = EVENT_KEY if EVENT_KEY in meta[0] else "Event"
            # get the well name from metadata
            pos_name = (
                meta[0].get(event_key, {}).get("pos_name", f"pos_{str(pos).zfill(4)}")
            )
            # save the data as tiff
            tifffile.imwrite(Path(path) / f"{pos_name}.tiff", data)
            yield pos + 1

    def _show_save_as_csv_dialog(self) -> None:
        """Show the save as csv dialog."""
        if not self._analysis_data:
            show_error_dialog(self, "No data to save! Run or load analysis data first.")
            return

        dialog = _SaveAsCSV(self)
        dialog.resize(500, dialog.sizeHint().height())

        if dialog.exec():
            path = dialog.value()
            if not Path(path).is_dir():
                show_error_dialog(
                    self, f"The path {path} is not a directory! Cannot save the data."
                )
                return

            save_trace_data_to_csv(path, self._analysis_data)
            save_analysis_data_to_csv(path, self._analysis_data)

    def _update_progress(self, value: int | str) -> None:
        """Update the progress bar value."""
        if isinstance(value, str):
            show_error_dialog(self, value)
        else:
            self._loading_bar.setValue(value)

    def _on_scene_well_changed(self) -> None:
        """Update the FOV table when a well is selected."""
        self._fov_table.clear()
        self._image_viewer._clear_highlight()

        if self._data is None:
            return

        if self._data.sequence is None:
            show_error_dialog(
                self,
                "useq.MDASequence not found! Cannot retrieve the Well data without "
                "the tensorstore useq.MDASequence!",
            )
            return

        well_dict: set[QAbstractGraphicsShapeItem] = self._plate_view._selected_items
        if not well_dict or len(well_dict) != 1:
            return
        well_name = next(iter(well_dict)).data(DATA_POSITION).name

        # add the fov per position to the table
        for idx, pos in enumerate(self._data.sequence.stage_positions):
            if self._default_plate_plan:
                self._fov_table.add_position(WellInfo(idx, pos))
            elif pos.name and well_name in pos.name:
                self._fov_table.add_position(WellInfo(idx, pos))

        if self._fov_table.rowCount() > 0:
            self._fov_table.selectRow(0)

    def _on_fov_table_selection_changed(self) -> None:
        """Update the image viewer with the first frame of the selected FOV."""
        self._image_viewer._clear_highlight()
        value = self._fov_table.value() if self._fov_table.selectedItems() else None

        if value is None:
            self._image_viewer.setData(None, None)
            self._update_single_wells_graphs_combo(combo_red=True, clear=True)
            return

        if self._data is None:
            return

        if not self._data.sequence:
            return

        # get a single frame for the selected FOV (at 2/3 of the time points)
        t = int(len(self._data.sequence.stage_positions) / 3 * 2)
        data = cast(np.ndarray, self._data.isel(p=value.pos_idx, t=t, c=0))
        # get labels if they exist
        labels = self._get_labels(value)
        # get the analysis data for the current fov if it exists
        fov_data = self._get_fov_data(value)
        # flip data and labels vertically or will look different from the StackViewer
        data = np.flip(data, axis=0)
        labels = np.flip(labels, axis=0) if labels is not None else None
        self._image_viewer.setData(data, labels)
        self._set_graphs_fov(value)

        self._update_single_wells_graphs_combo(
            combo_red=(fov_data is None), clear=(fov_data is None)
        )

    def _get_fov_data(self, value: WellInfo) -> dict[str, ROIData] | None:
        """Get the analysis data for the given FOV."""
        fov_name = f"{value.fov.name}_p{value.pos_idx}"
        fov_data = self._analysis_data.get(str(value.fov.name), None)
        # use the old name we used to save the data (without position index. e.g. "_p0")
        if fov_data is None:
            fov_data = self._analysis_data.get(fov_name, None)
        return fov_data

    def _set_graphs_fov(self, value: WellInfo | None) -> None:
        """Set the FOV title for the graphs."""
        if value is None:
            return
        title = value.fov.name or f"Position {value.pos_idx}"
        self._update_single_wells_graphs_combo(set_title=title)

    def _get_labels(self, value: WellInfo) -> np.ndarray | None:
        """Get the labels for the given FOV."""
        if self._labels_path is None:
            return None

        if not Path(self._labels_path).is_dir():
            show_error_dialog(
                self,
                f"Error while loading the labels. Path {self._labels_path} is not a "
                "directory!",
            )
            return None
        # the labels tif file should have the same name as the position
        # and should end with _on where n is the position number (e.g. C3_0000_p0.tif)
        pos_idx = f"p{value.pos_idx}"
        pos_name = value.fov.name
        for f in Path(self._labels_path).iterdir():
            name = f.name.replace(f.suffix, "")
            if pos_name and pos_name in f.name and name.endswith(f"_{pos_idx}"):
                return tifffile.imread(f)  # type: ignore
        return None

    def _on_fov_double_click(self) -> None:
        """Open the selected FOV in a new StackViewer window."""
        value = self._fov_table.value() if self._fov_table.selectedItems() else None
        if value is None or self._data is None:
            return

        data = self._data.isel(p=value.pos_idx)
        viewer = NDViewer(data, parent=self)
        viewer._ndims_btn.hide()
        viewer.setWindowTitle(value.fov.name or f"Position {value.pos_idx}")
        viewer.setWindowFlag(Qt.WindowType.Dialog)
        viewer.show()

    def _update_single_wells_graphs_combo(
        self,
        set_title: str | None = None,
        combo_red: bool = False,
        clear: bool = False,
    ) -> None:
        for sw_graph in self.SW_GRAPHS:
            if set_title is not None:
                sw_graph.fov = set_title

            if clear:
                sw_graph.clear_plot()

            sw_graph.set_combo_text_red(combo_red)

    def _update_multi_wells_graphs_combo(self) -> None:
        for mw_graph in self.MW_GRAPHS:
            mw_graph.set_combo_text_red(not self._analysis_data)
