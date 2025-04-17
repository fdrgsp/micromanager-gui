from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._plot_methods import (
    plot_multi_cond_data,
    plot_multi_well_data,
    plot_single_well_data,
)
from ._util import (
    COND1,
    COND2,
    MULTI_WELL_COMBO_OPTIONS,
    SINGLE_WELL_COMBO_OPTIONS,
    ROIData,
)

if TYPE_CHECKING:
    from ._fov_table import WellInfo
    from ._plate_viewer import PlateViewer

RED = "#C33"
HEIGHT = 20
RANDOM_CHOICE = 5
SEPARATOR = "_"


def _get_fov_data(
    table_data: WellInfo, analysis_data: dict[str, dict[str, ROIData]]
) -> dict[str, ROIData] | None:
    """Return the analysis data for the current FOV."""
    fov_name = f"{table_data.fov.name}_p{table_data.pos_idx}"
    # if the well is not in the analysis data, use the old name we used to store
    # the data (without the position index. e.g. "_p0")
    if fov_name not in analysis_data:
        fov_name = str(table_data.fov.name)
    return analysis_data.get(fov_name, None)


class _DisplaySingleWellTraces(QGroupBox):
    def __init__(self, parent: _SingleWellGraphWidget) -> None:
        super().__init__(parent)
        self.setTitle("Choose which ROI to display")
        self.setCheckable(True)
        self.setChecked(False)

        self.setToolTip(
            "By default, the widget will display the traces form all the ROIs from the "
            "current FOV. Here you can choose to only display a subset of ROIs. You "
            "can input a range (e.g. 1-10 to plot the first 10 ROIs), single ROIs "
            "(e.g. 30, 33 to plot ROI 30 and 33) or, if you only want to pick n random "
            "ROIs, you can type 'rnd' followed by the number or ROIs you want to "
            "display (e.g. rnd10 to plot 10 random ROIs)."
        )

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )

        self._graph: _SingleWellGraphWidget = parent

        self._roi_le = QLineEdit()
        self._roi_le.setPlaceholderText("e.g. 1-10, 30, 33 or rnd10")
        # when pressing enter in the line edit, update the graph
        self._roi_le.returnPressed.connect(self._update)
        self._update_btn = QPushButton("Update", self)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(QLabel("ROIs:"))
        main_layout.addWidget(self._roi_le)
        main_layout.addWidget(self._update_btn)
        self._update_btn.clicked.connect(self._update)

        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, state: bool) -> None:
        """Enable or disable the random spin box and the update button."""
        if not state:
            self._graph._on_combo_changed(self._graph._combo.currentText())
        else:
            self._update()

    def _update(self) -> None:
        """Update the graph with random traces."""
        self._graph.clear_plot()
        text = self._graph._combo.currentText()
        table_data = self._graph._plate_viewer._fov_table.value()
        if table_data is None:
            return
        data = _get_fov_data(table_data, self._graph._plate_viewer._analysis_data)
        if data is not None:
            rois = self._get_rois(data)
            if rois is None:
                return
            plot_single_well_data(self._graph, data, text, rois=rois)

    def _get_rois(self, data: dict) -> list[int] | None:
        """Return the list of ROIs to be displayed."""
        text = self._roi_le.text()
        if not text:
            return None
        # return n random rois
        try:
            if text[:3] == "rnd" and text[3:].isdigit():
                random_keys = np.random.choice(
                    list(data.keys()), int(text[3:]), replace=False
                )
                return list(map(int, random_keys))
        except ValueError:
            return None
        # parse the input string
        rois = self._parse_input(text)
        return rois or None

    def _parse_input(self, input_str: str) -> list[int]:
        """Parse the input string and return a list of ROIs."""
        parts = input_str.split(",")
        numbers: list[int] = []
        for part in parts:
            part = part.strip()  # remove any leading/trailing whitespace
            if "-" in part:
                with contextlib.suppress(ValueError):
                    start, end = map(int, part.split("-"))
                    numbers.extend(range(start, end + 1))
            else:
                with contextlib.suppress(ValueError):
                    numbers.append(int(part))
        return numbers


class _SingleWellGraphWidget(QWidget):
    roiSelected = Signal(str)

    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._combo = QComboBox(self)
        self._combo.addItems(["None", *SINGLE_WELL_COMBO_OPTIONS])
        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._save_btn = QPushButton("Save", self)
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._save_btn, 0)

        self._choose_dysplayed_traces = _DisplaySingleWellTraces(self)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._choose_dysplayed_traces)
        layout.addWidget(self.canvas)

        self.set_combo_text_red(True)

    @property
    def fov(self) -> str:
        return self._fov

    @fov.setter
    def fov(self, fov: str) -> None:
        self._fov = fov
        self._on_combo_changed(self._combo.currentText())

    def clear_plot(self) -> None:
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()

    def set_combo_text_red(self, state: bool) -> None:
        """Set the combo text color to red if state is True or to black otherwise."""
        if state:
            self._combo.setStyleSheet(f"color: {RED};")
        else:
            self._combo.setStyleSheet("")

    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        # clear the plot
        self.clear_plot()
        if text == "None" or not self._fov:
            return
        # get the data for the current fov
        table_data = self._plate_viewer._fov_table.value()
        if table_data is None:
            return
        data = _get_fov_data(table_data, self._plate_viewer._analysis_data)
        if data is not None:
            plot_single_well_data(self, data, text, rois=None)
            if self._choose_dysplayed_traces.isChecked():
                self._choose_dysplayed_traces._update()

    def _on_save(self) -> None:
        """Save the current plot as a .png file."""
        # open a file dialog to select the save location
        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", name, "PNG Image (*.png)"
        )
        if not filename:
            return
        self.figure.savefig(filename, dpi=300)


class _MultiConditionSelection(QGroupBox):
    def __init__(self, parent: _MultilWellGraphWidget) -> None:
        super().__init__(parent)
        self.setTitle("Choose which well(s)/condition(s) to display.")
        self.setCheckable(True)
        self.setChecked(False)

        self.setToolTip(
            "By default, the widget will display data from all the wells if no specific"
            "wells/conditions selected. Here you can enter the wells you want to "
            "visualize or select conditions to display, which is populated by"
            " the platemaps generated or loaded in the analysis tab."
        )

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )

        self._graph: _MultilWellGraphWidget = parent
        self._plate_map: dict[str, dict[str, str]] = self._graph._plate_map_data
        self._plate_map_color: dict[str, str] = self._graph._plate_map_color

        self._cond_1_list: dict[str, list[str]] = {}
        self._cond_2_list: dict[str, list[str]] = {}

        cond_sel_layout = QHBoxLayout()
        self._cond_sel = QLineEdit()
        cond_sel_layout.addWidget(QLabel("Wells/Conditions:"))
        cond_sel_layout.addWidget(self._cond_sel)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.list_widget.setMaximumHeight(HEIGHT)
        self.list_widget.itemSelectionChanged.connect(self._update_text)

        self._update_btn = QPushButton("Update", self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addLayout(cond_sel_layout)
        main_layout.addWidget(self.list_widget)
        main_layout.addWidget(self._update_btn)
        self._update_btn.clicked.connect(self._update)

        self.toggled.connect(self._on_toggle)

    def _clear_plate_maps(self) -> None:
        """Clear plate maps saved."""
        self._plate_map.clear()
        self._plate_map_color.clear()
        self._cond_1_list.clear()
        self._cond_2_list.clear()

    def _update_condition_list(self) -> None:
        """Update the conditions to display."""
        self._clear_plate_maps()
        self._graph._update_plate_map()
        self._plate_map.update(self._graph._plate_map_data)
        self._plate_map_color.update(self._graph._plate_map_color)
        self.list_widget.clear()

        for well_name, conditions in self._plate_map.items():
            if COND1 in conditions:
                if conditions[COND1] not in self._cond_1_list:
                    self._cond_1_list[conditions[COND1]] = []
                self._cond_1_list[conditions[COND1]].append(well_name)

            if COND2 in conditions:
                if conditions[COND2] not in self._cond_2_list:
                    self._cond_2_list[conditions[COND2]] = []
                self._cond_2_list[conditions[COND2]].append(well_name)

        cond_list = list(self._cond_1_list.keys()) + list(self._cond_2_list.keys())
        self.list_widget.setMaximumHeight(min(len(cond_list), 4) * HEIGHT)
        self.list_widget.addItems(cond_list)

    def _update_text(self) -> None:
        """Update text displayed to show selected conditions."""
        self._cond_sel.setText("")
        selected_items = [item.text() for item in self.list_widget.selectedItems()]
        if selected_items:
            self._cond_sel.setText(", ".join(selected_items))
        else:
            self._cond_sel.setText("")

    def _update(self) -> None:
        """Update graphs."""
        self._graph.clear_plot()
        text = self._graph._combo.currentText()

        cond_selected = self._cond_sel.text().split(",")
        cond_selected = [cond.strip() for cond in cond_selected]

        # if the user chose conditions to visualize
        if len(self.list_widget.selectedItems()) > 0:
            wells_sel = self._get_well_from_cond(cond_selected)
            cond_ordered = self._reorder_cond(cond_selected)
            wells_by_cond = self._wells_by_cond(wells_sel, cond_ordered)
            cond_data = self._get_cond_data(
                wells_by_cond, self._graph._plate_viewer._analysis_data
            )

            if cond_data is not None:
                plot_multi_cond_data(
                    self._graph, text, cond_data, cond_ordered, self._plate_map_color
                )

        # if the user chose to visualize by wells
        else:
            well_data = self._get_fov_data_by_well(
                cond_selected, self._graph._plate_viewer._analysis_data
            )

            if well_data is not None:
                plot_multi_well_data(
                    self._graph,
                    text,
                    well_data,
                )

    def _on_toggle(self, state: bool) -> None:
        """On toggle."""
        self._update_condition_list()
        if not state:
            self._graph._on_combo_changed(self._graph._combo.currentText())
        else:
            self._update()

    def _get_fov_names(
        self, wells: list[str] | None, analysis_data: dict[str, dict[str, ROIData]]
    ) -> list[str] | None:
        """Get fov names based on the wells."""
        fovs_sel: list[str] = []

        if wells is None:
            return None

        fov_names = list(analysis_data.keys())

        for well in wells:
            for fov_name in fov_names:
                if well.lower() in fov_name.lower():
                    fovs_sel.append(fov_name)

        return fovs_sel

    def _get_fov_data(
        self,
        fovs_sel: list[str] | None,
        analysis_data: dict[str, dict[str, ROIData]],
    ) -> list[ROIData] | None:
        """Get fov data from analysis data."""
        data: list[ROIData] = []

        if fovs_sel is None:
            return None

        for fov in fovs_sel:
            if fov not in analysis_data:
                continue

            roi_data_dict = analysis_data.get(fov)

            if roi_data_dict:
                data += list(roi_data_dict.values())

        return data

    def _get_fov_data_single_roi(
        self,
        fovs_sel: list[str] | None,
        analysis_data: dict[str, dict[str, ROIData]],
    ) -> dict[str, dict[str, ROIData]] | None:
        """Get FOV data for single ROIs."""
        data: dict[str, dict[str, ROIData]] = {}

        if fovs_sel is None:
            return None

        for fov in fovs_sel:
            if fov not in analysis_data:
                continue

            roi_data_dict = analysis_data.get(fov)

            if roi_data_dict:
                data[fov] = roi_data_dict

        return data

    def _get_fov_data_by_well(
        self, well_sel: list[str], analysis_data: dict[str, dict[str, ROIData]]
    ) -> dict[str, dict[str, ROIData]] | None:
        """Group fovs by well."""
        if len(well_sel) > 1:
            wells = well_sel.copy()
        else:
            all_wells = self._graph._all_wells
            wells = list(np.random.choice(all_wells, RANDOM_CHOICE))

        fovs_sel = self._get_fov_names(wells, analysis_data)

        return (
            self._get_fov_data_single_roi(fovs_sel, analysis_data) if fovs_sel else None
        )

    # ----- multi-condition -----
    def _get_well_from_cond(self, conditions: list[str]) -> list:
        """Get the wells based on selected conditions."""
        well_sel_cond_1: list[str] = []
        well_sel_cond_2: list[str] = []
        well_sel: list[str] = []

        for cond in conditions:
            if cond in self._cond_1_list:
                well_sel_cond_1 += self._cond_1_list[cond]
            elif cond in self._cond_2_list:
                well_sel_cond_2 += self._cond_2_list[cond]

        # if no condition 1 was selected
        if len(well_sel_cond_1) == 0:
            well_sel = well_sel_cond_2

        # if no condition 2 was selected
        elif len(well_sel_cond_2) == 0:
            well_sel = well_sel_cond_1

        # if some cond1 and cond2 were selected, only display the wells that match both
        elif len(well_sel_cond_1) > 0 and len(well_sel_cond_2) > 0:
            well_sel = list(set(well_sel_cond_1) & set(well_sel_cond_2))

        return well_sel

    def _wells_by_cond(
        self,
        wells_sel: list[str],
        cond_ordered: list[str],
    ) -> dict[str, list[str]]:
        """Group wells by conditions."""
        wells_by_cond: dict[str, list[str]] = {}

        for well in wells_sel:
            well_info = self._plate_map.get(well, None)

            if well_info is None:
                continue

            cond1 = well_info.get(COND1) if COND1 in well_info else ""
            cond2 = well_info.get(COND2) if COND2 in well_info else ""

            conds = f"{cond1}{SEPARATOR}{cond2}"

            if conds not in cond_ordered:
                continue

            if conds not in wells_by_cond:
                wells_by_cond[conds] = []

            wells_by_cond[conds].append(well)

        return wells_by_cond

    def _reorder_cond(self, cond_selected: list[str]) -> list[str]:
        """Order the conditions for plotting."""
        cond1_list = [cond for cond in cond_selected if cond in self._cond_1_list]
        cond2_list = [cond for cond in cond_selected if cond in self._cond_2_list]

        if len(cond1_list) == 0 and len(cond2_list) > 0:
            cond_1_platemap = list(self._cond_1_list.keys())
            cond1_list = cond_1_platemap if len(cond_1_platemap) > 0 else [""]

        elif len(cond1_list) > 0 and len(cond2_list) == 0:
            cond_2_platemap = list(self._cond_2_list.keys())
            cond2_list = cond_2_platemap if len(cond_2_platemap) > 0 else [""]

        cond_ordered: list[str] = []
        for cond1 in cond1_list:
            for cond2 in cond2_list:
                cond_ordered.append(f"{cond1}{SEPARATOR}{cond2}")

        return cond_ordered

    def _get_cond_data(
        self,
        well_by_cond: dict[str, list[str]],
        analysis_data: dict[str, dict[str, ROIData]],
    ) -> dict[str, list[ROIData]]:
        """Get data for the wells selected."""
        cond_data: dict[str, list[ROIData]] = {}

        for cond, wells in well_by_cond.items():
            fovs_sel = self._get_fov_names(wells, analysis_data)
            fov_data = self._get_fov_data(fovs_sel, analysis_data)
            if fov_data:
                cond_data[cond] = fov_data

        return cond_data


class _MultilWellGraphWidget(QWidget):
    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._plate_map_data: dict[str, dict[str, str]] = {}
        self._plate_map_color: dict[str, str] = {}
        self._all_wells: list = []

        self._combo = QComboBox(self)
        self._combo.addItems(["None", *MULTI_WELL_COMBO_OPTIONS])
        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._save_btn = QPushButton("Save", self)
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._save_btn, 0)

        # hiding this for now, to be implemented
        self._choose_displayed_conditions = _MultiConditionSelection(self)

        selection_layout = QHBoxLayout()
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(5)
        selection_layout.addWidget(self._choose_displayed_conditions, 1)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addLayout(selection_layout)
        layout.addWidget(self.canvas)

        self.set_combo_text_red(True)

    @property
    def fov(self) -> str:
        return self._fov

    @fov.setter
    def fov(self, fov: str) -> None:
        self._fov = fov
        self._on_combo_changed(self._combo.currentText())

    def clear_plate_map(self) -> None:
        self._plate_map_data.clear()

    def clear_plot(self) -> None:
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()

    def set_combo_text_red(self, state: bool) -> None:
        """Set the combo text color to red if state is True or to black otherwise."""
        if state:
            self._combo.setStyleSheet(f"color: {RED};")
        else:
            self._combo.setStyleSheet("")

    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        # clear the plot
        self.clear_plot()

        if text == "None":
            return

        # if not display specific well or condition
        plot_multi_well_data(
            self,
            text,
            self._plate_viewer._analysis_data,
        )
        if self._choose_displayed_conditions.isChecked():
            self._choose_displayed_conditions._update()

    def _on_save(self) -> None:
        """Save the current plot as a .png file."""
        # open a file dialog to select the save location
        name = self._combo.currentText().replace(" ", "_")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", name, "PNG Image (*.png)"
        )
        if not filename:
            return
        self.figure.savefig(filename, dpi=300)

    def _update_plate_map(self) -> None:
        """Update the plate maps in multi-well tab."""
        self.clear_plate_map()

        condition_1_plate_map = self._plate_viewer._plate_map_genotype.value()
        condition_2_plate_map = self._plate_viewer._plate_map_treatment.value()

        self._all_wells = self._get_all_wells(self._plate_viewer._analysis_data)

        for data in condition_1_plate_map:
            self._plate_map_data[data.name] = {COND1: data.condition[0]}
            if data.condition[0]:
                self._plate_map_color[data.condition[0]] = data.condition[1]

        for data in condition_2_plate_map:
            if data.name in self._plate_map_data:
                self._plate_map_data[data.name][COND2] = data.condition[0]
            else:
                self._plate_map_data[data.name] = {COND2: data.condition[0]}

            if data.condition[0]:
                self._plate_map_color[data.condition[0]] = data.condition[1]

    def _get_all_wells(self, analysis_data: dict[str, dict[str, ROIData]]) -> list[str]:
        """Get a list of wells analyzed."""
        wells: list[str] = []
        for fov in analysis_data.keys():
            well = fov.split("_")[0]
            if well not in wells:
                wells.append(well)
        return wells
