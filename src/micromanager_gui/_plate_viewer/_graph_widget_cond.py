from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._plot_comparison import compare_conditions

if TYPE_CHECKING:
    from ._plate_map import PlateMapWidget
    from ._plate_viewer import PlateViewer

RED = "#C33"

# fmt: off
AVERAGE_AMPLITUDES = "Average amplitudes"
AVERAGE_FREQUENCY = "Average frequency"
AVERAGE_MAX_SLOPE = "Average max slope"
AVERAGE_RISE_TIME = "Average rise time"
AVERAGE_DECAY_TIME = "Average decay time"
AVERAGE_IEI = "Average interevent interval"
AVERAGE_CELL_SIZE = "Average cell size"
GLOBAL_CONNECTIVITY = "Global connectivity"

# dff=False, normalize=False, photobleach_corrected=False, with_peaks=False, used_for_bleach_correction=False  # noqa: E501
COMBO_OPTIONS: dict[str, dict[str, bool]] = {
    AVERAGE_AMPLITUDES: {},
    AVERAGE_FREQUENCY: {},
    AVERAGE_MAX_SLOPE: {},
    AVERAGE_RISE_TIME: {},
    AVERAGE_DECAY_TIME: {},
    AVERAGE_IEI: {},
    AVERAGE_CELL_SIZE: {},
    GLOBAL_CONNECTIVITY: {}
}
# fmt : on

class _CompareConditions(QWidget):
    def __init__(self, parent: _GraphWidget_cond) -> None:
        super().__init__(parent)


        self.setToolTip(
            "Compare treatment across genotype if genotype is given. If not, compare treatment."  # noqa: E501
        )

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )

        self._graph: _GraphWidget_cond = parent
        self._geno_dict: dict | None = None
        self._treatment_dict: dict | None = None

        self._group_sel = QComboBox()
        self._group_sel.addItems([" ", "Genotype", "Treatment"])
        self._group_sel.setPlaceholderText("Genotype")
        self._group_sel.currentTextChanged.connect(self._group_combo_change)

        self._cond_sel = QComboBox()
        self._cond_sel.setPlaceholderText("Choose the condition")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(QLabel("Group by"))
        main_layout.addWidget(self._group_sel)
        main_layout.addWidget(QLabel("Condition: "))
        main_layout.addWidget(self._cond_sel)

    def _group_combo_change(self):
        """If group changes."""
        group = self._group_sel.currentText()

        if not self._treatment_dict and \
                self._graph._plate_viewer._plate_map_treatment.cond_list:
            self._treatment_dict = \
                self._graph._plate_viewer._plate_map_treatment.cond_list
            self._graph._treatment_dict = self._treatment_dict

        if not self._geno_dict and \
            self._graph._plate_viewer._plate_map_genotype.cond_list:
            self._geno_dict = \
                self._graph._plate_viewer._plate_map_genotype.cond_list
            self._graph._geno_dict = self._geno_dict

        if self._treatment_dict or self._geno_dict:
            if group == "Genotype":
                self._cond_sel.clear()
                self._cond_sel.addItems(["All wells",
                                         *list(self._treatment_dict.keys())])
            elif group == "Treatment":
                self._cond_sel.clear()
                self._cond_sel.addItems(["All wells",
                                        *list(self._geno_dict.keys())])

class _GraphWidget_cond(QWidget):

    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent
        self._treatment_dict: dict = None
        self._geno_dict: dict = None

        self._choose_groups = _CompareConditions(self)
        self._choose_metrics = _ChooseMetrics(self)

        # self._treatment_pm: PlateMapWidget = self._plate_viewer._plate_map_treatment
        # self._geno_pm: PlateMapWidget = self._plate_viewer._plate_map_genotype

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._choose_groups)
        layout.addWidget(self._choose_metrics)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        # self._compare_btn.clicked.connect(self._choose_displayed_metrics._update)

        self.set_combo_text_red(True)

    def _compare_conditions(self):
        """Compare the conditions that the user wants."""
        self.clear_plot()
        group = self._choose_groups._group_sel.currentText()
        condition = self._choose_groups._cond_sel.currentText()
        metrics = self._choose_metrics._combo.currentText()

        print(f"        group: {group}, condition: {condition}, metrics: {metrics}")

        fovs = self._plate_viewer._analysis_data.keys()
        if group == "Genotype" and self._treatment_dict:
            wells_to_compare = self._treatment_dict[condition]['name']
            colors_to_plot = self._treatment_dict[condition]['color']
            # fov_grouped = [key for key, geno_data in self._geno_dict.items() if \
            #                any(well in geno_data['name'] for well in wells_to_compare)]
        elif group == "Treatment" and self._geno_dict:
            print("here")
            wells_to_compare = self._geno_dict[condition]['name']
            colors_to_plot = self._geno_dict[condition]['color']
            # fov_grouped = [key for key, treatment_data in self._treatment_dict.items()\
            #     if any(well in treatment_data['name'] for well in wells_to_compare)]
        elif condition == "All wells":
            ## TODO: all the wells
            return

        fovs_to_compare = [fov for fov in fovs if any(
            well in fov for well in wells_to_compare)]
        print(f" wells to compare for {condition} are {wells_to_compare}")
        print(f" fovs to compare are {fovs_to_compare}")

        data = {fov: self._plate_viewer.analysis_data[fov] for fov in fovs_to_compare\
                if fov in self._plate_viewer.analysis_data}

        compare_conditions(self, data, x_axis=group, y_axis=condition,
                            colors=colors_to_plot, **COMBO_OPTIONS[metrics])



    def clear_plot(self) -> None:
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()

    def set_combo_text_red(self, state: bool) -> None:
        """Set the combo text color to red if state is True or to black otherwise."""
        if state:
            self._choose_metrics._combo.setStyleSheet(f"color: {RED};")
        else:
            self._choose_metrics._combo.setStyleSheet("")

class _ChooseMetrics(QWidget):
    def __init__(self, parent: _GraphWidget_cond) -> None:
        super().__init__(parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        )

        self._graph: _GraphWidget_cond = parent
        self._combo = QComboBox(self)
        self._combo.addItems(["None", *list(COMBO_OPTIONS.keys())])
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        self._compare_btn = QPushButton("Compare", self)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(QLabel("Graphs: "))
        main_layout.addWidget(self._combo)
        main_layout.addWidget(self._compare_btn)
        self._compare_btn.clicked.connect(self._graph._compare_conditions)

    def _on_combo_changed(self, text: str) -> None:
        """Update the graph when the combo box is changed."""
        print("-> inside on combo changed")
        # clear the plot
        self._graph.clear_plot()
        if text == "None":
            return



