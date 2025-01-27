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

from ._plot_methods import plot_traces

if TYPE_CHECKING:
    from ._plate_viewer import PlateViewer

RED = "#C33"

# fmt: off
RAW_TRACES = "Raw Traces"
RAW_TRACES_WITH_PEAKS = "Raw Traces with Peaks"
NORMALIZED_TRACES = "Normalized Traces [0, 1]"
NORMALIZED_TRACES_WITH_PEAKS = "Normalized Traces [0, 1] with Peaks"
DFF = "DeltaF/F0"
DFF_NORMALIZED = "DeltaF/F0 Normalized [0, 1]"
DEC_DFF = "Deconvolved DeltaF/F0"
DEC_DFF_WITH_PEAKS = "Deconvolved DeltaF/F0 with Peaks"
DEC_DFF_NORMALIZED = "Deconvolved DeltaF/F0 Normalized [0, 1]"
DEC_DFF_NORMALIZED_WITH_PEAKS = "Deconvolved DeltaF/F0 Normalized [0, 1] with Peaks"
DEC_DFF_AMPLITUDE = "Deconvolved DeltaF/F0 Amplitudes"
DEC_DFF_FREQUENCY = "Deconvolved DeltaF/F0 Frequencies"

# dff=False, def=False, normalize=False, with_peaks=False, amp=False, freq=False
# NOTE: def & dec can't be True at the same time
# NOTE: amp & freq can't be True at the same time
COMBO_OPTIONS: dict[str, dict[str, bool]] = {
    RAW_TRACES: {},
    NORMALIZED_TRACES: {"normalize":True},
    DFF: {"dff":True},
    DFF_NORMALIZED: {"dff":True, "normalize":True},
    DEC_DFF: {"dec":True},
    DEC_DFF_WITH_PEAKS: {"dec":True, "with_peaks":True},
    DEC_DFF_NORMALIZED: {"dec":True, "normalize":True},
    DEC_DFF_NORMALIZED_WITH_PEAKS: {"dec":True, "normalize":True, "with_peaks":True},
    DEC_DFF_AMPLITUDE: {"dec":True, "amp":True},
    DEC_DFF_FREQUENCY: {"dec":True, "freq":True},
}
# fmt : on

class _DisplayTraces(QGroupBox):
    def __init__(self, parent: _GraphWidget) -> None:
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

        self._graph: _GraphWidget = parent

        self._roi_le = QLineEdit()
        self._roi_le.setPlaceholderText("e.g. 1-10, 30, 33 or rnd10")
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

    def _update(self) -> None:
        """Update the graph with random traces."""
        self._graph.clear_plot()
        text = self._graph._combo.currentText()
        table_data = self._graph._plate_viewer._fov_table.value()
        if table_data is None:
            return
        well_name = table_data.fov.name
        if well_name in self._graph._plate_viewer._analysis_data:
            data = self._graph._plate_viewer._analysis_data[well_name]
            rois = self._get_rois(data)
            if rois is None:
                return
            plot_traces(self._graph, data, rois=rois, **COMBO_OPTIONS[text])

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


class _GraphWidget(QWidget):

    roiSelected = Signal(str)

    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._combo = QComboBox(self)
        self._combo.addItems(["None", *list(COMBO_OPTIONS.keys())])
        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._choose_dysplayed_traces = _DisplayTraces(self)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)
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
        # get the segmentation labels
        # labels = self._plate_viewer._get_segmentation(table_data)
        # if labels is None:
        #     return
        well_name = table_data.fov.name
        if well_name in self._plate_viewer._analysis_data:
            data = self._plate_viewer._analysis_data[well_name]
            plot_traces(self, data,rois=None, **COMBO_OPTIONS[text])
            if self._choose_dysplayed_traces.isChecked():
                self._choose_dysplayed_traces._update()
