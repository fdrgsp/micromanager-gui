from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
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

from ._plot_methods import (
    plot_delta_f_over_f,
    plot_mean_amplitude,
    plot_mean_frequency,
    plot_normalized_traces,
    plot_normalized_traces_photobleach_corrected,
    plot_normalized_traces_photobleach_corrected_with_peaks,
    plot_normalized_traces_with_peaks,
    plot_photobleaching_fitted_curve,
    plot_raster_plot,
    plot_raw_traces,
    plot_raw_traces_photobleach_corrected,
    plot_raw_traces_photobleach_corrected_with_peaks,
    plot_raw_traces_with_peaks,
    plot_traces_used_for_bleach_correction,
)

if TYPE_CHECKING:
    from ._plate_viewer import PlateViewer

RAW_TRACES = "Raw Traces"
RAW_TRACES_WITH_PEAKS = "Raw Traces with Peaks"
RAW_TRACES_PHOTOBLEACH_CORRECTED = "Raw Traces Photobleach Corrected"
RAW_TRACES_PHOTOBLEACH_CORRECTED_WITH_PEAKS = (
    "Raw Traces Photobleach Corrected with Peaks"
)
NORMALIZED_TRACES = "Normalized Traces [0, 1]"
NORMALIZED_TRACES_PHOTOBLEACH_CORRECTED = (
    "Normalized Traces [0, 1] Photobleach Corrected"
)
NORMALIZED_TRACES_WITH_PEAKS = "Normalized Traces [0, 1] with Peaks"
NORMALIZED_TRACES_PHOTOBLEACH_CORRECTED_WITH_PEAKS = (
    "Normalized Traces [0, 1] Photobleach Correctedwith Peaks"
)
TRACES_FOR_BLEACH_CORRECTIONS = "Traces used for Photobleaching Correction"
BLEACH_FITTED_CURVE = "Photobleaching Fitted Curve"
DFF = "DeltaF/F0"

COMBO_OPTIONS: dict[str, Callable] = {
    RAW_TRACES: plot_raw_traces,
    RAW_TRACES_PHOTOBLEACH_CORRECTED: plot_raw_traces_photobleach_corrected,
    RAW_TRACES_WITH_PEAKS: plot_raw_traces_with_peaks,
    RAW_TRACES_PHOTOBLEACH_CORRECTED_WITH_PEAKS: plot_raw_traces_photobleach_corrected_with_peaks,  # noqa: E501
    NORMALIZED_TRACES: plot_normalized_traces,
    NORMALIZED_TRACES_PHOTOBLEACH_CORRECTED: plot_normalized_traces_photobleach_corrected,  # noqa: E501
    NORMALIZED_TRACES_WITH_PEAKS: plot_normalized_traces_with_peaks,
    NORMALIZED_TRACES_PHOTOBLEACH_CORRECTED_WITH_PEAKS: plot_normalized_traces_photobleach_corrected_with_peaks,  # noqa: E501
    TRACES_FOR_BLEACH_CORRECTIONS: plot_traces_used_for_bleach_correction,
    BLEACH_FITTED_CURVE: plot_photobleaching_fitted_curve,
    DFF: plot_delta_f_over_f,
    "Mean Amplitude ± StD": plot_mean_amplitude,
    "Mean Frequency ± StD": plot_mean_frequency,
    "Raster Plot": plot_raster_plot,
}
RED = "#C33"


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
            COMBO_OPTIONS[text](self._graph, data, rois)

    def _get_rois(self, data: dict) -> list[int] | None:
        """Return the list of ROIs to be displayed."""
        text = self._roi_le.text()
        if not text:
            return None
        # return n random rois
        if text[:3] == "rnd" and text[3:].isdigit():
            random_keys = np.random.choice(
                list(data.keys()), int(text[3:]), replace=False
            )
            return list(map(int, random_keys))
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
            COMBO_OPTIONS[text](self, self._plate_viewer._analysis_data[well_name])
            if self._choose_dysplayed_traces.isChecked():
                self._choose_dysplayed_traces._update()
