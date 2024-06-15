from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget

from ._plot_methods import (
    plot_delta_f_over_f,
    plot_mean_amplitude,
    plot_mean_frequency,
    plot_raster_plot,
    plot_raw_traces,
    plot_traces_with_peaks,
)

if TYPE_CHECKING:
    from ._plate_viewer import PlateViewer


COMBO_OPTIONS: dict[str, Callable] = {
    "Raw Traces": plot_raw_traces,
    "DeltaF/F0": plot_delta_f_over_f,
    "Traces with Peaks": plot_traces_with_peaks,
    "Mean Amplitude ± StD": plot_mean_amplitude,
    "Mean Frequency ± StD": plot_mean_frequency,
    "Raster Plot": plot_raster_plot,
}
RED = "#C33"


class _GraphWidget(QWidget):
    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._combo = QComboBox(self)
        self._combo.addItems(["None", *list(COMBO_OPTIONS.keys())])
        self._combo.currentTextChanged.connect(self._on_combo_changed)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)
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
