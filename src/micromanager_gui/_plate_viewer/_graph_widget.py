from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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

RAW_TRACES = "Raw Traces"
DFF = "DeltaF/F0"
COMBO_OPTIONS: dict[str, Callable] = {
    RAW_TRACES: plot_raw_traces,
    DFF: plot_delta_f_over_f,
    "Traces with Peaks": plot_traces_with_peaks,
    "Mean Amplitude ± StD": plot_mean_amplitude,
    "Mean Frequency ± StD": plot_mean_frequency,
    "Raster Plot": plot_raster_plot,
}
RED = "#C33"


class _DisplayTraces(QWidget):
    def __init__(self, parent: _GraphWidget) -> None:
        super().__init__(parent)

        self._graph: _GraphWidget = parent

        # TODO: switch from QCheckBox to radio buttons and add a QLineEdit to input the
        # roi or the rois to be displayed (like 1, 2, 3 or 1-3, 5, 7-9)

        self._random_cbox = QCheckBox("Random Traces:", self)
        self._random_spin = QSpinBox(self)
        self._random_spin.setRange(1, 1000)
        self._random_spin.setValue(10)
        self._random_spin.setEnabled(False)
        self._random_btn = QPushButton("Update", self)
        self._random_btn.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._random_cbox)
        layout.addWidget(self._random_spin)
        layout.addWidget(self._random_btn)
        layout.addStretch()

        self._random_cbox.stateChanged.connect(self._enable)
        self._random_btn.clicked.connect(self._update)

    def _enable(self, state: bool) -> None:
        """Enable or disable the random spin box and the update button."""
        self._random_spin.setEnabled(state)
        self._random_btn.setEnabled(state)
        if not state:
            self._graph._on_combo_changed(self._graph._combo.currentText())

    def _update(self) -> None:
        """Update the graph with random traces."""
        self._graph.clear_plot()
        text = self._graph._combo.currentText()
        if text not in {RAW_TRACES, DFF}:
            return
        table_data = self._graph._plate_viewer._fov_table.value()
        if table_data is None:
            return
        well_name = table_data.fov.name
        if well_name in self._graph._plate_viewer._analysis_data:
            COMBO_OPTIONS[text](
                self._graph,
                self._graph._plate_viewer._analysis_data[well_name],
                self._random_spin.value(),
            )


class _GraphWidget(QWidget):
    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._combo = QComboBox(self)
        self._combo.addItems(["None", *list(COMBO_OPTIONS.keys())])
        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._random_traces_wdg = _DisplayTraces(self)
        self._random_traces_wdg.hide()
        # self._random_traces_wdg = QWidget(self)
        # _random_traces_layout = QHBoxLayout(self._random_traces_wdg)
        # _random_traces_layout.setContentsMargins(0, 0, 0, 0)
        # self._random_cbox = QCheckBox("Random Traces:", self._random_traces_wdg)
        # self._random_spin = QSpinBox(self._random_traces_wdg)
        # self._random_spin.setRange(1, 1000)
        # self._random_btn = QPushButton("Update", self._random_traces_wdg)
        # _random_traces_layout.addWidget(self._random_cbox)
        # _random_traces_layout.addWidget(self._random_spin)
        # _random_traces_layout.addWidget(self._random_btn)
        # _random_traces_layout.addStretch()

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)
        layout.addWidget(self._random_traces_wdg)
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

        self._random_traces_wdg.hide()

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

            if text in {RAW_TRACES, DFF}:
                self._random_traces_wdg.show()
                self._random_traces_wdg._random_spin.setRange(
                    1, len(self._plate_viewer._analysis_data[well_name])
                )
