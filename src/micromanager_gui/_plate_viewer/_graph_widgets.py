from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QMouseEvent, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._plot_methods import plot_multi_well_data, plot_single_well_data
from ._plot_methods._main_plot import (
    MULTI_WELL_COMBO_OPTIONS_DICT,
    SINGLE_WELL_COMBO_OPTIONS_DICT,
)

if TYPE_CHECKING:
    from ._fov_table import WellInfo
    from ._plate_viewer import PlateViewer
    from ._util import ROIData

RED = "#C33"
SECTION_ROLE = Qt.ItemDataRole.UserRole + 1


class _PersistentMenu(QMenu):
    """A QMenu that stays open when checkable actions are triggered."""

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        """Override mouseReleaseEvent to prevent menu closing on checkable actions."""
        if a0 is None:
            super().mouseReleaseEvent(a0)
            return

        action = self.actionAt(a0.pos())
        if action and action.isCheckable():
            # Toggle the action state manually
            action.setChecked(not action.isChecked())
            # Emit the triggered signal manually
            action.triggered.emit(action.isChecked())
            # Don't call the parent implementation to prevent menu closing
            return
        # For non-checkable actions, use default behavior (close menu)
        super().mouseReleaseEvent(a0)


def _get_fov_data(
    table_data: WellInfo, analysis_data: dict[str, dict[str, ROIData]]
) -> dict[str, ROIData] | None:
    """Return the analysis data for the current FOV."""
    fov_name = f"{table_data.fov.name}_p{table_data.pos_idx}"
    # if the well is not in the analysis data, use the old name we used to store
    # the data (without the position index. e.g. "_p0")
    if fov_name not in analysis_data:
        fov_name = str(table_data.fov.name)
    return analysis_data.get(fov_name)


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
        data = _get_fov_data(table_data, self._graph._plate_viewer._pv_analysis_data)
        if data is not None:
            rois = self._get_rois(data, self._graph._combo.currentText())
            if rois is None:
                return
            plot_single_well_data(self._graph, data, text, rois=rois)

    def _get_rois(self, data: dict[str, ROIData], plot_text: str) -> list[int] | None:
        """Return the list of ROIs to be displayed."""
        text = self._roi_le.text()
        if not text:
            return None
        # return n random rois
        try:
            if text[:3] == "rnd" and text[3:].isdigit():
                # if the plot text contains any active word, consider only active ROIs
                active = {
                    "peaks",
                    "amplitudes",
                    "frequencies",
                    "inter-event",
                    "synchrony",
                    "correlation",
                    "hierarchical",
                }
                if any(word in plot_text.lower() for word in active):
                    data_keys = [k for k in data if data[k].active]
                else:
                    data_keys = list(data.keys())
                random_keys = np.random.choice(
                    list(data_keys), int(text[3:]), replace=False
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
    roiSelected = Signal(object)

    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._combo = QComboBox(self)
        model = QStandardItemModel()
        self._combo.setModel(model)

        # add the "None" selectable option to the combo box
        none_item = QStandardItem("None")
        model.appendRow(none_item)

        for key, value in SINGLE_WELL_COMBO_OPTIONS_DICT.items():
            section = QStandardItem(key)
            section.setFlags(Qt.ItemFlag.NoItemFlags)
            section.setData(True, SECTION_ROLE)
            model.appendRow(section)
            for item in value:
                model.appendRow(QStandardItem(item))

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
        data = _get_fov_data(table_data, self._plate_viewer._pv_analysis_data)
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


class _MultilWellGraphWidget(QWidget):
    def __init__(self, parent: PlateViewer) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer = parent

        self._fov: str = ""

        self._conditions: dict[str, bool] = {}

        self._combo = QComboBox(self)
        model = QStandardItemModel()
        self._combo.setModel(model)

        # add the "None" selectable option to the combo box
        none_item = QStandardItem("None")
        model.appendRow(none_item)

        for key, value in MULTI_WELL_COMBO_OPTIONS_DICT.items():
            section = QStandardItem(key)
            section.setFlags(Qt.ItemFlag.NoItemFlags)
            section.setData(True, SECTION_ROLE)
            model.appendRow(section)
            for item in value:
                model.appendRow(QStandardItem(item))

        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self._conditions_btn = QPushButton("Conditions...", self)
        self._conditions_btn.setEnabled(False)
        self._conditions_btn.clicked.connect(self._show_conditions_menu)

        self._save_btn = QPushButton("Save", self)
        self._save_btn.clicked.connect(self._on_save)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.addWidget(self._combo, 1)
        top.addWidget(self._conditions_btn, 0)
        top.addWidget(self._save_btn, 0)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.canvas)

        self.set_combo_text_red(True)

    @property
    def fov(self) -> str:
        return self._fov

    @fov.setter
    def fov(self, fov: str) -> None:
        self._fov = fov
        self._on_combo_changed(self._combo.currentText())

    @property
    def conditions(self) -> dict[str, bool]:
        """Return the list of conditions."""
        return self._conditions

    @conditions.setter
    def conditions(self, conditions: dict[str, bool]) -> None:
        self._conditions = conditions

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
        self._conditions_btn.setEnabled(text != "None")
        if text == "None":
            return

        plot_multi_well_data(self, text, self._plate_viewer._pv_analysis_path)

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

    def _show_conditions_menu(self) -> None:
        """Show a context menu with condition checkboxes."""
        # Create the persistent context menu
        menu = _PersistentMenu(self)

        for condition, state in self._conditions.items():
            action = QAction(condition, self)
            action.setCheckable(True)
            action.setChecked(state)
            action.triggered.connect(
                lambda checked, text=condition: self._on_condition_toggled(
                    checked, text
                )
            )

            menu.addAction(action)

        # Show the menu at the button position
        button_pos = self._conditions_btn.mapToGlobal(
            self._conditions_btn.rect().bottomLeft()
        )
        menu.exec(button_pos)

    def _on_condition_toggled(self, checked: bool, condition: str) -> None:
        """Handle when a condition checkbox is toggled."""
        self._conditions[condition] = checked
        self._on_combo_changed(self._combo.currentText())
