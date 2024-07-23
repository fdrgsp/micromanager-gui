from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from fonticon_mdi6 import MDI6
from pymmcore_widgets.hcs._plate_model import Plate
from pymmcore_widgets.hcs._util import _ResizingGraphicsView, draw_plate
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QIcon, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

from ._plate_map_graphic_scene import _PlateMapScene
from ._util import GREEN, RED

if TYPE_CHECKING:
    from pymmcore_widgets.hcs._graphics_items import _WellGraphicsItem

ALIGN_LEFT = "QPushButton { text-align: left;}"
UNSELECTED_COLOR = QBrush(Qt.GlobalColor.lightGray)
PEN = QPen(Qt.GlobalColor.black)
PEN.setWidth(3)
OPACITY = 0.7
DATA_KEY = 0


plate = Plate(
    id="standard 96 wp",
    circular=True,
    rows=8,
    columns=12,
    well_spacing_x=9.0,
    well_spacing_y=9.0,
    well_size_x=6.4,
    well_size_y=6.4,
)


class _ConditionWidget(QWidget):
    valueChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._condition_lineedit = QLineEdit()
        self._condition_lineedit.setPlaceholderText("condition name")
        self._condition_lineedit.editingFinished.connect(self._on_value_changed)

        self._color_combo = QComboBox()
        self._color_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        for color_name in QColor.colorNames():
            color_icon = QIcon(icon(MDI6.square, color=color_name))
            self._color_combo.addItem(color_icon, color_name)
        self._color_combo.currentIndexChanged.connect(self._on_value_changed)

        self._assign_btn = QPushButton("Assign")
        self._assign_btn.clicked.connect(self._on_value_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(QLabel("Condition:"), 0)
        layout.addWidget(self._condition_lineedit, 1)
        layout.addWidget(QLabel("Color:"), 0)
        layout.addWidget(self._color_combo, 0)
        layout.addWidget(self._assign_btn, 0)

    def value(self) -> tuple[str, str]:
        return self._condition_lineedit.text(), self._color_combo.currentText()

    def setValue(self, value: tuple[str, str]) -> None:
        self._condition_lineedit.setText(value[0])
        self._color_combo.setCurrentText(value[1])

    def _on_value_changed(self) -> None:
        self.valueChanged.emit(self.value())


class _ConditionTable(QGroupBox):
    valueChanged = Signal(object)
    row_deleted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._table = QTableWidget()
        # self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["Conditions"])
        self._table.horizontalHeader().setStretchLastSection(True)
        vh = self._table.verticalHeader()
        vh.setSectionResizeMode(vh.ResizeMode.ResizeToContents)
        vh.setVisible(False)

        self._add_btn = QPushButton("Add Condition")
        self._add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._add_btn.setIcon(icon(MDI6.plus_thick, color=GREEN))
        self._add_btn.setStyleSheet(ALIGN_LEFT)
        self._add_btn.clicked.connect(self._add_row)
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._remove_btn.setIcon(icon(MDI6.close_box_outline, color=RED))
        self._remove_btn.setStyleSheet(ALIGN_LEFT)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._remove_all_btn = QPushButton("Remove All")
        self._remove_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._remove_all_btn.setIcon(icon(MDI6.close_box_multiple_outline, color=RED))
        self._remove_all_btn.setStyleSheet(ALIGN_LEFT)
        self._remove_all_btn.clicked.connect(self._remove_all)

        btns_layout = QHBoxLayout()
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.addWidget(self._add_btn)
        btns_layout.addWidget(self._remove_btn)
        btns_layout.addWidget(self._remove_all_btn)
        btns_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        layout.addLayout(btns_layout)
        layout.addWidget(self._table)

    def value(self) -> list[tuple[str, str]]:
        """Return the list of conditions and their colors."""
        return [
            self._table.cellWidget(i, 0).value() for i in range(self._table.rowCount())
        ]

    def setValue(
        self, value: list[tuple[str, str]], block_signal: bool = False
    ) -> None:
        """Set the value of the widget."""
        self._clear_table()
        for row, (condition, color) in enumerate(value):
            self._add_row()
            wdg = cast(_ConditionWidget, self._table.cellWidget(row, 0))
            wdg.setValue((condition, color))

    def _clear_table(self) -> None:
        self._table.setRowCount(0)
        self._table.clearContents()

    def _add_row(self) -> None:
        # check the other colors and make sure the color is unique
        current_colors = [
            self._table.cellWidget(i, 0).value()[1]
            for i in range(self._table.rowCount() - 1)
        ]
        new_color = QColor.colorNames()[0]
        while True:
            idx = np.random.randint(0, len(QColor.colorNames()))
            new_color = QColor.colorNames()[idx]
            if new_color not in current_colors:
                break
        self._table.insertRow(self._table.rowCount())
        wdg = _ConditionWidget()
        wdg.setValue(("", new_color))
        wdg.valueChanged.connect(self._on_value_changed)
        self._table.setCellWidget(self._table.rowCount() - 1, 0, wdg)

    def _remove_selected(self) -> None:
        if selected := [i.row() for i in self._table.selectedIndexes()]:
            for sel in reversed(selected):
                value = self._table.cellWidget(sel, 0).value()
                self._table.removeRow(sel)
                self.row_deleted.emit(value)

    def _remove_all(self) -> None:
        for value in self.value():
            self.row_deleted.emit(value)
        self._clear_table()

    def _on_value_changed(self, value: tuple[str, str]) -> None:
        self._check_for_errors()
        self.valueChanged.emit(value)

    def _check_for_errors(self) -> None:
        table_values = self.value()

        # TODO: show a message box

        condition_names = [value[0] for value in table_values]
        if len(set(condition_names)) != len(condition_names):
            raise ValueError("Condition names must be unique.")

        color_names = [value[1] for value in table_values]
        if len(set(color_names)) != len(color_names):
            raise ValueError("Color names must be unique.")


class PlateMapData(NamedTuple):
    name: str
    row: int
    column: int
    condition: str
    color: str


class PlateMapWidget(QWidget):
    """A widget to create a plate map."""

    def __init__(
        self,
        parent: QWidget | None = None,
        title: str = "",
        plate: Plate | None = None,
    ) -> None:
        super().__init__(parent)
        self.cond_list: dict | None = None

        self.scene = _PlateMapScene()
        self.view = _ResizingGraphicsView(self.scene)
        self.view.setStyleSheet("background:grey; border-radius: 5px;")

        self._clear_selection_btn = QPushButton("Clear Selection")
        self._clear_selection_btn.clicked.connect(self.scene._clear_selection)
        self._save_map_btn = QPushButton("Save Plate Map")
        self._save_map_btn.clicked.connect(self._save_plate_map)
        self._load_map_btn = QPushButton("Load Plate Map")
        self._load_map_btn.clicked.connect(self._load_plate_map)

        btns_layout = QHBoxLayout()
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.addWidget(self._clear_selection_btn)
        btns_layout.addStretch()
        btns_layout.addWidget(self._save_map_btn)
        btns_layout.addWidget(self._load_map_btn)

        top_wdg = QGroupBox()
        top_layout = QVBoxLayout(top_wdg)
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(5)
        top_layout.addLayout(btns_layout)
        top_layout.addWidget(self.view)

        self.list = _ConditionTable()
        self.list.valueChanged.connect(self._assign_condition)
        self.list.row_deleted.connect(self._remove_condition)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        if title:
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 20pt;")
            layout.addWidget(title_label)
        layout.addWidget(top_wdg)
        layout.addWidget(self.list)

        if plate is not None:
            self.setPlate(plate)

    def setPlate(self, plate: Plate | dict) -> None:
        if isinstance(plate, dict):
            plate = Plate(**plate)
        draw_plate(self.view, self.scene, plate, UNSELECTED_COLOR, PEN, OPACITY)

    def value(self) -> list[PlateMapData]:
        """Return the list of classified wells and their data.

        Returns a tuple containing the (well_name, row and column) and the data assigned
        to it, (condition_name, color_name).
        """
        selected = []
        for item in reversed(self.scene.items()):
            item = cast("_WellGraphicsItem", item)
            if item.brush != UNSELECTED_COLOR:
                well = item.value()
                condition_name, color_name = item.data(DATA_KEY)
                selected.append(
                    PlateMapData(
                        name=well.name,
                        row=well.row,
                        column=well.column,
                        condition=condition_name,
                        color=color_name,
                    )
                )
        return selected

    def setValue(self, value: list[PlateMapData] | list[str] | Path | str) -> None:
        """Set the value of the widget."""
        # unset all the wells and reset the items data
        for item in self.scene.items():
            item = cast("_WellGraphicsItem", item)
            item.brush = UNSELECTED_COLOR
            item.setData(DATA_KEY, None)

        if isinstance(value, (Path, str)):
            with open(value) as pmap:
                data = json.load(pmap)
            value = cast(list, data)

        add_to_conditions_list = set()
        for data in value:
            # convert the data to a PlateMapData object if it is a list of strings
            if not isinstance(data, PlateMapData):
                data = PlateMapData(*data)
            # store the data in a list to update the condition table
            add_to_conditions_list.add((data.condition, data.color))
            # update the color and data of the selected wells
            for item in self.scene.items():
                item = cast("_WellGraphicsItem", item)
                if item.value().name == data.name:
                    item.brush = QBrush(QColor(data.color))
                    item.setData(DATA_KEY, (data.condition, data.color))
            # update the condition table
            self.list.setValue(list(add_to_conditions_list))

    def clear(self) -> None:
        """Clear the plate map."""
        for item in self.scene.items():
            item = cast("_WellGraphicsItem", item)
            item.brush = UNSELECTED_COLOR
            item.setData(DATA_KEY, None)
        self.list.setValue([])

    def _assign_condition(self, value: tuple[str, str]) -> None:
        condition_name, color_name = value

        # update the color or data of already classified wells
        for item in self.scene.items():
            item = cast("_WellGraphicsItem", item)
            if item.data(DATA_KEY) is not None:
                cond, col = item.data(DATA_KEY)
                # if the condition is assigned but the color is different, change color
                if cond == condition_name and col != color_name:
                    brush = QBrush(QColor(color_name))
                    item.brush = brush
                    item.setData(DATA_KEY, (condition_name, color_name))
                # if the color is assigned but the condition is different, symply
                # update the item data
                elif cond != condition_name and col == color_name:
                    item.setData(DATA_KEY, (condition_name, color_name))

        # assign the color and the data to the selected wells
        brush = QBrush(QColor(color_name))
        for item in self.scene.selectedItems():
            item = cast("_WellGraphicsItem", item)
            item.brush = brush
            item.setData(DATA_KEY, (condition_name, color_name))
            item.setSelected(False)

    def _remove_condition(self, value: tuple[str, str]) -> None:
        condition_name, color_name = value
        for item in self.scene.items():
            item = cast("_WellGraphicsItem", item)
            if item.data(DATA_KEY) is not None:
                cond, col = item.data(DATA_KEY)
                if cond == condition_name and col == color_name:
                    item.brush = UNSELECTED_COLOR
                    item.setData(DATA_KEY, None)
                    item.setSelected(False)

    def _save_plate_map(self) -> None:
        # if no items in the scene, return
        if not self.scene.items():
            return

        (dir_file, _) = QFileDialog.getSaveFileName(
            self, "Saving directory and filename.", "", "json(*.json)"
        )
        if not dir_file:
            return
        with open(dir_file, "w") as pmap:
            json.dump(self.value(), pmap)

        if not self.cond_list:
            self.cond_list = self._get_cond_list()

    def _load_plate_map(self) -> None:
        (filename, _) = QFileDialog.getOpenFileName(
            self, "Open plate map json file.", "", "json(*.json)"
        )
        if not filename:
            return
        with open(filename) as pmap:
            data = json.load(pmap)
        self.setValue(data)

        if not self.cond_list:
            self.cond_list = self._get_cond_list()

    # NOTE: should row and column be included?
    def _get_cond_list(self) -> dict[str, dict[str]]:
        """Get a list of condtions."""
        wells = self.value()
        cond_list = {}

        if len(wells) > 0:
            for well in wells:
                cond = well.condition
                if not cond_list.get(cond):
                    cond_list[cond] = {'color': well.color,
                                       'name': []}
                cond_list[cond]['name'].append(well.name)

        return cond_list
