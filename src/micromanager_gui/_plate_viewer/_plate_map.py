from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from fonticon_mdi6 import MDI6
from pymmcore_widgets.useq_widgets._well_plate_widget import (
    DATA_COLOR,
    DATA_INDEX,
    DATA_POSITION,
    DATA_SELECTED,
    WellPlateView,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon
from qtpy.QtWidgets import (
    QAbstractGraphicsShapeItem,
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

from ._util import GREEN, RED

if TYPE_CHECKING:
    from collections.abc import Iterable

    import useq

ALIGN_LEFT = "QPushButton { text-align: left;}"
DATA_CONDITION = 5


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
        self._color_combo.setMaxVisibleItems(10)

        self._color_combo.currentIndexChanged.connect(self._on_value_changed)

        self._assign_btn = QPushButton("Assign")
        self._assign_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
    """Data structure for the plate map."""

    name: str  # well name
    row_col: tuple[int, int]  # row, column
    condition: tuple[str, str]  # condition name, color name


class PlateMapDataOld(NamedTuple):
    """Old data structure for the plate map."""

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
        plate: useq.WellPlate | None = None,
    ) -> None:
        super().__init__(parent)

        self._plate_view = WellPlateView()
        self._plate_view._change_selection = self._change_selection

        self._clear_selection_btn = QPushButton("Clear Selection")
        self._clear_selection_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._clear_selection_btn.setToolTip("Clear the selection of the wells.")
        self._clear_selection_btn.clicked.connect(self._plate_view.clearSelection)
        self._clear_condition_btn = QPushButton("Clear Condition")
        self._clear_condition_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._clear_condition_btn.setToolTip(
            "Clear the condition of the selected wells."
        )
        self._clear_condition_btn.clicked.connect(self.clear_condition)
        self._clear_all_btn = QPushButton("Clear All")
        self._clear_all_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._clear_all_btn.setToolTip(
            "Clear all the selection of the wells and all the conditions."
        )
        self._clear_all_btn.clicked.connect(self.clear)
        self._save_map_btn = QPushButton("Save Plate Map")
        self._save_map_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save_map_btn.clicked.connect(self.save_plate_map)
        self._load_map_btn = QPushButton("Load Plate Map")
        self._load_map_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._load_map_btn.clicked.connect(self.load_plate_map)

        btns_layout = QHBoxLayout()
        btns_layout.setContentsMargins(0, 0, 0, 0)
        btns_layout.addWidget(self._clear_selection_btn)
        btns_layout.addWidget(self._clear_condition_btn)
        btns_layout.addWidget(self._clear_all_btn)
        btns_layout.addStretch()
        btns_layout.addWidget(self._save_map_btn)
        btns_layout.addWidget(self._load_map_btn)

        top_wdg = QGroupBox()
        top_layout = QVBoxLayout(top_wdg)
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(5)
        top_layout.addLayout(btns_layout)
        top_layout.addWidget(self._plate_view)

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

    def setPlate(self, plate: useq.WellPlate) -> None:
        self._plate_view.drawPlate(plate)

    def clear(self) -> None:
        """Clear all the wells and the conditions."""
        # clear the selection of the plate view
        self._plate_view.clearSelection()
        # clear the color and data of the condition wells
        wells: dict[tuple[int, int], QAbstractGraphicsShapeItem] = (
            self._plate_view._well_items
        )
        for (r, c), well in wells.items():
            if well.data(DATA_COLOR):
                self._plate_view.setWellColor(r, c, None)
                well.setData(DATA_COLOR, None)
                well.setData(DATA_CONDITION, None)
                well.setData(DATA_SELECTED, False)

    def clear_condition(self) -> None:
        """Clear the condition of the selected wells."""
        wells: tuple[tuple[int, int]] = self._plate_view.selectedIndices()
        for well in wells:
            r, c = well
            self._plate_view.setWellColor(r, c, None)
            self._plate_view._well_items[well].setData(DATA_COLOR, None)
            self._plate_view._well_items[well].setData(DATA_CONDITION, None)
        self._plate_view.clearSelection()

    def value(self) -> list[PlateMapData]:
        """Return the list of classified wells and their data.

        Returns a tuple containing the (well_name, row and column) and the data assigned
        to it, (condition_name, color_name).
        """

        def convert_np_int64(value: Any) -> Any:
            """Convert np.int64 to int."""
            return int(value) if isinstance(value, np.int64) else value

        wells: dict[tuple[int, int], QAbstractGraphicsShapeItem] = (
            self._plate_view._well_items
        )
        return [
            PlateMapData(
                well.data(DATA_POSITION).name,
                tuple(map(convert_np_int64, well.data(DATA_INDEX))),
                well.data(DATA_CONDITION),
            )
            for well in wells.values()
            if well.data(DATA_COLOR)
        ]

    def setValue(
        self, value: list[PlateMapData | PlateMapDataOld] | list | Path | str
    ) -> None:
        """Set the value of the widget."""
        # clear the selection and the conditions of the plate
        self._plate_view.clearSelection()
        self.clear()

        try:
            if isinstance(value, (Path, str)):
                with open(value) as pmap:
                    data = json.load(pmap)
                value = cast(list, data)

            add_to_conditions_list = set()
            for data in value:
                # convert the data to a PlateMapData object if it is a list of strings
                if not isinstance(data, PlateMapData):
                    # convert the old data to the new data
                    if isinstance(data, PlateMapDataOld):
                        data = PlateMapData(
                            data.name,
                            (data.row, data.column),
                            (data.condition, data.color),
                        )
                    # convert old list of strings to new PlateMapData
                    elif len(data) == 5:  # from PlateMapDataOld
                        name, r, c, condition, color = data
                        data = PlateMapData(name, (r, c), (condition, color))
                    else:
                        data = PlateMapData(*tuple(data))
                # store the data in a list to update the condition table
                add_to_conditions_list.add(tuple(data.condition))
                # update the color and data of the wells with the assigned conditions
                wells: dict[tuple[int, int], QAbstractGraphicsShapeItem] = (
                    self._plate_view._well_items
                )
                for well in wells.values():
                    if well.data(DATA_INDEX) == tuple(data.row_col):
                        r, c = well.data(DATA_INDEX)
                        _, color_name = data.condition
                        self._plate_view.setWellColor(r, c, color_name)
                        well.setData(DATA_CONDITION, tuple(data.condition))
                    # update the condition table
            self.list.setValue(list(add_to_conditions_list))
        except Exception as e:
            warnings.warn(f"Error loading the plate map: {e}", stacklevel=2)
            return

    def save_plate_map(self) -> None:
        """Save the plate map to a json file."""
        (dir_file, _) = QFileDialog.getSaveFileName(
            self, "Saving directory and filename.", "", "json(*.json)"
        )
        if not dir_file:
            return

        with open(dir_file, "w") as pmap:
            json.dump(self.value(), pmap, indent=2)

    def load_plate_map(self) -> None:
        """Load a plate map from a json file."""
        (filename, _) = QFileDialog.getOpenFileName(
            self, "Open plate map json file.", "", "json(*.json)"
        )
        if not filename:
            return
        with open(filename) as pmap:
            data = json.load(pmap)
        self.setValue(data)

    # override the super method to change the color of the selected wells
    # so it does not use the color stored in the data and this if we select a well
    # with a condition and color assigned, the color will change to the selected color
    # when the well is selected and back to the assigned color when it is deselected.
    def _change_selection(
        self,
        select: Iterable[QAbstractGraphicsShapeItem],
        deselect: Iterable[QAbstractGraphicsShapeItem],
    ) -> None:
        """Change the selection of the wells.

        Overriding the super method to change the color of the selected wells so it
        does not use the color stored in the data and this if we select a well with
        a condition and color assigned, the color will change to the selected color
        when the well is selected and back to the assigned color when deselected.
        """
        before = self._plate_view._selected_items.copy()

        for item in select:
            color = self._plate_view._selected_color
            item.setBrush(color)
            item.setData(DATA_SELECTED, True)
        self._plate_view._selected_items.update(select)

        for item in deselect:
            if item.data(DATA_SELECTED):
                color = item.data(DATA_COLOR) or self._plate_view._unselected_color
                item.setBrush(color)
                item.setData(DATA_SELECTED, False)
        self._plate_view._selected_items.difference_update(deselect)

        if before != self._plate_view._selected_items:
            self._plate_view.selectionChanged.emit()

    def _assign_condition(self, value: tuple[str, str]) -> None:
        condition_name, color_name = value
        wells: dict[tuple[int, int], QAbstractGraphicsShapeItem] = (
            self._plate_view._well_items
        )
        for well_key in wells:
            if wells[well_key].data(DATA_SELECTED):
                # remove the well from the selected items
                self._plate_view._selected_items.difference_update([wells[well_key]])
                # update the data of the well
                wells[well_key].setData(DATA_SELECTED, False)
                wells[well_key].setData(DATA_CONDITION, (condition_name, color_name))
                # setWellColor will also add the DATA_COLOR to the well item
                r, c = wells[well_key].data(DATA_INDEX)
                self._plate_view.setWellColor(r, c, color_name)

            if wells[well_key].data(DATA_CONDITION) is not None:
                cond, col = wells[well_key].data(DATA_CONDITION)
                # if the condition is assigned but the color is different, change color
                if cond == condition_name and col != color_name:
                    r, c = wells[well_key].data(DATA_INDEX)
                    self._plate_view.setWellColor(r, c, color_name)
                    wells[well_key].setData(
                        DATA_CONDITION, (condition_name, color_name)
                    )
                # if the color is assigned but the condition is different, symply
                # update the item data
                elif cond != condition_name and col == color_name:
                    wells[well_key].setData(
                        DATA_CONDITION, (condition_name, color_name)
                    )

    def _remove_condition(self, value: tuple[str, str]) -> None:
        condition_name, color_name = value
        wells: dict[tuple[int, int], QAbstractGraphicsShapeItem] = (
            self._plate_view._well_items
        )
        for well_key in wells:
            if wells[well_key].data(DATA_CONDITION) is not None:
                cond, col = wells[well_key].data(DATA_CONDITION)
                if cond == condition_name and col == color_name:
                    r, c = wells[well_key].data(DATA_INDEX)
                    self._plate_view.setWellColor(r, c, None)
                    wells[well_key].setData(DATA_CONDITION, None)
                    wells[well_key].setData(DATA_SELECTED, False)
