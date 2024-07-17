from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fonticon_mdi6 import MDI6
from pymmcore_widgets.hcs._plate_model import Plate
from pymmcore_widgets.hcs._util import _ResizingGraphicsView, draw_plate
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QIcon, QPen
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

from ._plate_map_graphic_scene import _PlateMapScene

if TYPE_CHECKING:
    from pymmcore_widgets.hcs._graphics_items import Well, _WellGraphicsItem

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
        layout.setContentsMargins(5, 0, 5, 0)
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

        self._toolbar = QToolBar(self)
        self._toolbar.setFloatable(False)
        self._toolbar.setIconSize(QSize(22, 22))
        self._toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self.act_add_row = QAction(
            icon(MDI6.plus_thick, color="#00FF00"), "Add new condition", self
        )
        self.act_add_row.triggered.connect(self._add_row)
        self.act_remove_row = QAction(
            icon(MDI6.close_box_outline, color="#C33"),
            "Remove selected condition",
            self,
        )
        self.act_remove_row.triggered.connect(self._remove_selected)
        self.act_clear = QAction(
            icon(MDI6.close_box_multiple_outline, color="#C33"),
            "Remove all conditions",
            self,
        )
        self.act_clear.triggered.connect(self._remove_all)

        self._toolbar.addAction(self.act_add_row)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self.act_remove_row)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self.act_clear)

        self._table = QTableWidget()
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["Conditions"])
        self._table.horizontalHeader().setStretchLastSection(True)
        vh = self._table.verticalHeader()
        vh.setSectionResizeMode(vh.ResizeMode.ResizeToContents)
        vh.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._table)

    def value(self) -> list[tuple[str, str]]:
        """Return the list of conditions and their colors."""
        return [
            self._table.cellWidget(i, 0).value() for i in range(self._table.rowCount())
        ]

    def setValue(self, value: list[tuple[str, str]]) -> None:
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
        self._table.insertRow(self._table.rowCount())
        wdg = _ConditionWidget()
        wdg.valueChanged.connect(self._on_value_changed)
        self._table.setCellWidget(self._table.rowCount() - 1, 0, wdg)

    def _remove_selected(self) -> None:
        if selected := [i.row() for i in self._table.selectedIndexes()]:
            value = self._table.cellWidget(selected[0], 0).value()
            self._table.removeRow(selected[0])
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


class PlateMapWidget(QWidget):
    """A widget to create a plate map."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.scene = _PlateMapScene()
        self.view = _ResizingGraphicsView(self.scene)
        self.view.setStyleSheet("background:grey; border-radius: 5px;")

        self.list = _ConditionTable()
        self.list.valueChanged.connect(self._assign_condition)
        self.list.row_deleted.connect(self._remove_condition)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addWidget(self.list)

        draw_plate(self.view, self.scene, plate, UNSELECTED_COLOR, PEN, OPACITY)

    def value(self) -> list[tuple[Well, tuple[str, str]]]:
        """Return the list of classified wells and their data.

        Returns a tuple containing the (well_name, row and column) and the data assigned
        to it, (condition_name, color_name).
        """
        selected = []
        for item in self.scene.items():
            item = cast("_WellGraphicsItem", item)
            if item.brush != UNSELECTED_COLOR:
                selected.append((item.value(), item.data(DATA_KEY)))
        return list(reversed(selected))

    def setValue(self, value: list[tuple[Well, tuple[str, str]]]) -> None:
        """Set the value of the widget."""
        # unset all the wells and reset the items data
        for item in self.scene.items():
            item = cast("_WellGraphicsItem", item)
            item.brush = UNSELECTED_COLOR
            item.setData(DATA_KEY, None)

        for well, data in value:
            condition_name, color_name = data
            # update the condition table
            self.list.setValue([(condition_name, color_name)])
            # update the color and data of the selected wells
            for item in self.scene.items():
                item = cast("_WellGraphicsItem", item)
                if item.value() == well:
                    item.brush = QBrush(QColor(color_name))
                    item.setData(DATA_KEY, data)

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
