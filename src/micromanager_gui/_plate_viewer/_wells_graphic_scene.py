from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QTransform
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)

if TYPE_CHECKING:
    from pymmcore_widgets.hcs._graphics_items import Well, _WellGraphicsItem

GREEN = "#00FF00"  # "#00C600"
SELECTED_COLOR = QBrush(QColor(GREEN))
UNSELECTED_COLOR = QBrush(Qt.GlobalColor.lightGray)
UNSELECTABLE_COLOR = QBrush(Qt.GlobalColor.darkGray)


class _WellsGraphicsScene(QGraphicsScene):
    """Custom QGraphicsScene to control the well selection.

    To get the list of selected well info, use the `value` method
    that returns a tuple (name, row, column) of the selected well.
    """

    selectedWellChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.well: _WellGraphicsItem | None = None

        # list of wells to exclude from selection. used to avoid selecting wells that
        # have not been imaged.
        self._exclude_wells: list[Well] = []

    @property
    def exclude_wells(self) -> list[Well]:
        return self._exclude_wells

    @exclude_wells.setter
    def exclude_wells(self, wells: list[Well]) -> None:
        self._exclude_wells = wells

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # origin point of the SCREEN
        self.origin_point = event.screenPos()
        # origin point of the SCENE
        self.scene_origin_point = event.scenePos()
        # clear the selection of all items
        self._clear_selection()
        # if there is an item where the mouse is pressed, select it. clear the selection
        # of all other items.
        if well := self.itemAt(self.scene_origin_point, QTransform()):
            self.well = cast("_WellGraphicsItem", well)
            if self.well.value() in self._exclude_wells:
                self.well.brush = UNSELECTABLE_COLOR
                self.well.setSelected(False)
            else:
                self.well.brush = SELECTED_COLOR
                self.well.setSelected(True)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Emit the valueChanged signal when the mouse is released."""
        if self.well is None:
            value = None
        elif self.well.value() in self._exclude_wells:
            value = None
        else:
            value = self.well.value()
        self.selectedWellChanged.emit(value)

    def _clear_selection(self) -> None:
        """Clear the selection of all wells."""
        self.well = None
        for item in self.items():
            item = cast("_WellGraphicsItem", item)
            if item.value() in self._exclude_wells:
                item.brush = UNSELECTABLE_COLOR
            else:
                item.brush = UNSELECTED_COLOR
            item.setSelected(False)

    def setValue(self, value: Well) -> None:
        """Select the wells listed in `value`."""
        self._clear_selection()

        if value in self._exclude_wells:
            return

        for item in self.items():
            item = cast("_WellGraphicsItem", item)
            if item.value() == value:
                item.brush = SELECTED_COLOR
                item.setSelected(True)
                self.well = item
                break

        self.selectedWellChanged.emit(self.value())

    def value(self) -> Well | None:
        """Return (name, row, column) of the selected well."""
        return self.well.value() if self.well is not None else None
