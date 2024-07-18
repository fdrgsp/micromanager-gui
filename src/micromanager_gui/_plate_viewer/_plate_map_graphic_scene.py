from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from qtpy.QtCore import QRect, QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QTransform
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QRubberBand,
    QWidget,
)

if TYPE_CHECKING:
    from pymmcore_widgets.hcs._graphics_items import Well, _WellGraphicsItem

GREEN = "#00FF00"  # "#00C600"
SELECTED_COLOR = QBrush(QColor(GREEN))
UNSELECTED_COLOR = QBrush(Qt.GlobalColor.lightGray)


class _PlateMapScene(QGraphicsScene):
    """Custom QGraphicsScene to control the plate/well selection.

    To get the list of selected well info, use the `value` method
    that returns a list of snake-row-wise ordered tuples (name, row, column).
    """

    valueChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.well_pos = 0
        self.new_well_pos = 0

        self._selected_wells: list[QGraphicsItem] = []

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # origin point of the SCREEN
        self.origin_point = event.screenPos()
        # rubber band to show the selection
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle)
        # origin point of the SCENE
        self.scene_origin_point = event.scenePos()

        # get the selected items
        self._selected_wells = [item for item in self.items() if item.isSelected()]

        # set the color of the selected wells to SELECTED_COLOR if they are within the
        # selection
        for item in self._selected_wells:
            item = cast("_WellGraphicsItem", item)
            item.brush = SELECTED_COLOR

        # if there is an item where the mouse is pressed and it is selected, deselect,
        # otherwise select it.
        if well := self.itemAt(self.scene_origin_point, QTransform()):
            well = cast("_WellGraphicsItem", well)
            if well.isSelected():
                well.brush = UNSELECTED_COLOR
                well.setSelected(False)
            else:
                well.brush = SELECTED_COLOR
                well.setSelected(True)
        self.valueChanged.emit()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # update the rubber band geometry using the SCREEN origin point and the current
        self.rubber_band.setGeometry(QRect(self.origin_point, event.screenPos()))
        self.rubber_band.show()
        # get the items within the selection (within the rubber band)
        selection = self.items(QRectF(self.scene_origin_point, event.scenePos()))
        # loop through all the items in the scene and select them if they are within
        # the selection or deselect them if they are not (or if the shift key is pressed
        # while moving the movuse).
        for item in self.items():
            item = cast("_WellGraphicsItem", item)

            if item in selection:
                # if pressing shift, remove from selection
                if event.modifiers() and Qt.KeyboardModifier.ShiftModifier:
                    self._set_selected(item, False)
                else:
                    self._set_selected(item, True)
            elif item not in self._selected_wells and item.brush == SELECTED_COLOR:
                self._set_selected(item, False)
        self.valueChanged.emit()

    def _set_selected(self, item: _WellGraphicsItem, state: bool) -> None:
        """Select or deselect the item."""
        item.brush = SELECTED_COLOR if state else UNSELECTED_COLOR
        item.setSelected(state)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.rubber_band.hide()
        self.valueChanged.emit()

    def _clear_selection(self) -> None:
        """Clear the selection of all wells."""
        for item in self.items():
            item = cast("_WellGraphicsItem", item)
            if item.brush != SELECTED_COLOR:
                continue
            item.setSelected(False)
            item.brush = UNSELECTED_COLOR
        self.valueChanged.emit()

    def setValue(self, value: list[Well]) -> None:
        """Select the wells listed in `value`."""
        self._clear_selection()

        for item in self.items():
            item = cast("_WellGraphicsItem", item)
            if item.value() in value:
                self._set_selected(item, True)
        self.valueChanged.emit()

    def value(self) -> list[tuple[Well, Any]]:
        """Return the list of Well objects."""
        return [item.value() for item in reversed(self.items()) if item.isSelected()]
