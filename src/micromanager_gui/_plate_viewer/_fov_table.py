from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from qtpy.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

if TYPE_CHECKING:
    import useq

ROLE = QTableWidgetItem.ItemType.UserType + 1


class WellInfo(NamedTuple):
    pos_idx: int
    fov: useq.Position


class _FOVTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(["Fields of View"])
        hh = self.horizontalHeader()
        hh.setStretchLastSection(True)
        # set size to the header size
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        # self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def add_position(self, position: WellInfo) -> None:
        """Add a position to the table.

        Each WellInfo object (it contains the position index and the position
        object) is stored in the item's data.
        """
        row = self.rowCount()
        self.insertRow(row)
        item = QTableWidgetItem(f"{position.fov.name} (position {position.pos_idx})")
        item.setData(ROLE, position)
        self.setItem(row, 0, item)

    def clear(self) -> None:
        """Clear the table."""
        self.clearContents()
        self.setRowCount(0)

    def value(self) -> WellInfo | None:
        """Return the selected position."""
        items = self.selectedItems()
        return items[0].data(ROLE) if items else None
