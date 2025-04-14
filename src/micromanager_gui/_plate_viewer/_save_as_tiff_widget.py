from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._util import _BrowseWidget, parse_lineedit_text, show_error_dialog

if TYPE_CHECKING:
    from ._plate_viewer import PlateViewer


class _SaveAsTiff(QDialog):
    def __init__(self, parent: PlateViewer | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save As Tiff")

        # position selection widget
        pos_label = QLabel("Positions:")
        pos_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pos_line_edit = QLineEdit()
        self._pos_line_edit.setPlaceholderText("0-10 or 0, 1, 2")
        tooltip = (
            "Select the Positions to save as .tiff."
            "\nLeave blank to segment all Positions.\n"
            "You can input single Positions (e.g. 30, 33) a range "
            "(e.g. 1-10), or a mix of single Positions and ranges "
            "(e.g. 1-10, 30, 50-65).\n"
            "NOTE: The Positions are 0-indexed."
        )
        pos_label.setToolTip(tooltip)
        self._pos_line_edit.setToolTip(tooltip)
        pos_wdg = QWidget()
        pos_layout = QHBoxLayout(pos_wdg)
        pos_layout.addWidget(pos_label)
        pos_layout.addWidget(self._pos_line_edit)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setSpacing(5)

        # save folder selection widget
        self._browse_widget = _BrowseWidget(
            self, "Save Path", "", "Select the path to save the .tiff files."
        )

        # ok, cancel buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(pos_wdg)
        main_layout.addWidget(self._browse_widget)
        main_layout.addWidget(self._button_box)

    def accept(self) -> Any:
        """Override QDialog accept method."""
        path, _ = self.value()
        if not path:
            show_error_dialog(self, "Please select a path to save the .tiff files.")
            return
        return super().accept()

    def value(self) -> tuple[str, list[int]]:
        """Return the selected path and positions list."""
        positions = parse_lineedit_text(self._pos_line_edit.text())
        return self._browse_widget.value(), positions
