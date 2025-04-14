from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QWidget,
)

from ._util import _BrowseWidget


class _InitDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        datastore_path: str | None = None,
        labels_path: str | None = None,
        analysis_path: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Data Source")

        # datastore_path
        self._browse_datastrore = _BrowseWidget(
            self,
            "Datastore Path",
            datastore_path,
            "The path to the zarr datastore.",
        )

        # labels_path with labels images
        self._browse_labels = _BrowseWidget(
            self,
            "Segmentation Path",
            labels_path,
            "The path to the labels images. The images should be tif files and "
            "their name should end with _on where n is the position number "
            "(e.g. C3_0000_p0.tif, C3_0001_p1.tif).",
        )

        # analysis_path with json files
        self._browse_analysis = _BrowseWidget(
            self,
            "Analysis Path",
            analysis_path,
            "The path to the analysis of the current data. The images should be "
            "a path to a `json` file.",
            is_dir=True,
        )

        # styling
        fix_width = self._browse_labels._label.minimumSizeHint().width()
        self._browse_datastrore._label.setFixedWidth(fix_width)
        self._browse_analysis._label.setFixedWidth(fix_width)

        # Create the button box
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        # Connect the signals
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Add the button box to the layout
        layout = QGridLayout(self)
        layout.addWidget(self._browse_datastrore, 0, 0)
        layout.addWidget(self._browse_labels, 1, 0)
        layout.addWidget(self._browse_analysis, 2, 0)
        layout.addWidget(self.buttonBox, 3, 0, 1, 2)

    def value(self) -> tuple[str, str, str]:
        return (
            self._browse_datastrore.value(),
            self._browse_labels.value(),
            self._browse_analysis.value(),
        )
