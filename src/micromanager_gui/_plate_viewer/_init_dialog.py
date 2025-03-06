from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)


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

        self._datastrore = _BrowseWidget(
            self,
            "Datastore Path",
            datastore_path,
            "The path to the zarr datastore.",
        )
        self._labels = _BrowseWidget(
            self,
            "Segmentation Path",
            labels_path,
            "The path to the labels images. The images should be tif files and "
            "their name should end with _on where n is the position number "
            "(e.g. C3_0000_p0.tif, C3_0001_p1.tif).",
        )

        self._analysis = _BrowseWidget(
            self,
            "Analysis Path",
            analysis_path,
            "The path to the analysis of the current data. The images should be "
            "a path to a `json` file.",
            is_dir=True,
        )
        self._datastrore._label.setFixedWidth(
            self._labels._label.minimumSizeHint().width()
        )
        self._analysis._label.setFixedWidth(
            self._labels._label.minimumSizeHint().width()
        )

        # Create the button box
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        # Connect the signals
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Add the button box to the layout
        layout = QGridLayout(self)
        layout.addWidget(self._datastrore, 0, 0)
        layout.addWidget(self._labels, 1, 0)
        layout.addWidget(self._analysis, 2, 0)
        layout.addWidget(self.buttonBox, 3, 0, 1, 2)

    def value(self) -> tuple[str, str, str]:
        return (
            self._datastrore.value(),
            self._labels.value(),
            self._analysis.value(),
        )


class _BrowseWidget(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        path: str | None = None,
        tooltip: str = "",
        *,
        is_dir: bool = True,
    ) -> None:
        super().__init__(parent)

        self._is_dir = is_dir

        self._current_path = path or ""

        self._label_text = label

        self._label = QLabel(f"{self._label_text}:")
        self._label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._label.setToolTip(tooltip)

        self._path = QLineEdit()
        self._path.setText(self._current_path)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._on_browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._label)
        layout.addWidget(self._path)
        layout.addWidget(self._browse_btn)

    def value(self) -> str:
        return self._path.text()  # type: ignore

    def setValue(self, path: str) -> None:
        self._path.setText(path)

    def _on_browse(self) -> None:
        if self._is_dir:
            if path := QFileDialog.getExistingDirectory(
                self, f"Select the {self._label_text}.", self._current_path
            ):
                self._path.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select the {self._label_text}.",
                "",
                "JSON (*.json);; Images (*.tif *.tiff *.png *.jpg)",
            )
            if path:
                self._path.setText(path)
