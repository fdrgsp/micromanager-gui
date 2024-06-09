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


class InitDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Data Source")

        self._tensorstore = BrowseWdg(
            self,
            "Tensorstore Path:",
            "The path to the tensorstore.zarr.",
        )
        self._segmentation = BrowseWdg(
            self,
            "Segmentation Path:",
            "The path to the segmentation images. The images should be tif files and "
            "their name should end with _on where n is the position number "
            "(e.g. C3_0000_p0.tif, C3_0001_p1.tif).",
        )
        self._tensorstore._label.setFixedWidth(
            self._segmentation._label.minimumSizeHint().width()
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
        layout.addWidget(self._tensorstore, 0, 0)
        layout.addWidget(self._segmentation, 1, 0)
        layout.addWidget(self.buttonBox, 2, 0, 1, 2)

    def value(self) -> tuple[str, str]:
        return self._tensorstore.value(), self._segmentation.value()


class BrowseWdg(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        tooltip: str = "",
    ) -> None:
        super().__init__(parent)

        self._label_text = label

        self._label = QLabel(self._label_text)
        self._label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._label.setToolTip(tooltip)

        self._path = QLineEdit()
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

    def _on_browse(self) -> None:
        if path := QFileDialog.getExistingDirectory(
            self, f"Select the {self._label_text}."
        ):
            self._path.setText(path)


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])

    dialog = InitDialog()
    dialog.exec()

    print(dialog.value())
