from __future__ import annotations

from typing import TypedDict, cast

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from micromanager_gui._plate_viewer._segmentation import _SelectModelPath

CUSTOM = "custom"
CYTO3 = "cyto3"
MODELS = [CYTO3, CUSTOM]
CUSTOM_MODEL_PATH = "cellpose_models/cp_img8_epoch7000_py"
FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


def _sort_plate(item: str) -> tuple[int, int | str]:
    """Sort well plate keys by number first, then by string."""
    parts = item.split("-")
    return (0, int(parts[0])) if parts[0].isdigit() else (1, item)  # type: ignore


class RealTimeCellposeSegmentationWidget(QGroupBox):
    """Widget to enable Segmentation and Analysis while recording."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._enable_segmentation_checkbox = QCheckBox("Enable Segmentation")
        self._enable_segmentation_checkbox.setSizePolicy(*FIXED)
        self._enable_segmentation_checkbox.toggled.connect(
            self._on_enable_segmentation_toggled
        )
        self._segmentation_settings_btn = QPushButton("Segmentation Settings...")
        self._segmentation_settings_btn.setSizePolicy(*FIXED)
        self._segmentation_settings_btn.setEnabled(False)
        self._segmentation_settings_btn.clicked.connect(
            self._show_segmentation_settings
        )

        self._segmentation_dialog = RealTimeSegmentationDialog(self)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        main_layout.addWidget(self._enable_segmentation_checkbox)
        main_layout.addWidget(self._segmentation_settings_btn)
        main_layout.addStretch()

    def _on_enable_segmentation_toggled(self, checked: bool) -> None:
        """Enable/disable the analysis settings button based on the checkbox state."""
        self._segmentation_settings_btn.setEnabled(checked)
        if not checked:
            self._segmentation_dialog.hide()

    def _show_segmentation_settings(self) -> None:
        """Show the segmentation settings dialog."""
        if self._segmentation_dialog.isVisible():
            self._segmentation_dialog.raise_()
        else:
            h = self._segmentation_dialog.sizeHint().height()
            self._segmentation_dialog.resize(400, h)
            self._segmentation_dialog.show()

    def isSegmentationEnabled(self) -> bool:
        """Return True if the checkbox is checked."""
        return cast(bool, self._enable_segmentation_checkbox.isChecked())

    def value(self) -> SegmentationParameters | None:
        """Return the current value of the widget."""
        if self.isSegmentationEnabled():
            return self._segmentation_dialog.value()
        return None


class SegmentationParameters(TypedDict):
    """A class to store the values of RealTimeAnalysisWidget."""

    model_type: str  # cellpose model type
    model_path: str  # path to the custom model


class RealTimeSegmentationDialog(QDialog):
    """A Widget to set the segmentation parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Segmentation Parameters")

        # Cellpose -----------------------------------------------------------------
        self._browse_custom_model = _SelectModelPath(self, label="Path")
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        self._models_combo = QComboBox()
        self._models_combo.addItems(MODELS)
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)

        model_lbl = QLabel("Model:")
        model_lbl.setSizePolicy(*FIXED)

        cellpose_wdg = QGroupBox(self, title="Cellpose")
        cellpose_wdg.setCheckable(False)
        cellpose_wdg_layout = QGridLayout(cellpose_wdg)
        cellpose_wdg_layout.setContentsMargins(10, 10, 10, 10)
        cellpose_wdg_layout.setSpacing(5)
        cellpose_wdg_layout.addWidget(model_lbl, 0, 0)
        cellpose_wdg_layout.addWidget(self._models_combo, 0, 1)
        cellpose_wdg_layout.addWidget(self._browse_custom_model, 1, 0, 1, 2)

        # Styling ------------------------------------------------------------------
        fixed_width = model_lbl.sizeHint().width()
        self._browse_custom_model._label.setFixedWidth(fixed_width)

        # Layout -------------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(cellpose_wdg)

    def _on_model_combo_changed(self, model: str) -> None:
        """Show the custom model path if the model is 'custom'."""
        (
            self._browse_custom_model.show()
            if model == "custom"
            else self._browse_custom_model.hide()
        )

    def value(self) -> SegmentationParameters:
        """Return the model type and path."""
        model_type = self._models_combo.currentText()
        model_path = self._browse_custom_model.value() if model_type == CUSTOM else ""
        return {"model_type": model_type, "model_path": model_path}
