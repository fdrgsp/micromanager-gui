from __future__ import annotations

from typing import TypedDict, cast

import useq
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from micromanager_gui._plate_viewer._plate_map import PlateMapData, PlateMapWidget
from micromanager_gui._plate_viewer._segmentation import _SelectModelPath
from micromanager_gui._plate_viewer._util import _BrowseWidget

CUSTOM = "custom"
CYTO3 = "cyto3"
MODELS = [CYTO3, CUSTOM]
CUSTOM_MODEL_PATH = "models/cp_img8_epoch7000_py"
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"
FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


def _sort_plate(item: str) -> tuple[int, int | str]:
    """Sort well plate keys by number first, then by string."""
    parts = item.split("-")
    return (0, int(parts[0])) if parts[0].isdigit() else (1, item)  # type: ignore


class RealTimeAnalysisWidget(QWidget):
    """Widget to enable Segmentation and Analysis while recording."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._segmentation_checkbox = QCheckBox("Enable Segmentation")
        self._segmentation_checkbox.setSizePolicy(*FIXED)
        self._segmentation_checkbox.toggled.connect(
            self._on_segmentation_checkbox_toggled
        )
        self._segmentation_settings_btn = QPushButton("Segmentation Settings...")
        self._segmentation_settings_btn.setSizePolicy(*FIXED)
        self._segmentation_settings_btn.clicked.connect(
            self._show_segmentation_settings
        )
        self.segmentation_gbox = QGroupBox(self)
        self.segmentation_gbox.setCheckable(False)
        segmentation_gbox_layout = QHBoxLayout(self.segmentation_gbox)
        segmentation_gbox_layout.setContentsMargins(10, 10, 10, 10)
        segmentation_gbox_layout.setSpacing(20)
        segmentation_gbox_layout.addWidget(self._segmentation_checkbox)
        segmentation_gbox_layout.addWidget(self._segmentation_settings_btn)
        segmentation_gbox_layout.addStretch(1)

        self._analysis_checkbox = QCheckBox("Enable Analysis")
        self._analysis_checkbox.setSizePolicy(*FIXED)
        self._analysis_settings_btn = QPushButton("Analysis Settings...")
        self._analysis_settings_btn.setSizePolicy(*FIXED)
        self._analysis_settings_btn.clicked.connect(self._show_analysis_settings)
        self.analysis_gbox = QGroupBox(self)
        self.analysis_gbox.setCheckable(False)
        analysis_gbox_layout = QHBoxLayout(self.analysis_gbox)
        analysis_gbox_layout.setContentsMargins(10, 10, 10, 10)
        analysis_gbox_layout.setSpacing(20)
        analysis_gbox_layout.addWidget(self._analysis_checkbox)
        analysis_gbox_layout.addWidget(self._analysis_settings_btn)
        self.analysis_gbox.setEnabled(False)
        analysis_gbox_layout.addStretch(1)

        self._segmentation_dialog = RealTimeSegmentationDialog(self)
        self._analysis_dialog = RealTimeAnalysisDialog(self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        main_layout.addWidget(self.segmentation_gbox)
        main_layout.addWidget(self.analysis_gbox)

        fixed_width = self._segmentation_checkbox.sizeHint().width()
        self._analysis_checkbox.setFixedWidth(fixed_width)

    def _on_segmentation_checkbox_toggled(self, checked: bool) -> None:
        """Enable/disable the analysis settings button based on the checkbox state."""
        self.analysis_gbox.setEnabled(checked)

    def _show_segmentation_settings(self) -> None:
        """Show the segmentation settings dialog."""
        if self._segmentation_dialog.isVisible():
            self._segmentation_dialog.raise_()
        else:
            self._segmentation_dialog.show()

    def _show_analysis_settings(self) -> None:
        """Show the analysis settings dialog."""
        if self._analysis_dialog.isVisible():
            self._analysis_dialog.raise_()
        else:
            self._analysis_dialog.show()

    def isSegmentationEnabled(self) -> bool:
        """Return True if the checkbox is checked."""
        return cast(bool, self._segmentation_checkbox.isChecked())

    def isAnalysisEnabled(self) -> bool:
        """Return True if the checkbox is checked."""
        return cast(
            bool, self._analysis_checkbox.isChecked() and self.analysis_gbox.isEnabled()
        )

    def value(
        self,
    ) -> tuple[SegmentationParameters | None, AnalysisParameters | None]:
        """Return the current value of the widget."""
        value: list = [None, None]
        if self.isSegmentationEnabled():
            value[0] = self._segmentation_dialog.value()
        if self.isAnalysisEnabled():
            value[1] = self._analysis_dialog.value()
        return tuple(value)


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


class _PlateMap(QGroupBox):
    """A widget to show and edit the plate map."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Plate Map")
        self.setCheckable(False)

        self._well_plate_lbl = QLabel("Well Plate:")
        self._well_plate_lbl.setSizePolicy(*FIXED)
        self._well_plate_combo = QComboBox()
        plate_names = sorted(useq.registered_well_plate_keys(), key=_sort_plate)
        self._well_plate_combo.addItems(plate_names)
        self._well_plate_combo.currentTextChanged.connect(self._on_plate_changed)

        top = QWidget(self)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)
        top_layout.addWidget(self._well_plate_lbl)
        top_layout.addWidget(self._well_plate_combo)

        self._plate_map_genotype = PlateMapWidget(self, title="Genotype Map")
        self._plate_map_treatment = PlateMapWidget(self, title="Treatment Map")

        bottom = QWidget(self)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        bottom_layout.setSpacing(5)
        bottom_layout.addWidget(self._plate_map_genotype)
        bottom_layout.addWidget(self._plate_map_treatment)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(top)
        main_layout.addWidget(bottom)

        self._on_plate_changed(self._well_plate_combo.currentText())

    def _on_plate_changed(self, text: str) -> None:
        """Set the values of the genotype and treatment maps."""
        plate = useq.WellPlate.from_str(self._well_plate_combo.currentText())
        self._plate_map_genotype.setPlate(plate)
        self._plate_map_treatment.setPlate(plate)


class AnalysisParameters(TypedDict):
    """A class to store the values of RealTimeAnalysisDialog."""

    min_peaks_height: float  # min height for the peaks detection
    experiment_type: str  # SPONTANEOUS or EVOKED
    led_power_equation: str  # linear equation to convert LED power to mW
    stimulation_mask_path: str  # path to the stimulated area mask
    genotype_map: list[PlateMapData]
    treatment_map: list[PlateMapData]


class RealTimeAnalysisDialog(QDialog):
    """A Widget to set the analysis parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analysis Parameters")

        # Experiment type ----------------------------------------------------------
        experiment_type_label = QLabel("Type:")
        experiment_type_label.setSizePolicy(*FIXED)
        self._experiment_type_combo = QComboBox()
        self._experiment_type_combo.addItems([SPONTANEOUS, EVOKED])
        self._experiment_type_combo.currentTextChanged.connect(
            self._on_activity_changed
        )

        self._stimulation_area_path = _BrowseWidget(
            self,
            label="Stimulated Area File",
            tooltip=(
                "Select the path to the image of the stimulated area.\n"
                "The image should either be a binary mask or a grayscale image where "
                "the stimulated area is brighter than the rest.\n"
                "Accepted formats: .tif, .tiff."
            ),
            is_dir=False,
        )
        self._stimulation_area_path.hide()

        self._led_power_wdg = QWidget(self)
        self._led_power_wdg.setToolTip(
            "Insert the linear equation to convert the LED power to mW (in the form of "
            "y = a * x + b). Leave it empty to use the values from the metadata."
        )
        led_lbl = QLabel("LED Power Equation:")
        led_lbl.setSizePolicy(*FIXED)
        self._led_power_equation_le = QLineEdit(self)
        self._led_power_equation_le.setText("y = 11.07 * x - 6.63")
        self._led_power_equation_le.setPlaceholderText(
            "y= a * x + b ( Leave it empty to use the values from the metadata)."
        )
        led_layout = QHBoxLayout(self._led_power_wdg)
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.setSpacing(5)
        led_layout.addWidget(led_lbl)
        led_layout.addWidget(self._led_power_equation_le)
        self._led_power_wdg.hide()

        experiment_type_wdg = QGroupBox(self, title="Experiment")
        experiment_type_wdg.setCheckable(False)
        experiment_type_wdg_layout = QGridLayout(experiment_type_wdg)
        experiment_type_wdg_layout.setContentsMargins(10, 10, 10, 10)
        experiment_type_wdg_layout.setSpacing(5)
        experiment_type_wdg_layout.addWidget(experiment_type_label, 0, 0)
        experiment_type_wdg_layout.addWidget(self._experiment_type_combo, 0, 1)
        experiment_type_wdg_layout.addWidget(self._led_power_wdg, 1, 0, 1, 2)
        experiment_type_wdg_layout.addWidget(self._stimulation_area_path, 2, 0, 1, 2)

        # peaks settings -------------------------------------------------------------
        min_peaks_lbl_wdg = QGroupBox(self, title="Find Peaks Settings")
        min_peaks_lbl_wdg.setToolTip(
            "Set the min height for the peaks (used by the scipy find_peaks method)."
        )
        min_peaks_lbl = QLabel("Min Peaks Height:")
        min_peaks_lbl.setSizePolicy(*FIXED)
        self._min_peaks_height_spin = QDoubleSpinBox(self)
        self._min_peaks_height_spin.setDecimals(4)
        self._min_peaks_height_spin.setRange(0.0, 100000.0)
        self._min_peaks_height_spin.setSingleStep(0.01)
        self._min_peaks_height_spin.setValue(0.0075)
        min_peaks_layout = QHBoxLayout(min_peaks_lbl_wdg)
        min_peaks_layout.setContentsMargins(10, 10, 10, 10)
        min_peaks_layout.setSpacing(5)
        min_peaks_layout.addWidget(min_peaks_lbl)
        min_peaks_layout.addWidget(self._min_peaks_height_spin)

        # plate map ---------------------------------------------------------------
        self._plate_map = _PlateMap(self)

        # ok and cancel buttons ----------------------------------------------------
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # Styling ------------------------------------------------------------------
        fixed_width = self._stimulation_area_path._label.sizeHint().width()
        experiment_type_label.setFixedWidth(fixed_width)
        self._plate_map._well_plate_lbl.setFixedWidth(fixed_width)
        min_peaks_lbl.setFixedWidth(fixed_width)
        led_lbl.setFixedWidth(fixed_width)

        # Layout -------------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)
        main_layout.addWidget(experiment_type_wdg)
        main_layout.addWidget(min_peaks_lbl_wdg)
        main_layout.addWidget(self._plate_map)
        main_layout.addWidget(self._button_box)

    def value(self) -> AnalysisParameters | None:
        """Return the model type and path."""
        experiment_type = self._experiment_type_combo.currentText()
        stimulation_mask_path = (
            self._stimulation_area_path.value() if experiment_type == EVOKED else ""
        )
        return {
            "min_peaks_height": self._min_peaks_height_spin.value(),
            "experiment_type": experiment_type,
            "led_power_equation": self._led_power_equation_le.text(),
            "stimulation_mask_path": stimulation_mask_path,
            "genotype_map": self._plate_map._plate_map_genotype.value(),
            "treatment_map": self._plate_map._plate_map_treatment.value(),
        }

    def _on_activity_changed(self, text: str) -> None:
        """Show or hide the stimulation area path and LED power widgets."""
        if text == EVOKED:
            self._stimulation_area_path.show()
            self._led_power_wdg.show()
        else:
            self._stimulation_area_path.hide()
            self._led_power_wdg.hide()
