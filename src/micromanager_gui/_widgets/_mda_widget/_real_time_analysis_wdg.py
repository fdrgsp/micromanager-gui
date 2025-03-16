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


class RealTimeAnalysisParameters(TypedDict):
    """A class to store the values of _SegmentAndAnalyseWidget."""

    min_peaks_height: float  # min height for the peaks detection
    experiment_type: str  # SPONTANEOUS or EVOKED
    stimulation_mask_path: str  # path to the stimulated area mask
    model_type: str  # CYTO3 or CUSTOM
    model_path: str  # path to the custom model
    genotype_map: list[PlateMapData]
    treatment_map: list[PlateMapData]


class RealTimeAnalysisWidget(QGroupBox):
    """Widget to enable Segmentation and Analysis while recording."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._analysis_dialog = RealTimeAnalysisDialog(self)
        self._settings_btn = QPushButton("Segmentation and Analysis Settings...")
        self._settings_btn.setSizePolicy(*FIXED)
        self._settings_btn.clicked.connect(self._show_settings)

        self._enable = QCheckBox("Enable Segmentation and Analysis")
        self._enable.setSizePolicy(*FIXED)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        layout.addWidget(self._enable)
        layout.addWidget(self._settings_btn)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        """Return True if the checkbox is checked."""
        return cast(bool, self._enable.isChecked())

    def value(self) -> RealTimeAnalysisParameters | None:
        """Return the current value of the widget."""
        return self._analysis_dialog.value() if self._enable.isChecked() else None

    def _show_settings(self) -> None:
        """Show the settings dialog."""
        if self._analysis_dialog.isVisible():
            self._analysis_dialog.raise_()
        else:
            self._analysis_dialog.show()


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


class RealTimeAnalysisDialog(QDialog):
    """A Widget to set the segmentation and analysis parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Segmentation and Analysis Parameters")

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

        experiment_type_wdg = QGroupBox(self, title="Experiment")
        experiment_type_wdg.setCheckable(False)
        experiment_type_wdg_layout = QGridLayout(experiment_type_wdg)
        experiment_type_wdg_layout.setContentsMargins(10, 10, 10, 10)
        experiment_type_wdg_layout.setSpacing(5)
        experiment_type_wdg_layout.addWidget(experiment_type_label, 0, 0)
        experiment_type_wdg_layout.addWidget(self._experiment_type_combo, 0, 1)
        experiment_type_wdg_layout.addWidget(self._stimulation_area_path, 1, 0, 1, 2)

        # peaks settings -------------------------------------------------------------
        min_peaks_lbl_wdg = QGroupBox(self, title="Find Peaks Settings")
        min_peaks_lbl_wdg.setToolTip(
            "Set the min height for the peaks (used by the scipy find_peaks method)."
        )
        min_peaks_lbl = QLabel("Min Peaks Height:")
        min_peaks_lbl.setSizePolicy(*FIXED)
        self._min_peaks_height_spin = QDoubleSpinBox(self)
        self._min_peaks_height_spin.setRange(0.0, 100000.0)
        self._min_peaks_height_spin.setSingleStep(0.01)
        self._min_peaks_height_spin.setValue(0.01)
        min_peaks_layout = QHBoxLayout(min_peaks_lbl_wdg)
        min_peaks_layout.setContentsMargins(10, 10, 10, 10)
        min_peaks_layout.setSpacing(5)
        min_peaks_layout.addWidget(min_peaks_lbl)
        min_peaks_layout.addWidget(self._min_peaks_height_spin)

        # Cellpose -----------------------------------------------------------------
        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        self._models_combo = QComboBox()
        self._models_combo.addItems(MODELS)
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)

        model_lbl = QLabel("Model:")

        cellpose_wdg = QGroupBox(self, title="Cellpose")
        cellpose_wdg.setCheckable(False)
        cellpose_wdg_layout = QGridLayout(cellpose_wdg)
        cellpose_wdg_layout.setContentsMargins(10, 10, 10, 10)
        cellpose_wdg_layout.setSpacing(5)
        cellpose_wdg_layout.addWidget(model_lbl, 0, 0)
        cellpose_wdg_layout.addWidget(self._models_combo, 0, 1)
        cellpose_wdg_layout.addWidget(self._browse_custom_model, 1, 0, 1, 2)

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
        self._browse_custom_model._label.setFixedWidth(fixed_width)
        self._plate_map._well_plate_lbl.setFixedWidth(fixed_width)
        model_lbl.setFixedWidth(fixed_width)
        min_peaks_lbl.setFixedWidth(fixed_width)

        # Layout -------------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)
        main_layout.addWidget(experiment_type_wdg)
        main_layout.addWidget(min_peaks_lbl_wdg)
        main_layout.addWidget(cellpose_wdg)
        main_layout.addWidget(self._plate_map)
        main_layout.addWidget(self._button_box)

        # self.setMinimumHeight(1000)

    def value(self) -> RealTimeAnalysisParameters:
        """Return the model type and path."""
        # Get experiment type and stimulation mask path
        experiment_type = self._experiment_type_combo.currentText()
        stimulation_mask_path = (
            self._stimulation_area_path.value() if experiment_type == EVOKED else ""
        )
        # Get model type and path
        model_type = self._models_combo.currentText()
        model_path = self._browse_custom_model.value() if model_type == CUSTOM else ""
        return {
            "min_peaks_height": self._min_peaks_height_spin.value(),
            "experiment_type": experiment_type,
            "stimulation_mask_path": stimulation_mask_path,
            "model_type": model_type,
            "model_path": model_path,
            "genotype_map": self._plate_map._plate_map_genotype.value(),
            "treatment_map": self._plate_map._plate_map_treatment.value(),
        }

    def _on_activity_changed(self, text: str) -> None:
        """Show or hide the stimulated area path widget."""
        (
            self._stimulation_area_path.show()
            if text == EVOKED
            else self._stimulation_area_path.hide()
        )

    def _on_model_combo_changed(self, model: str) -> None:
        """Show the custom model path if the model is 'custom'."""
        (
            self._browse_custom_model.show()
            if model == "custom"
            else self._browse_custom_model.hide()
        )
