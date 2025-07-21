from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fonticon_mdi6 import MDI6
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

# from ._plate_map import PlateMapWidget
from ._plate_map import PlateMapData, PlateMapDataOld, PlateMapWidget
from ._util import (
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CALCIUM_NETWORK_THRESHOLD,
    DEFAULT_CALCIUM_SYNC_JITTER_WINDOW,
    DEFAULT_DFF_WINDOW,
    DEFAULT_HEIGHT,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    _BrowseWidget,
    create_divider_line,
)

if TYPE_CHECKING:
    from pathlib import Path

    import useq

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

RUNNER_TIME_KEY = "runner_time_ms"
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"
EXCLUDE_AREA_SIZE_THRESHOLD = 10
STIMULATION_AREA_THRESHOLD = 0.1  # 10%
GLOBAL_HEIGHT = "global_height"
GLOBAL_SPIKE_THRESHOLD = "global_spike_threshold"
MULTIPLIER = "multiplier"


@dataclass(frozen=True)
class AnalysisSettingsData:
    """Data structure to hold the analysis settings."""

    plate_map_data: tuple[list[PlateMapData], list[PlateMapData]] | None = None
    experiment_type_data: ExperimentTypeData | None = None
    trace_extraction_data: TraceExtractionData | None = None
    calcium_peaks_data: CalciumPeaksData | None = None
    spikes_data: SpikeData | None = None
    positions: str | None = None


@dataclass(frozen=True)
class ExperimentTypeData:
    """Data structure to hold the experiment type settings."""

    experiment_type: str | None = None
    led_power_equation: str | None = None
    stimulation_area_path: str | None = None


@dataclass(frozen=True)
class TraceExtractionData:
    """Data structure to hold the trace extraction settings."""

    dff_window_size: int
    decay_constant: float


@dataclass(frozen=True)
class CalciumPeaksData:
    """Data structure to hold the calcium peaks settings."""

    peaks_height: float
    peaks_height_mode: str
    peaks_distance: int
    peaks_prominence_multiplier: float
    calcium_synchrony_jitter: int
    calcium_network_threshold: float


@dataclass(frozen=True)
class SpikeData:
    """Data structure to hold the spikes settings."""

    spike_threshold: float
    spike_threshold_mode: str
    burst_threshold: float
    burst_min_duration: int
    burst_blur_sigma: float
    synchrony_lag: int


class _PlateMapWidget(QWidget):
    """Widget to show and edit the plate maps."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # label
        self._plate_map_lbl = QLabel("Set/Edit Plate Map:")
        self._plate_map_lbl.setSizePolicy(*FIXED)

        # button to show the plate map dialog
        self._plate_map_btn = QPushButton("Show/Edit Plate Map")
        self._plate_map_btn.setIcon(icon(MDI6.view_comfy))
        self._plate_map_btn.clicked.connect(self._show_plate_map_dialog)

        # dialog to show the plate maps
        self._plate_map_dialog = QDialog(self)
        plate_map_layout = QHBoxLayout(self._plate_map_dialog)
        plate_map_layout.setContentsMargins(10, 10, 10, 10)
        plate_map_layout.setSpacing(5)
        self._plate_map_genotype = PlateMapWidget(self, title="Genotype Map")
        self._plate_map_treatment = PlateMapWidget(self, title="Treatment Map")
        plate_map_layout.addWidget(self._plate_map_genotype)
        plate_map_layout.addWidget(self._plate_map_treatment)

        # main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._plate_map_lbl)
        layout.addWidget(self._plate_map_btn)
        layout.addStretch(1)

    # PUBLIC METHODS -------------------------------------------------------------

    def value(self) -> tuple[list[PlateMapData], list[PlateMapData]]:
        """Get the plate map data."""
        return (self._plate_map_genotype.value(), self._plate_map_treatment.value())

    def setValue(
        self,
        genotype_map: list[PlateMapData | PlateMapDataOld] | list | Path | str,
        treatment_map: list[PlateMapData | PlateMapDataOld] | list | Path | str,
    ) -> None:
        """Set the plate map data."""
        self._plate_map_genotype.setValue(genotype_map)
        self._plate_map_treatment.setValue(treatment_map)

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._plate_map_lbl.setFixedWidth(width)

    def setPlate(self, plate: useq.WellPlate) -> None:
        """Set the plate for the plate maps."""
        self._plate_map_genotype.setPlate(plate)
        self._plate_map_treatment.setPlate(plate)

    def clear(self) -> None:
        """Clear the plate map data."""
        self._plate_map_genotype.clear()
        self._plate_map_treatment.clear()

    # PRIVATE METHODS ------------------------------------------------------------

    def _show_plate_map_dialog(self) -> None:
        """Show the plate map dialog."""
        # ensure the dialog is visible and properly positioned
        if self._plate_map_dialog.isHidden() or not self._plate_map_dialog.isVisible():
            self._plate_map_dialog.show()
        # always try to bring to front and activate
        self._plate_map_dialog.raise_()
        self._plate_map_dialog.activateWindow()
        # force focus on the dialog
        self._plate_map_dialog.setFocus()


class _ExperimentTypeWidget(QWidget):
    """Widget to select the type of experiment (spontaneous or evoked)...

    ...and related settings.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # experiment type combo
        self._experiment_type_lbl = QLabel("Experiment Type:")
        self._experiment_type_lbl.setSizePolicy(*FIXED)
        self._experiment_type_combo = QComboBox()
        self._experiment_type_combo.addItems([SPONTANEOUS, EVOKED])
        self._experiment_type_combo.currentTextChanged.connect(
            self._on_activity_changed
        )
        experiment_type_layout = QHBoxLayout()
        experiment_type_layout.setContentsMargins(0, 0, 0, 0)
        experiment_type_layout.setSpacing(5)
        experiment_type_layout.addWidget(self._experiment_type_lbl)
        experiment_type_layout.addWidget(self._experiment_type_combo)

        # path selector for stimulated area mask
        self._stimulation_area_path_dialog = _BrowseWidget(
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
        self._stimulation_area_path_dialog.hide()

        # LED power equation widget
        self._led_power_wdg = QWidget(self)
        self._led_power_wdg.setToolTip(
            "Insert an equation to convert the LED power to mW.\n"
            "Supported formats:\n"
            "• Linear: y = m*x + q (e.g. y = 2*x + 3)\n"
            "• Quadratic: y = a*x^2 + b*x + c (e.g. y = 0.5*x^2 + 2*x + 1)\n"
            "• Exponential: y = a*exp(b*x) + c (e.g. y = 2*exp(0.1*x) + 1)\n"
            "• Power: y = a*x^b + c (e.g. y = 2*x^0.5 + 1)\n"
            "• Logarithmic: y = a*log(x) + b (e.g. y = 2*log(x) + 1)\n"
            "Leave empty to use values from the acquisition metadata (%)."
        )
        self._led_lbl = QLabel("LED Power Equation:")
        self._led_lbl.setSizePolicy(*FIXED)
        self._led_power_equation_le = QLineEdit(self)
        self._led_power_equation_le.setPlaceholderText(
            "e.g. y = 2*x + 3 (Leave empty to use values from acquisition metadata)"
        )
        led_layout = QHBoxLayout(self._led_power_wdg)
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.setSpacing(5)
        led_layout.addWidget(self._led_lbl)
        led_layout.addWidget(self._led_power_equation_le)
        self._led_power_wdg.hide()

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addLayout(experiment_type_layout)
        layout.addWidget(self._stimulation_area_path_dialog)
        layout.addWidget(self._led_power_wdg)

    # PUBLIC METHODS ------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._experiment_type_lbl.setFixedWidth(width)
        self._stimulation_area_path_dialog._label.setFixedWidth(width)
        self._led_lbl.setFixedWidth(width)

    def value(self) -> ExperimentTypeData:
        """Get the current values of the widget."""
        return ExperimentTypeData(
            self._experiment_type_combo.currentText(),
            self._led_power_equation_le.text(),
            self._stimulation_area_path_dialog.value(),
        )

    def setValue(self, value: ExperimentTypeData) -> None:
        """Set the values of the widget."""
        if value.led_power_equation is not None:
            self._led_power_equation_le.setText(value.led_power_equation)
        if value.stimulation_area_path is not None:
            self._stimulation_area_path_dialog.setValue(value.stimulation_area_path)
        if value.experiment_type is not None:
            self._experiment_type_combo.setCurrentText(value.experiment_type)
            # update visibility based on experiment type
            self._on_activity_changed(value.experiment_type)

    # PRIVATE METHODS ------------------------------------------------------------

    def _on_activity_changed(self, text: str) -> None:
        """Show or hide the stimulation area path and LED power widgets."""
        if text == EVOKED:
            self._stimulation_area_path_dialog.show()
            self._led_power_wdg.show()
        else:
            self._stimulation_area_path_dialog.hide()
            self._led_power_wdg.hide()


class _TraceExtarctionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ΔF/F0 windows
        self._dff_wdg = QWidget(self)
        self._dff_wdg.setToolTip(
            "Controls the sliding window size for calculating ΔF/F₀ baseline "
            "(expressed in frames).\n\n"
            "The algorithm uses a sliding window to estimate the background "
            "fluorescence:\n"
            "• For each timepoint, calculates the 10th percentile within the window\n"
            "• Window extends from current timepoint backwards by window_size/2 "
            "frames\n"
            "• ΔF/F₀ = (fluorescence - background) / background\n\n"
            "Window size considerations:\n"
            "• Larger values (200-500): More stable baseline, good for slow drifts\n"
            "• Smaller values (50-100): More adaptive, follows local fluorescence "
            "changes\n"
            "• Too small (<20): May track signal itself, reducing ΔF/F₀ sensitivity\n"
            "• Too large (>1000): May not adapt to legitimate baseline shifts."
        )
        self._dff_lbl = QLabel("ΔF/F0 Window Size")
        self._dff_lbl.setSizePolicy(*FIXED)
        self._dff_window_size_spin = QSpinBox(self)
        self._dff_window_size_spin.setRange(0, 10000)
        self._dff_window_size_spin.setSingleStep(1)
        self._dff_window_size_spin.setValue(DEFAULT_DFF_WINDOW)
        dff_layout = QHBoxLayout(self._dff_wdg)
        dff_layout.setContentsMargins(0, 0, 0, 0)
        dff_layout.setSpacing(5)
        dff_layout.addWidget(self._dff_lbl)
        dff_layout.addWidget(self._dff_window_size_spin)

        # Deconvolution decay constant
        self._dec_wdg = QWidget(self)
        self._dec_wdg.setToolTip(
            "Decay constant (tau) for calcium indicator deconvolution.\n"
            "Set to 0 for automatic estimation by OASIS algorithm.\n\n"
            "The decay constant represents how quickly the calcium indicator\n"
            "returns to baseline after a calcium transient."
        )
        self._decay_const_lbl = QLabel("Decay Constant (s):")
        self._decay_const_lbl.setSizePolicy(*FIXED)
        self._decay_constant_spin = QDoubleSpinBox(self)
        dec_wdg_layout = QHBoxLayout(self._dec_wdg)
        self._decay_constant_spin.setDecimals(2)
        self._decay_constant_spin.setRange(0.0, 10.0)
        self._decay_constant_spin.setSingleStep(0.1)
        self._decay_constant_spin.setSpecialValueText("Auto")
        dec_wdg_layout.setContentsMargins(0, 0, 0, 0)
        dec_wdg_layout.setSpacing(5)
        dec_wdg_layout.addWidget(self._decay_const_lbl)
        dec_wdg_layout.addWidget(self._decay_constant_spin)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._dff_wdg)
        layout.addWidget(self._dec_wdg)

    # PUBLIC METHODS ------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._dff_lbl.setFixedWidth(width)
        self._decay_const_lbl.setFixedWidth(width)

    def value(self) -> TraceExtractionData:
        """Get the current values of the widget."""
        return TraceExtractionData(
            self._dff_window_size_spin.value(),
            self._decay_constant_spin.value(),
        )

    def setValue(self, value: TraceExtractionData) -> None:
        """Set the values of the widget."""
        self._dff_window_size_spin.setValue(value.dff_window_size)
        self._decay_constant_spin.setValue(value.decay_constant)


class _PeaksHeightWidget(QWidget):
    """Widget to select the peaks height multiplier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Peak height threshold for detecting calcium transients in deconvolved "
            "ΔF/F0 traces using scipy.signal.find_peaks.\n\n"
            "Two modes:\n"
            "• Global Minimum: Same absolute threshold applied to ALL ROIs across "
            "ALL FOVs. Peaks below this value are rejected everywhere.\n\n"
            "• Noise Multiplier: Adaptive threshold computed individually for EACH "
            "ROI in EACH FOV.\n"
            "  Threshold = noise_level * multiplier, where noise_level "
            "is calculated per ROI using Median Absolute Deviation (MAD).\n\n"
            "For example, a multiplier of 3.0 can be use to detect events 3 standard "
            "deviations above noise."
        )

        self._peaks_height_lbl = QLabel("Minimum Peaks Height:")
        self._peaks_height_lbl.setSizePolicy(*FIXED)

        self._peaks_height_spin = QDoubleSpinBox(self)
        self._peaks_height_spin.setDecimals(4)
        self._peaks_height_spin.setRange(0.0, 100000.0)
        self._peaks_height_spin.setSingleStep(0.01)
        self._peaks_height_spin.setValue(DEFAULT_HEIGHT)

        self._global_peaks_height = QRadioButton("Use as Global Minimum Peaks Height")

        self._height_multiplier = QRadioButton("Use as Noise Level Multiplier")
        self._height_multiplier.setChecked(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._peaks_height_lbl)
        layout.addWidget(self._peaks_height_spin, 1)
        layout.addWidget(self._height_multiplier, 0)
        layout.addWidget(self._global_peaks_height, 0)

    # PUBLIC METHODS ------------------------------------------------------------

    def value(self) -> tuple[float, str]:
        """Return the value of the peaks height multiplier."""
        return (
            self._peaks_height_spin.value(),
            GLOBAL_HEIGHT if self._global_peaks_height.isChecked() else MULTIPLIER,
        )

    def setValue(self, value: tuple[float, str]) -> None:
        """Set the value of the peaks height widget."""
        height, mode = value
        self._peaks_height_spin.setValue(height)
        self._global_peaks_height.setChecked(mode == GLOBAL_HEIGHT)
        self._height_multiplier.setChecked(mode == MULTIPLIER)


class _CalciumPeaksWidget(QWidget):
    """Widget to select the calcium peaks settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # peaks height
        self._peaks_height = _PeaksHeightWidget(self)

        # peaks minimum distance
        self._peaks_distance_wdg = QWidget(self)
        self._peaks_distance_wdg.setToolTip(
            "Minimum distance between peaks in frames.\n"
            "This prevents detecting multiple peaks from the same calcium event.\n\n"
            "Example: If exposure time = 50ms and you want 100ms minimum separation,\n"
            "set distance = 2 frames (100ms ÷ 50ms = 2 frames).\n\n"
            "• Higher values: More conservative, fewer detected peaks\n"
            "• Lower values: More sensitive, may detect noise or incomplete decay\n"
            "• Minimum value: 1 (adjacent frames allowed)."
        )
        self._peaks_distance_lbl = QLabel("Minimum Peaks Distance:")
        self._peaks_distance_lbl.setSizePolicy(*FIXED)
        self._peaks_distance_spin = QSpinBox(self)
        self._peaks_distance_spin.setRange(1, 1000)
        self._peaks_distance_spin.setSingleStep(1)
        self._peaks_distance_spin.setValue(2)
        peaks_distance_layout = QHBoxLayout(self._peaks_distance_wdg)
        peaks_distance_layout.setContentsMargins(0, 0, 0, 0)
        peaks_distance_layout.setSpacing(5)
        peaks_distance_layout.addWidget(self._peaks_distance_lbl)
        peaks_distance_layout.addWidget(self._peaks_distance_spin)

        # peaks prominence
        self._peaks_prominence_wdg = QWidget(self)
        self._peaks_prominence_wdg.setToolTip(
            "Controls the prominence threshold multiplier for peak validation.\n"
            "Prominence measures how much a peak stands out from surrounding\n"
            "baseline, helping distinguish real calcium events from noise.\n\n"
            "Prominence threshold = noise_level * multiplier\n\n"
            "• Value of 1.0: Uses noise level as prominence threshold\n"
            "• Values >1.0: Requires peaks to be more prominent than noise level\n"
            "• Values <1.0: More lenient, allows peaks closer to noise level\n\n"
            "Increase if detecting too many noise artifacts as peaks."
        )
        self._peaks_prominence_lbl = QLabel("Peaks Prominence Multiplier:")
        self._peaks_prominence_lbl.setSizePolicy(*FIXED)
        self._peaks_prominence_multiplier_spin = QDoubleSpinBox(self)
        self._peaks_prominence_multiplier_spin.setDecimals(4)
        self._peaks_prominence_multiplier_spin.setRange(0, 100000.0)
        self._peaks_prominence_multiplier_spin.setSingleStep(0.01)
        self._peaks_prominence_multiplier_spin.setValue(1)
        peaks_prominence_layout = QHBoxLayout(self._peaks_prominence_wdg)
        peaks_prominence_layout.setContentsMargins(0, 0, 0, 0)
        peaks_prominence_layout.setSpacing(5)
        peaks_prominence_layout.addWidget(self._peaks_prominence_lbl)
        peaks_prominence_layout.addWidget(self._peaks_prominence_multiplier_spin)

        # synchrony jitter window
        self._calcium_synchrony_wdg = QWidget(self)
        self._calcium_synchrony_wdg.setToolTip(
            "Calcium Peak Synchrony Analysis Settings\n\n"
            "Jitter Window Parameter:\n"
            "Controls the temporal tolerance for detecting synchronous "
            "calcium peaks.\n\n"
            "What the value means:\n"
            "• Value = 2: Peaks within ±2 frames are considered synchronous\n"
            "• Larger values detect more synchrony but may include false positives\n"
            "• Smaller values are more strict but may miss genuine synchrony\n\n"
            "Example with Jitter = 2:\n"
            "ROI 1 peaks: [10, 25, 40]  ROI 2 peaks: [12, 24, 41]\n"
            "Result: All pairs are synchronous (differences ≤ 2 frames)."
        )
        self._calcium_jitter_window_lbl = QLabel("Synchrony Jitter (frames):")
        self._calcium_jitter_window_lbl.setSizePolicy(*FIXED)
        self._calcium_synchrony_jitter_spin = QSpinBox(self)
        self._calcium_synchrony_jitter_spin.setRange(0, 100)
        self._calcium_synchrony_jitter_spin.setSingleStep(1)
        self._calcium_synchrony_jitter_spin.setValue(DEFAULT_CALCIUM_SYNC_JITTER_WINDOW)
        calcium_synchrony_layout = QHBoxLayout(self._calcium_synchrony_wdg)
        calcium_synchrony_layout.setContentsMargins(0, 0, 0, 0)
        calcium_synchrony_layout.setSpacing(5)
        calcium_synchrony_layout.addWidget(self._calcium_jitter_window_lbl)
        calcium_synchrony_layout.addWidget(self._calcium_synchrony_jitter_spin)

        # network connectivity threshold
        self._calcium_network_wdg = QWidget(self)
        self._calcium_network_wdg.setToolTip(
            "Network Connectivity Threshold (Percentile)\n\n"
            "Controls which correlation values become network connections.\n"
            "Uses PERCENTILE-based thresholding, not absolute correlation values.\n\n"
            "How it works:\n"
            "• Calculates percentile of ALL pairwise correlations\n"
            "• Only correlations above this percentile become connections\n"
            "• 90th percentile = top 10% of correlations become edges\n"
            "• 95th percentile = top 5% (more conservative)\n"
            "• 80th percentile = top 20% (more liberal)\n\n"
            "Important: A 0.95 correlation may show as 'not connected'\n"
            "if most correlations in your data are higher (e.g., 0.96-0.99).\n"
            "This ensures only the STRONGEST connections are shown\n"
            "relative to your specific dataset."
        )
        self._calcium_network_lbl = QLabel("Network Threshold (%):")
        self._calcium_network_lbl.setSizePolicy(*FIXED)
        self._calcium_network_threshold_spin = QDoubleSpinBox(self)
        self._calcium_network_threshold_spin.setRange(1.0, 100.0)
        self._calcium_network_threshold_spin.setSingleStep(5.0)
        self._calcium_network_threshold_spin.setDecimals(1)
        self._calcium_network_threshold_spin.setValue(DEFAULT_CALCIUM_NETWORK_THRESHOLD)
        calcium_network_layout = QHBoxLayout(self._calcium_network_wdg)
        calcium_network_layout.setContentsMargins(0, 0, 0, 0)
        calcium_network_layout.setSpacing(5)
        calcium_network_layout.addWidget(self._calcium_network_lbl)
        calcium_network_layout.addWidget(self._calcium_network_threshold_spin)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._peaks_height)
        layout.addWidget(self._peaks_distance_wdg)
        layout.addWidget(self._peaks_prominence_wdg)
        layout.addWidget(self._calcium_synchrony_wdg)
        layout.addWidget(self._calcium_network_wdg)

    # PUBLIC METHODS ------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._peaks_height._peaks_height_lbl.setFixedWidth(width)
        self._peaks_distance_lbl.setFixedWidth(width)
        self._peaks_prominence_lbl.setFixedWidth(width)
        self._calcium_jitter_window_lbl.setFixedWidth(width)
        self._calcium_network_lbl.setFixedWidth(width)

    def value(self) -> CalciumPeaksData:
        """Get the current values of the widget."""
        return CalciumPeaksData(
            *self._peaks_height.value(),
            self._peaks_distance_spin.value(),
            self._peaks_prominence_multiplier_spin.value(),
            self._calcium_synchrony_jitter_spin.value(),
            self._calcium_network_threshold_spin.value(),
        )

    def setValue(self, value: CalciumPeaksData) -> None:
        """Set the values of the widget."""
        self._peaks_height.setValue((value.peaks_height, value.peaks_height_mode))
        self._peaks_distance_spin.setValue(value.peaks_distance)
        self._peaks_prominence_multiplier_spin.setValue(
            value.peaks_prominence_multiplier
        )
        self._calcium_synchrony_jitter_spin.setValue(value.calcium_synchrony_jitter)
        self._calcium_network_threshold_spin.setValue(value.calcium_network_threshold)


class _SpikeThresholdWidget(QWidget):
    """Widget to select the spike threshold multiplier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Spike detection threshold for identifying spikes in OASIS-deconvolved "
            "inferred spike traces.\n\n"
            "Two modes:\n"
            "• Global Minimum: Same absolute threshold applied to ALL ROIs across "
            "ALL FOVs. Spike amplitudes below this value are rejected (set to 0) "
            "everywhere.\n\n"
            "• Noise Multiplier: Adaptive threshold computed individually for EACH "
            "ROI in EACH FOV.\n"
            "  For ROIs with ≥10 detected spikes: "
            "Threshold = 10th_percentile_of_spikes * multiplier\n"
            "  For ROIs with <10 spikes: Threshold = 0.01 * multiplier (fallback)"
        )

        self._spike_threshold_lbl = QLabel("Spike Detection Threshold:")
        self._spike_threshold_lbl.setSizePolicy(*FIXED)

        self._spike_threshold_spin = QDoubleSpinBox(self)
        self._spike_threshold_spin.setDecimals(4)
        self._spike_threshold_spin.setRange(0.0, 10000.0)
        self._spike_threshold_spin.setSingleStep(0.1)
        self._spike_threshold_spin.setValue(DEFAULT_SPIKE_THRESHOLD)

        self._global_spike_threshold = QRadioButton("Use as Global Minimum Threshold")

        self._threshold_multiplier = QRadioButton("Use as Noise Level Multiplier")
        self._threshold_multiplier.setChecked(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._spike_threshold_lbl)
        layout.addWidget(self._spike_threshold_spin, 1)
        layout.addWidget(self._threshold_multiplier, 0)
        layout.addWidget(self._global_spike_threshold, 0)

    # PUBLIC METHODS ------------------------------------------------------------

    def value(self) -> tuple[float, str]:
        """Return the value of the spike threshold."""
        return (
            self._spike_threshold_spin.value(),
            (
                GLOBAL_SPIKE_THRESHOLD
                if self._global_spike_threshold.isChecked()
                else MULTIPLIER
            ),
        )

    def setValue(self, value: tuple[float, str]) -> None:
        """Set the value of the spike threshold widget."""
        threshold, mode = value
        self._spike_threshold_spin.setValue(threshold)
        self._global_spike_threshold.setChecked(mode == GLOBAL_SPIKE_THRESHOLD)
        self._threshold_multiplier.setChecked(mode == MULTIPLIER)


class _BurstWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Settings to control the detection of network bursts in population "
            "activity.\n\n"
            "• Burst Threshold:\n"
            "   Minimum percentage of ROIs that must be active simultaneously to "
            "detect a network burst.\n"
            "   Population activity above this threshold is considered burst "
            "activity.\n"
            "   Higher values (50-80%) detect only strong network-wide events.\n"
            "   Lower values (10-30%) capture weaker coordinated activity.\n\n"
            "• Burst Min Duration (frames):\n"
            "   Minimum duration (in frames) for a detected burst to be "
            "considered valid.\n"
            "   Filters out brief spikes that don't represent sustained "
            "network activity.\n"
            "   Higher values ensure only sustained bursts are detected.\n\n"
            "• Burst Gaussian Blur Sigma:\n"
            "   Gaussian smoothing applied to population activity before "
            "burst detection.\n"
            "   Reduces noise and connects nearby activity peaks into "
            "coherent bursts.\n"
            "   Higher values (2-5) provide more smoothing, merging closer events.\n"
            "   Lower values (0.5-1) preserve temporal precision but may "
            "fragment bursts.\n"
            "   Set to 0 to disable smoothing."
        )

        self._burst_threshold_lbl = QLabel("Burst Threshold (%):")
        self._burst_threshold_lbl.setSizePolicy(*FIXED)
        self._burst_threshold = QDoubleSpinBox(self)
        self._burst_threshold.setDecimals(2)
        self._burst_threshold.setRange(0.0, 100.0)
        self._burst_threshold.setSingleStep(1)
        self._burst_threshold.setValue(DEFAULT_BURST_THRESHOLD)

        self._burst_min_threshold_label = QLabel("Burst Min Duration (frames):")
        self._burst_min_threshold_label.setSizePolicy(*FIXED)
        self._burst_min_duration_frames = QSpinBox(self)
        self._burst_min_duration_frames.setRange(0, 100)
        self._burst_min_duration_frames.setSingleStep(1)
        self._burst_min_duration_frames.setValue(DEFAULT_MIN_BURST_DURATION)

        self._burst_blur_label = QLabel("Burst Gaussian Blur Sigma:")
        self._burst_blur_label.setSizePolicy(*FIXED)
        self._burst_blur_sigma = QDoubleSpinBox(self)
        self._burst_blur_sigma.setDecimals(2)
        self._burst_blur_sigma.setRange(0.0, 100.0)
        self._burst_blur_sigma.setSingleStep(0.5)
        self._burst_blur_sigma.setValue(DEFAULT_BURST_GAUSS_SIGMA)

        burst_layout = QGridLayout(self)
        burst_layout.setContentsMargins(0, 0, 0, 0)
        burst_layout.setSpacing(5)
        burst_layout.addWidget(self._burst_threshold_lbl, 0, 0)
        burst_layout.addWidget(self._burst_threshold, 0, 1)
        burst_layout.addWidget(self._burst_min_threshold_label, 1, 0)
        burst_layout.addWidget(self._burst_min_duration_frames, 1, 1)
        burst_layout.addWidget(self._burst_blur_label, 2, 0)
        burst_layout.addWidget(self._burst_blur_sigma, 2, 1)

    # PUBLIC METHODS ------------------------------------------------------------

    def value(self) -> tuple[float, int, float]:
        """Return the burst detection parameters."""
        return (
            self._burst_threshold.value(),
            self._burst_min_duration_frames.value(),
            self._burst_blur_sigma.value(),
        )

    def setValue(self, value: tuple[float, int, float]) -> None:
        """Set the value of the burst widget."""
        threshold, duration, sigma = value
        self._burst_threshold.setValue(threshold)
        self._burst_min_duration_frames.setValue(duration)
        self._burst_blur_sigma.setValue(sigma)


class _SpikeWidget(QWidget):
    """Widget to select the spike detection settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # spikes threshold
        self._spike_threshold_wdg = _SpikeThresholdWidget(self)

        # burst detection settings
        self._burst_wdg = _BurstWidget(self)

        # spike synchrony settings
        self._spike_synchrony_wdg = QWidget(self)
        self._spike_synchrony_wdg.setToolTip(
            "Inferred Spike Synchrony Analysis Settings\n\n"
            "Max Lag Parameter:\n"
            "Controls the maximum temporal offset for cross-correlation analysis.\n\n"
            "What the value means:\n"
            "• Value = 5: Checks correlations within ±5 frames window\n"
            "• Algorithm slides one spike train over another, looking for "
            "best match within this range\n"
            "• Takes the MAXIMUM correlation found within the lag window\n"
            "• Larger values are more permissive, smaller values more strict\n\n"
            "Example with Max Lag = 5:\n"
            "ROI 1 spikes: [10, 25, 40]  ROI 2 spikes: [12, 24, 41]\n"
            "Algorithm finds high correlation at lag +2 and -1 frames\n"
            "Result: High synchrony score based on best alignment."
        )
        self._spikes_sync_cross_corr_lag = QLabel("Synchrony Lag (frames):")
        self._spikes_sync_cross_corr_lag.setSizePolicy(*FIXED)
        self._spikes_sync_cross_corr_max_lag = QSpinBox(self)
        self._spikes_sync_cross_corr_max_lag.setRange(0, 100)
        self._spikes_sync_cross_corr_max_lag.setSingleStep(1)
        self._spikes_sync_cross_corr_max_lag.setValue(5)
        spikes_sync_cross_corr_layout = QHBoxLayout(self._spike_synchrony_wdg)
        spikes_sync_cross_corr_layout.setContentsMargins(0, 0, 0, 0)
        spikes_sync_cross_corr_layout.setSpacing(DEFAULT_SPIKE_SYNCHRONY_MAX_LAG)
        spikes_sync_cross_corr_layout.addWidget(self._spikes_sync_cross_corr_lag)
        spikes_sync_cross_corr_layout.addWidget(self._spikes_sync_cross_corr_max_lag)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._spike_threshold_wdg)
        layout.addWidget(self._burst_wdg)
        layout.addWidget(self._spike_synchrony_wdg)

    # PUBLIC METHODS ------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._spike_threshold_wdg._spike_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_min_threshold_label.setFixedWidth(width)
        self._burst_wdg._burst_blur_label.setFixedWidth(width)
        self._spikes_sync_cross_corr_lag.setFixedWidth(width)

    def value(self) -> SpikeData:
        """Get the current values of the widget."""
        spike_threshold, spike_threshold_mode = self._spike_threshold_wdg.value()
        burst_threshold, burst_min_duration, burst_blur_sigma = self._burst_wdg.value()
        synchrony_lag = self._spikes_sync_cross_corr_max_lag.value()

        return SpikeData(
            spike_threshold=spike_threshold,
            spike_threshold_mode=spike_threshold_mode,
            burst_threshold=burst_threshold,
            burst_min_duration=burst_min_duration,
            burst_blur_sigma=burst_blur_sigma,
            synchrony_lag=synchrony_lag,
        )

    def setValue(self, value: SpikeData) -> None:
        """Set the values of the widget."""
        tr = (value.spike_threshold, value.spike_threshold_mode)
        self._spike_threshold_wdg.setValue(tr)
        bst = (value.burst_threshold, value.burst_min_duration, value.burst_blur_sigma)
        self._burst_wdg.setValue(bst)
        self._spikes_sync_cross_corr_max_lag.setValue(value.synchrony_lag)


class _ChoosePositionsWidget(QWidget):
    """Widget to select the positions to analyze."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Select the Positions to analyze. Leave blank to analyze all Positions. "
            "You can input single Positions (e.g. 30, 33), a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65). Leave empty "
            "to analyze all Positions.\n\n"
            "NOTE: The Positions are 0-indexed."
        )

        self._pos_lbl = QLabel("Analyze Positions:")
        self._pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit(self)
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33. Leave empty for all.")

        pos_layout = QHBoxLayout(self)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setSpacing(5)
        pos_layout.addWidget(self._pos_lbl)
        pos_layout.addWidget(self._pos_le)

    # PUBLIC METHODS ------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the label."""
        self._pos_lbl.setFixedWidth(width)

    def value(self) -> str:
        """Get the current value of the positions line edit."""
        return cast("str", self._pos_le.text())

    def setValue(self, value: str) -> None:
        """Set the value of the positions line edit."""
        self._pos_le.setText(value)


class _CalciumAnalysisGUI(QGroupBox):
    progress_bar_updated = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("Run Analysis")

        self._plate_map_wdg = _PlateMapWidget(self)
        self._experiment_type_wdg = _ExperimentTypeWidget(self)
        self._trace_extraction_wdg = _TraceExtarctionWidget(self)
        self._calcium_peaks_wdg = _CalciumPeaksWidget(self)
        self._spike_wdg = _SpikeWidget(self)
        self._positions_wdg = _ChoosePositionsWidget(self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(create_divider_line("Plate Map"))
        main_layout.addWidget(self._plate_map_wdg)
        main_layout.addWidget(create_divider_line("Type of Experiment"))
        main_layout.addWidget(self._experiment_type_wdg)
        main_layout.addWidget(create_divider_line("ΔF/F0 and Deconvolution"))
        main_layout.addWidget(self._trace_extraction_wdg)
        main_layout.addWidget(create_divider_line("Calcium Peaks"))
        main_layout.addWidget(self._calcium_peaks_wdg)
        main_layout.addWidget(create_divider_line("Spikes and Bursts"))
        main_layout.addWidget(self._spike_wdg)
        main_layout.addWidget(create_divider_line("Positions to Analyze"))
        main_layout.addWidget(self._positions_wdg)

        # STYLING ------------------------------------------------------------
        fix_width = self._calcium_peaks_wdg._peaks_prominence_lbl.sizeHint().width()
        self._plate_map_wdg.set_labels_width(fix_width)
        self._experiment_type_wdg.set_labels_width(fix_width)
        self._trace_extraction_wdg.set_labels_width(fix_width)
        self._calcium_peaks_wdg.set_labels_width(fix_width)
        self._spike_wdg.set_labels_width(fix_width)
        self._positions_wdg.set_labels_width(fix_width)

    # PUBLIC METHODS ------------------------------------------------------------

    def value(self) -> AnalysisSettingsData:
        """Get the current values of the widget."""
        return AnalysisSettingsData(
            self._plate_map_wdg.value(),
            self._experiment_type_wdg.value(),
            self._trace_extraction_wdg.value(),
            self._calcium_peaks_wdg.value(),
            self._spike_wdg.value(),
            self._positions_wdg.value(),
        )

    def setValue(self, value: AnalysisSettingsData) -> None:
        """Set the values of the widget."""
        if value.plate_map_data is not None:
            self._plate_map_wdg.setValue(*value.plate_map_data)
        if value.experiment_type_data is not None:
            self._experiment_type_wdg.setValue(value.experiment_type_data)
        if value.trace_extraction_data is not None:
            self._trace_extraction_wdg.setValue(value.trace_extraction_data)
        if value.calcium_peaks_data is not None:
            self._calcium_peaks_wdg.setValue(value.calcium_peaks_data)
        if value.spikes_data is not None:
            self._spike_wdg.setValue(value.spikes_data)
        if value.positions is not None:
            self._positions_wdg.setValue(value.positions)

    def enable(self, enable: bool) -> None:
        """Enable or disable the widget."""
        self._plate_map_wdg.setEnabled(enable)
        self._experiment_type_wdg.setEnabled(enable)
        self._trace_extraction_wdg.setEnabled(enable)
        self._calcium_peaks_wdg.setEnabled(enable)
        self._spike_wdg.setEnabled(enable)
        self._positions_wdg.setEnabled(enable)


class _AnalysisProgressBarWidget(QWidget):
    """Widget to display the progress of the analysis."""

    updated = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        # main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._progress_bar)
        main_layout.addWidget(self._progress_pos_label)
        main_layout.addWidget(self._elapsed_time_label)

    def maximum(self) -> int:
        """Return the maximum value of the progress bar."""
        return cast("int", self._progress_bar.maximum())

    def set_progress_bar_label(self, text: str) -> None:
        """Update the progress label with elapsed time."""
        self._progress_pos_label.setText(text)

    def set_time_label(self, elapsed_time: str) -> None:
        """Update the elapsed time label."""
        self._elapsed_time_label.setText(elapsed_time)

    def set_range(self, minimum: int, maximum: int) -> None:
        """Set the range of the progress bar."""
        self._progress_bar.setRange(minimum, maximum)

    def reset(self) -> None:
        """Reset the progress bar and elapsed time label."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def auto_update(self) -> None:
        """Automatically update the progress bar value and label.

        The value is incremented by 1 each time this method is called.
        """
        value = self._progress_bar.value() + 1
        self._progress_bar.setValue(value)
        self._progress_pos_label.setText(f"[{value}/{self._progress_bar.maximum()}]")
