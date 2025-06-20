from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import tifffile
import useq
from fonticon_mdi6 import MDI6
from oasis.functions import deconvolve
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from ._logger import LOGGER
from ._plate_map import PlateMapWidget
from ._to_csv import save_analysis_data_to_csv, save_trace_data_to_csv
from ._util import (
    BURST_GAUSSIAN_SIGMA,
    BURST_MIN_DURATION,
    BURST_THRESHOLD,
    CALCIUM_NETWORK_THRESHOLD,
    CALCIUM_SYNC_JITTER_WINDOW,
    COND1,
    COND2,
    DECAY_CONSTANT,
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CALCIUM_NETWORK_THRESHOLD,
    DEFAULT_CALCIUM_SYNC_JITTER_WINDOW,
    DEFAULT_DFF_WINDOW,
    DEFAULT_HEIGHT,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    DFF_WINDOW,
    EVENT_KEY,
    GENOTYPE_MAP,
    GREEN,
    LED_POWER_EQUATION,
    PEAKS_DISTANCE,
    PEAKS_HEIGHT_MODE,
    PEAKS_HEIGHT_VALUE,
    PEAKS_PROMINENCE_MULTIPLIER,
    RED,
    SETTINGS_PATH,
    SPIKE_THRESHOLD_MODE,
    SPIKE_THRESHOLD_VALUE,
    SPIKES_SYNC_CROSS_CORR_MAX_LAG,
    STIMULATION_MASK,
    TREATMENT_MAP,
    ROIData,
    _BrowseWidget,
    _ElapsedTimer,
    _WaitingProgressBarWidget,
    calculate_dff,
    create_divider_line,
    create_stimulation_mask,
    equation_from_str,
    get_iei,
    get_overlap_roi_with_stimulated_area,
    parse_lineedit_text,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

    from ._plate_map import PlateMapData
    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

RUNNER_TIME_KEY = "runner_time_ms"
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"
EXCLUDE_AREA_SIZE_THRESHOLD = 10
STIMULATION_AREA_THRESHOLD = 0.1  # 10%
GLOBAL_HEIGHT = "global_height"
GLOBAL_SPIKE_THRESHOLD = "global_spike_threshold"
MULTIPLIER = "multiplier"


def single_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.array(a * np.exp(-b * x) + c)


class _PeaksHeightData(TypedDict):
    """TypedDict to store the peaks height data."""

    value: float
    mode: str


class _SpikeThresholdData(TypedDict):
    """TypedDict to store the spike threshold data."""

    value: float
    mode: str


class _BurstData(TypedDict):
    """TypedDict to store the burst data."""

    burst_threshold: float
    burst_min_duration_frames: int
    burst_gauss_sigma: float


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

    def value(self) -> _PeaksHeightData:
        """Return the value of the peaks height multiplier."""
        return {
            "value": self._peaks_height_spin.value(),
            "mode": (
                GLOBAL_HEIGHT if self._global_peaks_height.isChecked() else MULTIPLIER
            ),
        }

    def setValue(self, value: _PeaksHeightData | dict) -> None:
        """Set the value of the peaks height widget."""
        if isinstance(value, dict):
            self._peaks_height_spin.setValue(value["value"])
            if value["mode"] == GLOBAL_HEIGHT:
                self._global_peaks_height.setChecked(True)
            else:
                self._height_multiplier.setChecked(True)
        else:
            # default values
            self._peaks_height_spin.setValue(DEFAULT_HEIGHT)
            self._global_peaks_height.setChecked(True)


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

    def value(self) -> _SpikeThresholdData:
        """Return the value of the spike threshold."""
        return {
            "value": self._spike_threshold_spin.value(),
            "mode": (
                GLOBAL_SPIKE_THRESHOLD
                if self._global_spike_threshold.isChecked()
                else MULTIPLIER
            ),
        }

    def setValue(self, value: _SpikeThresholdData | dict) -> None:
        """Set the value of the spike threshold widget."""
        if isinstance(value, dict):
            self._spike_threshold_spin.setValue(value["value"])
            if value["mode"] == GLOBAL_SPIKE_THRESHOLD:
                self._global_spike_threshold.setChecked(True)
            else:
                self._threshold_multiplier.setChecked(True)
        else:
            # default values
            self._spike_threshold_spin.setValue(DEFAULT_SPIKE_THRESHOLD)
            self._threshold_multiplier.setChecked(True)


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

    def value(self) -> _BurstData:
        """Return the burst detection parameters."""
        return {
            "burst_threshold": self._burst_threshold.value(),
            "burst_min_duration_frames": self._burst_min_duration_frames.value(),
            "burst_gauss_sigma": self._burst_blur_sigma.value(),
        }

    def setValue(self, value: _BurstData | dict) -> None:
        """Set the value of the burst widget."""
        if isinstance(value, dict):
            self._burst_threshold.setValue(value["burst_threshold"])
            self._burst_min_duration_frames.setValue(value["burst_min_duration_frames"])
            self._burst_blur_sigma.setValue(value["burst_gauss_sigma"])
        else:
            # default values
            self._burst_threshold.setValue(DEFAULT_BURST_THRESHOLD)
            self._burst_min_duration_frames.setValue(DEFAULT_MIN_BURST_DURATION)
            self._burst_blur_sigma.setValue(DEFAULT_BURST_GAUSS_SIGMA)


class _AnalyseCalciumTraces(QWidget):
    """Widget to extract the roi traces from the data."""

    progress_bar_updated = Signal()

    def __init__(
        self,
        parent: PlateViewer | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
        labels_path: str | None = None,
    ) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer | None = parent

        self._data: TensorstoreZarrReader | OMEZarrReader | None = data

        self._analysis_path: str | None = None
        self._plate_map_data: dict[str, dict[str, str]] = {}
        self._stimulated_area_mask: np.ndarray | None = None
        self._labels_path: str | None = labels_path
        self._analysis_data: dict[str, dict[str, ROIData]] = {}

        self._worker: GeneratorWorker | None = None
        self._cancelled: bool = False

        # list to store the failed labels if they will not be found during the
        # analysis. used to show at the end of the analysis to the user which labels
        # are failed to be found.
        self._failed_labels: list[str] = []

        # ELAPSED TIME TIMER ---------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        # WIDGET TO SHOW/EDIT THE PLATE MAPS -----------------------------------------
        self._plate_map_dialog = QDialog(self)
        plate_map_layout = QHBoxLayout(self._plate_map_dialog)
        plate_map_layout.setContentsMargins(10, 10, 10, 10)
        plate_map_layout.setSpacing(5)
        self._plate_map_genotype = PlateMapWidget(self, title="Genotype Map")
        plate_map_layout.addWidget(self._plate_map_genotype)
        self._plate_map_treatment = PlateMapWidget(self, title="Treatment Map")
        plate_map_layout.addWidget(self._plate_map_treatment)

        self._plate_map_btn = QPushButton("Show/Edit Plate Map")
        self._plate_map_btn.setIcon(icon(MDI6.view_comfy))
        # self._plate_map_btn.setIconSize(QSize(25, 25))
        self._plate_map_btn.clicked.connect(self._show_plate_map_dialog)
        self._plate_map_wdg = QWidget()
        plate_map_lbl = QLabel("Set/Edit Plate Map:")
        plate_map_lbl.setSizePolicy(*FIXED)
        plate_map_group_layout = QHBoxLayout(self._plate_map_wdg)
        plate_map_group_layout.setContentsMargins(0, 0, 0, 0)
        plate_map_group_layout.setSpacing(5)
        plate_map_group_layout.addWidget(plate_map_lbl)
        plate_map_group_layout.addWidget(self._plate_map_btn)
        plate_map_group_layout.addStretch(1)

        # WIDGET TO SELECT THE EXPERIMENT TYPE ---------------------------------------
        self._experiment_type_wdg = QWidget(self)
        experiment_type_wdg_layout = QHBoxLayout(self._experiment_type_wdg)
        experiment_type_wdg_layout.setContentsMargins(0, 0, 0, 0)
        experiment_type_wdg_layout.setSpacing(5)
        activity_combo_label = QLabel("Experiment Type:")
        activity_combo_label.setSizePolicy(*FIXED)
        self._experiment_type_combo = QComboBox()
        self._experiment_type_combo.addItems([SPONTANEOUS, EVOKED])
        self._experiment_type_combo.currentTextChanged.connect(
            self._on_activity_changed
        )
        experiment_type_wdg_layout.addWidget(activity_combo_label)
        experiment_type_wdg_layout.addWidget(self._experiment_type_combo)

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
            "Insert an equation to convert the LED power to mW.\n"
            "Supported formats:\n"
            "• Linear: y = m*x + q (e.g. y = 2*x + 3)\n"
            "• Quadratic: y = a*x^2 + b*x + c (e.g. y = 0.5*x^2 + 2*x + 1)\n"
            "• Exponential: y = a*exp(b*x) + c (e.g. y = 2*exp(0.1*x) + 1)\n"
            "• Power: y = a*x^b + c (e.g. y = 2*x^0.5 + 1)\n"
            "• Logarithmic: y = a*log(x) + b (e.g. y = 2*log(x) + 1)\n"
            "Leave empty to use values from the acquisition metadata (%)."
        )
        led_lbl = QLabel("LED Power Equation:")
        led_lbl.setSizePolicy(*FIXED)
        self._led_power_equation_le = QLineEdit(self)
        self._led_power_equation_le.setPlaceholderText(
            "e.g. y = 2*x + 3 (Leave empty to use values from acquisition metadata)"
        )
        led_layout = QHBoxLayout(self._led_power_wdg)
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.setSpacing(5)
        led_layout.addWidget(led_lbl)
        led_layout.addWidget(self._led_power_equation_le)
        self._led_power_wdg.hide()

        # DF/F SETTINGS --------------------------------------------------------
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
        dff_lbl = QLabel("ΔF/F0 Window Size")
        dff_lbl.setSizePolicy(*FIXED)
        self._dff_window_size_spin = QSpinBox(self)
        self._dff_window_size_spin.setRange(0, 10000)
        self._dff_window_size_spin.setSingleStep(1)
        self._dff_window_size_spin.setValue(DEFAULT_DFF_WINDOW)
        dff_layout = QHBoxLayout(self._dff_wdg)
        dff_layout.setContentsMargins(0, 0, 0, 0)
        dff_layout.setSpacing(5)
        dff_layout.addWidget(dff_lbl)
        dff_layout.addWidget(self._dff_window_size_spin)

        # DECONVOLUTION SETTINGS -------------------------------------------------
        self._dec_wdg = QWidget(self)
        self._dec_wdg.setToolTip(
            "Decay constant (tau) for calcium indicator deconvolution.\n"
            "Set to 0 for automatic estimation by OASIS algorithm.\n\n"
            "The decay constant represents how quickly the calcium indicator\n"
            "returns to baseline after a calcium transient."
        )
        decay_const_lbl = QLabel("Decay Constant (s):")
        decay_const_lbl.setSizePolicy(*FIXED)
        self._decay_constant_spin = QDoubleSpinBox(self)
        dec_wdg_layout = QHBoxLayout(self._dec_wdg)
        self._decay_constant_spin.setDecimals(2)
        self._decay_constant_spin.setRange(0.0, 10.0)
        self._decay_constant_spin.setSingleStep(0.1)
        self._decay_constant_spin.setSpecialValueText("Auto")
        dec_wdg_layout.setContentsMargins(0, 0, 0, 0)
        dec_wdg_layout.setSpacing(5)
        dec_wdg_layout.addWidget(decay_const_lbl)
        dec_wdg_layout.addWidget(self._decay_constant_spin)

        # PEAKS SETTINGS -------------------------------------------------------------
        self._peaks_height_wdg = _PeaksHeightWidget(self)

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
        peaks_prominence_lbl = QLabel("Peaks Prominence Multiplier:")
        peaks_prominence_lbl.setSizePolicy(*FIXED)
        self._peaks_prominence_multiplier_spin = QDoubleSpinBox(self)
        self._peaks_prominence_multiplier_spin.setDecimals(4)
        self._peaks_prominence_multiplier_spin.setRange(0, 100000.0)
        self._peaks_prominence_multiplier_spin.setSingleStep(0.01)
        self._peaks_prominence_multiplier_spin.setValue(1)
        peaks_prominence_layout = QHBoxLayout(self._peaks_prominence_wdg)
        peaks_prominence_layout.setContentsMargins(0, 0, 0, 0)
        peaks_prominence_layout.setSpacing(5)
        peaks_prominence_layout.addWidget(peaks_prominence_lbl)
        peaks_prominence_layout.addWidget(self._peaks_prominence_multiplier_spin)

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
        peaks_distance_lbl = QLabel("Minimum Peaks Distance:")
        peaks_distance_lbl.setSizePolicy(*FIXED)
        self._peaks_distance_spin = QSpinBox(self)
        self._peaks_distance_spin.setRange(1, 1000)
        self._peaks_distance_spin.setSingleStep(1)
        self._peaks_distance_spin.setValue(2)
        peaks_distance_layout = QHBoxLayout(self._peaks_distance_wdg)
        peaks_distance_layout.setContentsMargins(0, 0, 0, 0)
        peaks_distance_layout.setSpacing(5)
        peaks_distance_layout.addWidget(peaks_distance_lbl)
        peaks_distance_layout.addWidget(self._peaks_distance_spin)

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
        calcium_jitter_window_lbl = QLabel("Synchrony Jitter (frames):")
        calcium_jitter_window_lbl.setSizePolicy(*FIXED)
        self._calcium_synchrony_jitter_spin = QSpinBox(self)
        self._calcium_synchrony_jitter_spin.setRange(0, 100)
        self._calcium_synchrony_jitter_spin.setSingleStep(1)
        self._calcium_synchrony_jitter_spin.setValue(DEFAULT_CALCIUM_SYNC_JITTER_WINDOW)
        calcium_synchrony_layout = QHBoxLayout(self._calcium_synchrony_wdg)
        calcium_synchrony_layout.setContentsMargins(0, 0, 0, 0)
        calcium_synchrony_layout.setSpacing(5)
        calcium_synchrony_layout.addWidget(calcium_jitter_window_lbl)
        calcium_synchrony_layout.addWidget(self._calcium_synchrony_jitter_spin)

        # CALCIUM NETWORK CONNECTIVITY THRESHOLD ----------------------------------
        self._calcium_network_wdg = QWidget(self)
        self._calcium_network_wdg.setToolTip(
            "Network Connectivity Threshold (Percentile)\n\n"
            "Controls which correlation values become network connections.\n"
            "Higher values = fewer, stronger connections.\n"
            "Lower values = more, weaker connections.\n\n"
            "90th percentile = top 10% of correlations become edges\n"
            "95th percentile = top 5% (more conservative)\n"
            "80th percentile = top 20% (more liberal)"
        )
        calcium_network_lbl = QLabel("Network Threshold (%):")
        calcium_network_lbl.setSizePolicy(*FIXED)
        self._calcium_network_threshold_spin = QDoubleSpinBox(self)
        self._calcium_network_threshold_spin.setRange(50.0, 99.9)
        self._calcium_network_threshold_spin.setSingleStep(5.0)
        self._calcium_network_threshold_spin.setDecimals(1)
        self._calcium_network_threshold_spin.setValue(DEFAULT_CALCIUM_NETWORK_THRESHOLD)
        calcium_network_layout = QHBoxLayout(self._calcium_network_wdg)
        calcium_network_layout.setContentsMargins(0, 0, 0, 0)
        calcium_network_layout.setSpacing(5)
        calcium_network_layout.addWidget(calcium_network_lbl)
        calcium_network_layout.addWidget(self._calcium_network_threshold_spin)

        # SPIKES SETTINGS ----------------------------------------------------------
        self._spike_threshold_wdg = _SpikeThresholdWidget(self)

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
        spikes_sync_cross_corr_lag = QLabel("Synchrony Lag (frames):")
        spikes_sync_cross_corr_lag.setSizePolicy(*FIXED)
        self._spikes_sync_cross_corr_max_lag = QSpinBox(self)
        self._spikes_sync_cross_corr_max_lag.setRange(0, 100)
        self._spikes_sync_cross_corr_max_lag.setSingleStep(1)
        self._spikes_sync_cross_corr_max_lag.setValue(5)
        spikes_sync_cross_corr_layout = QHBoxLayout(self._spike_synchrony_wdg)
        spikes_sync_cross_corr_layout.setContentsMargins(0, 0, 0, 0)
        spikes_sync_cross_corr_layout.setSpacing(DEFAULT_SPIKE_SYNCHRONY_MAX_LAG)
        spikes_sync_cross_corr_layout.addWidget(spikes_sync_cross_corr_lag)
        spikes_sync_cross_corr_layout.addWidget(self._spikes_sync_cross_corr_max_lag)

        self._burst_wdg = _BurstWidget(self)

        # WIDGET TO SELECT THE POSITIONS TO ANALYZE --------------------------------
        self._pos_wdg = QWidget(self)
        self._pos_wdg.setToolTip(
            "Select the Positions to analyze. Leave blank to analyze all Positions. "
            "You can input single Positions (e.g. 30, 33) a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65). "
            "NOTE: The Positions are 0-indexed."
        )
        pos_wdg_layout = QHBoxLayout(self._pos_wdg)
        pos_wdg_layout.setContentsMargins(0, 0, 0, 0)
        pos_wdg_layout.setSpacing(5)
        pos_lbl = QLabel("Analyze Positions:")
        pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit()
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33. Leave empty for all")
        pos_wdg_layout.addWidget(pos_lbl)
        pos_wdg_layout.addWidget(self._pos_le)

        # PROGRESS BAR -------------------------------------------------------------
        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        # RUN AND CANCEL BUTTONS ----------------------------------------------------
        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        # self._run_btn.setIconSize(QSize(25, 25))
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))
        # self._cancel_btn.setIconSize(QSize(25, 25))
        self._cancel_btn.clicked.connect(self.cancel)

        # STYLING ------------------------------------------------------------------
        fixed_width = peaks_prominence_lbl.sizeHint().width()
        activity_combo_label.setFixedWidth(fixed_width)
        self._stimulation_area_path._label.setFixedWidth(fixed_width)
        led_lbl.setFixedWidth(fixed_width)
        pos_lbl.setFixedWidth(fixed_width)
        self._peaks_height_wdg._peaks_height_lbl.setFixedWidth(fixed_width)
        self._spike_threshold_wdg._spike_threshold_lbl.setFixedWidth(fixed_width)
        peaks_distance_lbl.setFixedWidth(fixed_width)
        dff_lbl.setFixedWidth(fixed_width)
        plate_map_lbl.setFixedWidth(fixed_width)
        decay_const_lbl.setFixedWidth(fixed_width)
        self._spike_threshold_wdg._spike_threshold_lbl.setFixedWidth(fixed_width)
        self._spike_threshold_wdg._global_spike_threshold.setFixedWidth(
            self._peaks_height_wdg._global_peaks_height.sizeHint().width()
        )
        self._burst_wdg._burst_threshold_lbl.setFixedWidth(fixed_width)
        self._burst_wdg._burst_min_threshold_label.setFixedWidth(fixed_width)
        self._burst_wdg._burst_blur_label.setFixedWidth(fixed_width)
        spikes_sync_cross_corr_lag.setFixedWidth(fixed_width)
        calcium_jitter_window_lbl.setFixedWidth(fixed_width)
        calcium_network_lbl.setFixedWidth(fixed_width)

        # LAYOUT -------------------------------------------------------------------
        progress_wdg = QWidget(self)
        progress_wdg_layout = QHBoxLayout(progress_wdg)
        progress_wdg_layout.setContentsMargins(0, 0, 0, 0)
        progress_wdg_layout.addWidget(self._run_btn)
        progress_wdg_layout.addWidget(self._cancel_btn)
        progress_wdg_layout.addWidget(self._progress_bar)
        progress_wdg_layout.addWidget(self._progress_pos_label)
        progress_wdg_layout.addWidget(self._elapsed_time_label)

        self.groupbox = QGroupBox("Run Analysis", self)
        wdg_layout = QVBoxLayout(self.groupbox)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(create_divider_line("Set the Plate Map"))
        wdg_layout.addWidget(self._plate_map_wdg)
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(create_divider_line("Type of Experiment"))
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(self._experiment_type_wdg)
        wdg_layout.addWidget(self._led_power_wdg)
        wdg_layout.addWidget(self._stimulation_area_path)
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(create_divider_line("ΔF/F0 and Deconvolution"))
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(self._dff_wdg)
        wdg_layout.addWidget(self._dec_wdg)
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(create_divider_line("Calcium Peaks"))
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(self._peaks_prominence_wdg)
        wdg_layout.addWidget(self._peaks_distance_wdg)
        wdg_layout.addWidget(self._peaks_height_wdg)
        wdg_layout.addWidget(self._calcium_synchrony_wdg)
        wdg_layout.addWidget(self._calcium_network_wdg)
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(create_divider_line("Spikes and Bursts"))
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(self._spike_threshold_wdg)
        wdg_layout.addWidget(self._spike_synchrony_wdg)
        wdg_layout.addWidget(self._burst_wdg)
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(create_divider_line("Positions to Analyze"))
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(self._pos_wdg)
        wdg_layout.addSpacing(3)
        wdg_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

        self._cancel_waiting_bar = _WaitingProgressBarWidget(
            text="Stopping all the Tasks..."
        )

        # CONNECTIONS --------------------------------------------------------------
        self.progress_bar_updated.connect(self._update_progress_bar)

    @property
    def data(
        self,
    ) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader | None) -> None:
        self._data = data

    @property
    def analysis_data(self) -> dict[str, dict[str, ROIData]]:
        return self._analysis_data

    @analysis_data.setter
    def analysis_data(self, data: dict[str, dict[str, ROIData]]) -> None:
        self._analysis_data = data

    @property
    def labels_path(self) -> str | None:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, labels_path: str | None) -> None:
        self._labels_path = labels_path

    @property
    def analysis_path(self) -> str | None:
        return self._analysis_path

    @analysis_path.setter
    def analysis_path(self, analysis_path: str | None) -> None:
        self._analysis_path = analysis_path

    @property
    def stimulation_area_path(self) -> str | None:
        return self._stimulation_area_path.value()

    @stimulation_area_path.setter
    def stimulation_area_path(self, stim_area_path: str | None) -> None:
        self._stimulation_area_path.setValue(stim_area_path or "")

    # PUBLIC METHODS ---------------------------------------------------------------

    def run(self) -> None:
        """Extract the roi traces in a separate thread."""
        self._failed_labels.clear()

        pos = self._prepare_for_running()

        if pos is None:
            return

        LOGGER.info("Number of positions: %s", len(pos))

        self._reset_progress_bar()
        self._progress_bar.setRange(0, len(pos))
        self._progress_pos_label.setText(f"[0/{self._progress_bar.maximum()}]")

        # start elapsed timer
        self._elapsed_timer.start()

        self._cancelled = False

        self._enable(False)

        self._worker = create_worker(
            self._extract_traces_data,
            positions=pos,
            _start_thread=True,
            _connect={
                "yielded": self._show_and_log_error,
                "finished": self._on_worker_finished,
                "errored": self._on_worker_errored,
            },
        )

    def cancel(self) -> None:
        """Cancel the current run."""
        self._reset_progress_bar()
        self._enable(True)

        if self._worker is None or not self._worker.is_running:
            return

        self._cancelled = True
        self._worker.quit()
        # stop the elapsed timer
        self._elapsed_timer.stop()
        self._cancel_waiting_bar.start()

    def update_widget_form_settings(self) -> None:
        """Update the widget form from the JSON settings."""
        if not self._analysis_path:
            return None

        settings_json_file = Path(self._analysis_path) / SETTINGS_PATH
        if not settings_json_file.exists():
            LOGGER.warning(f"Settings file {settings_json_file} not found")
            return None

        try:
            with open(settings_json_file) as f:
                self._update_form_settings(f)
        except Exception as e:
            self._show_and_log_error(
                f"Failed to load settings from {settings_json_file}: {e}"
            )
            return None

    def _update_form_settings(self, f: Any) -> None:
        """Update the widget form from the JSON settings file."""
        settings = cast(dict, json.load(f))
        dff_window = cast(int, settings.get(DFF_WINDOW, DEFAULT_DFF_WINDOW))
        self._dff_window_size_spin.setValue(dff_window)
        decay = cast(float, settings.get(DECAY_CONSTANT, 0.0))
        self._decay_constant_spin.setValue(decay)
        pp = cast(str, settings.get(LED_POWER_EQUATION, ""))
        self._led_power_equation_le.setText(pp)
        h_val = cast(float, settings.get(PEAKS_HEIGHT_VALUE, DEFAULT_HEIGHT))
        h_mode = cast(str, settings.get(PEAKS_HEIGHT_MODE, GLOBAL_HEIGHT))
        self._peaks_height_wdg.setValue({"mode": h_mode, "value": h_val})
        spike_thresh_val = cast(
            float, settings.get(SPIKE_THRESHOLD_VALUE, DEFAULT_SPIKE_THRESHOLD)
        )
        spike_thresh_mode = cast(str, settings.get(SPIKE_THRESHOLD_MODE, MULTIPLIER))
        self._spike_threshold_wdg.setValue(
            {"mode": spike_thresh_mode, "value": spike_thresh_val}
        )
        prom_mult = cast(float, settings.get(PEAKS_PROMINENCE_MULTIPLIER, 1.0))
        self._peaks_prominence_multiplier_spin.setValue(prom_mult)
        peaks_distance = cast(int, settings.get(PEAKS_DISTANCE, 2))
        self._peaks_distance_spin.setValue(peaks_distance)

        burst_the = cast(float, settings.get(BURST_THRESHOLD, DEFAULT_BURST_THRESHOLD))
        burst_d = cast(
            int, settings.get(BURST_MIN_DURATION, DEFAULT_MIN_BURST_DURATION)
        )
        burst_g = cast(
            float, settings.get(BURST_GAUSSIAN_SIGMA, DEFAULT_BURST_GAUSS_SIGMA)
        )
        self._burst_wdg.setValue(
            {
                "burst_threshold": burst_the,
                "burst_min_duration_frames": burst_d,
                "burst_gauss_sigma": burst_g,
            }
        )
        spike_sync_lag = cast(
            int,
            settings.get(
                SPIKES_SYNC_CROSS_CORR_MAX_LAG, DEFAULT_SPIKE_SYNCHRONY_MAX_LAG
            ),
        )
        self._spikes_sync_cross_corr_max_lag.setValue(spike_sync_lag)
        calcium_jitter = cast(
            int,
            settings.get(
                CALCIUM_SYNC_JITTER_WINDOW, DEFAULT_CALCIUM_SYNC_JITTER_WINDOW
            ),
        )
        self._calcium_synchrony_jitter_spin.setValue(calcium_jitter)
        calcium_network_threshold = cast(
            float,
            settings.get(CALCIUM_NETWORK_THRESHOLD, DEFAULT_CALCIUM_NETWORK_THRESHOLD),
        )
        self._calcium_network_threshold_spin.setValue(calcium_network_threshold)

    # PRIVATE METHODS --------------------------------------------------------------

    # PREPARATION FOR RUNNING ------------------------------------------------------

    def _prepare_for_running(self) -> list[int] | None:
        """Prepare the widget for running.

        Returns the number of positions to analyze or None if an error occurred.
        """
        if self._worker is not None and self._worker.is_running:
            return None

        if not self._validate_input_data():
            LOGGER.error("Input data validation failed!")
            return None

        if not (analysis_path := self._get_valid_output_path()):
            LOGGER.error("Output path validation failed!")
            return None

        if self._plate_viewer and not self._validate_plate_map():
            return None

        if self._is_evoked_experiment() and not self._prepare_stimulation_mask(
            analysis_path
        ):
            return None

        # get the LED power equation from the line edit
        eq = self._led_power_equation_le.text()
        if equation_from_str(eq):
            self._save_led_equation_to_json_settings(eq)

        self._save_settings_as_json()

        return self._get_positions_to_analyze()

    def _validate_input_data(self) -> bool:
        """Check if required input data is available."""
        if self._data is None:
            self._show_and_log_error(
                "No Data provided!\n"
                "Please load data in File > Load Data and Set Directories..."
            )
            return False

        if self._labels_path is None:
            self._show_and_log_error(
                "Please select the Segmentation Path.\n"
                "You can do this in File > Load Data and Set Directories...' "
                "and set the Segmentation Path'."
            )
            return False

        if self._data.sequence is None:
            self._show_and_log_error("No useq.MDAsequence found in the data!")
            return False

        return True

    def _validate_plate_map(self) -> bool:
        """Validate plate map settings and prompt the user if needed."""
        if self._plate_viewer is None:
            return False

        tr_map = self._plate_map_treatment.value()
        gen_map = self._plate_map_genotype.value()

        if not gen_map and not tr_map:
            msg = "The Plate Map is not set!\n\nDo you want to continue?"
            return self._plate_map_msgbox(msg) == QMessageBox.StandardButton.Yes  # type: ignore

        if (gen_map and not tr_map) or not gen_map:
            map_type = "Genotype" if gen_map else "Treatment"
            msg = (
                f"Only the '{map_type}' Plate Map is set!\n\n"
                "Do you want to continue without both the Plate Maps?"
            )
            return self._plate_map_msgbox(msg) == QMessageBox.StandardButton.Yes  # type: ignore

        return True

    def _get_valid_output_path(self) -> Path | None:
        """Validate and return the output path."""
        if path := self._analysis_path:
            analysis_path = Path(path)
            if not analysis_path.is_dir():
                self._show_and_log_error(
                    "The Analysis Path is not a valid directory!\n"
                    "Please select a valid path in File > "
                    "Load Data and Set Directories...' and set a valid Analysis Path'."
                )
                return None
            return analysis_path

        self._show_and_log_error(
            "Please select the Analysis Path.\n"
            "You can do this in File > Load Data and Set Directories...' "
            "and set the Analysis Path'."
        )
        return None

    def _is_evoked_experiment(self) -> bool:
        """Return True if the activity type is evoked."""
        activity_type = self._experiment_type_combo.currentText()
        return activity_type == EVOKED  # type: ignore

    def _prepare_stimulation_mask(self, analysis_path: Path) -> bool:
        """Generate the stimulation mask if the experiment involves evoked activity."""
        if stim_area_file := self._stimulation_area_path.value():
            self._stimulated_area_mask = create_stimulation_mask(stim_area_file)
            stim_mask_path = analysis_path / STIMULATION_MASK
            tifffile.imwrite(str(stim_mask_path), self._stimulated_area_mask)
            LOGGER.info("Stimulated Area Mask saved at: %s", analysis_path)
            return True

        self._stimulated_area_mask = None
        self._show_and_log_error("No Stimulated Area File Provided!")
        return False

    def _get_positions_to_analyze(self) -> list[int] | None:
        """Get the positions to analyze."""
        if self._data is None or (sequence := self._data.sequence) is None:
            return None

        if not self._pos_le.text():
            positions = [
                i
                for i, p in enumerate(sequence.stage_positions)
                if self._get_labels_file(
                    f"{p.name or f'pos_{str(i).zfill(4)}'}_p{i}.tif"
                )
            ]
        else:
            positions = parse_lineedit_text(self._pos_le.text())
            if not positions:
                self._show_and_log_error("Invalid Positions provided!")
                return None
            if max(positions) >= len(sequence.stage_positions):
                self._show_and_log_error("Input Positions out of range!")
                return None

        LOGGER.info("Positions to analyze: %s", positions)
        return positions

    def _get_labels_file(self, label_name: str) -> str | None:
        """Get the labels file for the given name."""
        if self._labels_path is None:
            return None
        for label_file in Path(self._labels_path).glob("*.tif"):
            if label_file.name.endswith(label_name):
                return str(label_file)
        return None

    # RUN THE ANALYSIS -------------------------------------------------------------

    def _extract_traces_data(self, positions: list[int]) -> Generator[str, None, None]:
        """Extract the roi traces in multiple threads."""
        LOGGER.info("Starting traces analysis...")

        # save plate maps and update the stored _plate_map_data dict
        self._handle_plate_map()

        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)  # leave a couple of cores for the system
        LOGGER.info("CPU count: %s", cpu_count)

        try:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = [
                    executor.submit(self._extract_trace_data_per_position, p)
                    for p in positions
                ]

                for idx, future in enumerate(as_completed(futures)):
                    if self._check_for_abort_requested():
                        LOGGER.info("Abort requested, cancelling all futures...")
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                        LOGGER.info(f"Position {positions[idx]} completed.")
                    except Exception as e:
                        yield f"An error occurred in a position: {e}"
                        break

            LOGGER.info("All positions processed.")

        except Exception as e:
            yield f"An error occurred: {e}"

    def _check_for_abort_requested(self) -> bool:
        return bool(self._worker is not None and self._worker.abort_requested)

    def _handle_plate_map(self) -> None:
        if self._plate_viewer is None or not self._analysis_path:
            return

        condition_1_plate_map = self._plate_map_genotype.value()
        condition_2_plate_map = self._plate_map_treatment.value()

        # save plate map
        LOGGER.info("Saving Plate Maps.")
        path = Path(self._analysis_path) / GENOTYPE_MAP
        self._save_plate_map(path, self._plate_map_genotype.value())
        path = Path(self._analysis_path) / TREATMENT_MAP
        self._save_plate_map(path, self._plate_map_treatment.value())

        # update the stored _plate_map_data dict so we have the condition for each well
        # name as the key. e.g.:
        # {"A1": {"condition_1": "condition_1", "condition_2": "condition_2"}}
        self._plate_map_data.clear()
        for data in condition_1_plate_map:
            self._plate_map_data[data.name] = {COND1: data.condition[0]}

        for data in condition_2_plate_map:
            if data.name in self._plate_map_data:
                self._plate_map_data[data.name][COND2] = data.condition[0]
            else:
                self._plate_map_data[data.name] = {COND2: data.condition[0]}

    def _extract_trace_data_per_position(self, p: int) -> None:
        """Extract the roi traces for the given position."""
        if self._data is None or self._check_for_abort_requested():
            return

        # get the data and metadata for the position
        data, meta = self._data.isel(p=p, metadata=True)

        # the "Event" key was used in the old metadata format
        event_key = EVENT_KEY if EVENT_KEY in meta[0] else "Event"

        # get the fov_name name from metadata
        fov_name = self._get_fov_name(event_key, meta, p)

        # create the dict for the fov if it does not exist
        if fov_name not in self._analysis_data:
            self._analysis_data[fov_name] = {}

        # get the labels file for the position
        labels_path = self._get_labels_file_for_position(fov_name, p)
        if labels_path is None:
            return

        # open the labels file and create masks for each label
        labels = tifffile.imread(labels_path)
        labels_masks = self._create_label_masks_dict(labels)
        sequence = cast(useq.MDASequence, self._data.sequence)

        # get the exposure time from the metadata
        exp_time = meta[0][event_key].get("exposure", 0.0)
        # get timepoints
        timepoints = sequence.sizes["t"]
        # get the elapsed time from the metadata to calculate the total time in seconds
        elapsed_time_list = self.get_elapsed_time_list(meta)
        # if the elapsed time is not available or for any reason is different from
        # the number of timepoints, set it as list of timepoints every exp_time
        if len(elapsed_time_list) != timepoints:
            elapsed_time_list = [i * exp_time for i in range(timepoints)]
        # get the total time in seconds for the recording
        tot_time_sec = (elapsed_time_list[-1] - elapsed_time_list[0]) / 1000

        # check if it is an evoked activity experiment
        evoked_experiment = self._is_evoked_experiment()

        # get the stimulation metadata if it is an evoked activity experiment
        evoked_experiment_meta: dict[str, Any] | None = None
        if evoked_experiment and (seq := self._data.sequence) is not None:
            metadata = cast(dict, seq.metadata.get(PYMMCW_METADATA_KEY, {}))
            evoked_experiment_meta = metadata.get("stimulation")

        msg = f"Extracting Traces Data from Well {fov_name}."
        LOGGER.info(msg)
        for label_value, label_mask in tqdm(labels_masks.items(), desc=msg):
            if self._check_for_abort_requested():
                break

            # extract the data
            self._process_roi_trace(
                data,
                meta,
                evoked_experiment_meta,
                fov_name,
                label_value,
                label_mask,
                tot_time_sec,
                evoked_experiment,
                elapsed_time_list,
            )

        # save the analysis data for the well
        self._save_analysis_data(fov_name)

        # update the progress bar
        self.progress_bar_updated.emit()

    def _get_fov_name(self, event_key: str, meta: list[dict], p: int) -> str:
        """Retrieve the fov name from metadata."""
        # the "Event" key was used in the old metadata format
        pos_name = meta[0].get(event_key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
        return f"{pos_name}_p{p}"

    def _get_labels_file_for_position(self, fov: str, p: int) -> str | None:
        """Retrieve the labels file for the given position."""
        # if the fov name does not end with "_p{p}", add it
        labels_name = f"{fov}.tif" if fov.endswith(f"_p{p}") else f"{fov}_p{p}.tif"
        labels_path = self._get_labels_file(labels_name)
        if labels_path is None:
            self._failed_labels.append(labels_name)
            LOGGER.error("No labels found for %s!", labels_name)
        return labels_path

    def _create_label_masks_dict(self, labels: np.ndarray) -> dict[int, np.ndarray]:
        """Create masks for each label in the labels image."""
        # get the range of labels and remove the background (0)
        labels_range = np.unique(labels[labels != 0])
        return {label_value: (labels == label_value) for label_value in labels_range}

    def get_elapsed_time_list(self, meta: list[dict]) -> list[float]:
        elapsed_time_list: list[float] = []
        # get the elapsed time for each timepoint to calculate tot_time_sec
        if RUNNER_TIME_KEY in meta[0]:  # new metadata format
            for m in meta:
                rt = m[RUNNER_TIME_KEY]
                if rt is not None:
                    elapsed_time_list.append(float(rt))
        return elapsed_time_list

    def _process_roi_trace(
        self,
        data: np.ndarray,
        meta: list[dict],
        evoked_meta: dict[str, Any] | None,
        fov_name: str,
        label_value: int,
        label_mask: np.ndarray,
        tot_time_sec: float,
        evoked_exp: bool,
        elapsed_time_list: list[float],
    ) -> None:
        """Process individual ROI traces."""
        # get the data for the current label
        masked_data = data[:, label_mask]

        # get the size of the roi in µm or px if µm is not available
        roi_size_pixel = masked_data.shape[1]  # area
        px_keys = ["pixel_size_um", "PixelSizeUm"]
        px_size = None
        for key in px_keys:
            px_size = meta[0].get(key, None)
            if px_size:
                break
        # calculate the size of the roi in µm if px_size is available or not 0,
        # otherwise use the size is in pixels
        roi_size = roi_size_pixel * (px_size**2) if px_size else roi_size_pixel

        # exclude small rois, might not be necessary if trained cellpose performs
        # better
        if px_size and roi_size < EXCLUDE_AREA_SIZE_THRESHOLD:
            return

        # check if the roi is stimulated
        roi_stimulation_overlap_ratio = 0.0
        if evoked_exp and self._stimulated_area_mask is not None:
            roi_stimulation_overlap_ratio = get_overlap_roi_with_stimulated_area(
                self._stimulated_area_mask, label_mask
            )

        # compute the mean for each frame
        roi_trace: np.ndarray = masked_data.mean(axis=1)
        win = self._dff_window_size_spin.value()
        # calculate the dff of the roi trace
        dff = calculate_dff(roi_trace, window=win, plot=False)

        # compute the decay constant
        tau = self._decay_constant_spin.value()
        g: tuple[float, ...] | None = None
        if tau > 0.0:
            fs = len(dff) / tot_time_sec  # Sampling frequency (Hz)
            g = np.exp(-1 / (fs * tau))
        else:
            g = None
        # deconvolve the dff trace with adaptive penalty
        dec_dff, spikes, _, t, _ = deconvolve(dff, penalty=1, g=(g,))
        dec_dff = cast(np.ndarray, dec_dff)
        spikes = cast(np.ndarray, spikes)
        LOGGER.info(
            f"Decay constant: {t} seconds, "
            f"Sampling frequency: {len(roi_trace) / tot_time_sec} Hz"
        )

        # for spike amplitudes use percentile-based approach to determine noise level
        non_zero_spikes = spikes[spikes > 0]
        # need sufficient data for reliable percentile
        if len(non_zero_spikes) > 10:
            # Use 10th percentile of non-zero spikes as noise reference
            spike_noise_reference = float(np.percentile(non_zero_spikes, 10))
        else:
            # fallback to default
            spike_noise_reference = 0.01

        # Use the spike threshold widget to get the spike detection threshold
        spike_threshold_data = self._spike_threshold_wdg.value()
        spike_threshold_value = spike_threshold_data["value"]
        if spike_threshold_data["mode"] == GLOBAL_SPIKE_THRESHOLD:
            spike_detection_threshold = spike_threshold_value
        else:  # MULTIPLIER
            spike_detection_threshold = spike_noise_reference * spike_threshold_value

        # spike_thresholded: list[float] = []
        # for s in spikes:
        #     if s > spike_detection_threshold:
        #         spike_thresholded.append(s)
        #     else:
        #         spike_thresholded.append(0.0)

        # Get noise level from the ΔF/F0 trace using Median Absolute Deviation (MAD)
        # -	Step 1: np.median(dff) -> The median of the dataset dff is computed. The
        # median is the “middle” value of the dataset when sorted, which is robust
        # to outliers (unlike the mean).
        # -	Step 2: np.abs(dff - np.median(dff)) -> The absolute deviation of each
        # value in dff from the median is calculated. This measures how far each
        # value is from the central point (the median).
        # -	Step 3: np.median(...) -> The median of the absolute deviations is
        # computed. This gives the Median Absolute Deviation (MAD), which is a
        # robust measure of the spread of the data. Unlike standard deviation, the
        # MAD is not influenced by extreme outliers.
        # -	Step 4: Division by 0.6745 -> The constant 0.6745 rescales the MAD to
        # make it comparable to the standard deviation if the data follows a normal
        # (Gaussian) distribution. Specifically: for a normal distribution,
        # MAD ≈ 0.6745 * standard deviation. Dividing by 0.6745 converts the MAD
        # into an estimate of the standard deviation.
        # Calculate adaptive penalty based on noise level in the ΔF/F0 trace
        noise_level_dec_dff = float(
            np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
        )

        # Set prominence threshold (how much peaks must stand out from surroundings)
        # Use a fraction of noise level to be less restrictive than height threshold
        prom_multiplier = self._peaks_prominence_multiplier_spin.value()
        peaks_prominence_dec_dff: float = noise_level_dec_dff * prom_multiplier

        # use the peaks height widget to get the height threshold
        # if the mode is GLOBAL_HEIGHT, use the value directly, otherwise
        # use the value as a multiplier of the noise level
        peaks_height_data = self._peaks_height_wdg.value()
        peaks_height_value = peaks_height_data["value"]
        if peaks_height_data["mode"] == GLOBAL_HEIGHT:
            peaks_height_dec_dff = peaks_height_value
        else:  # MULTIPLIER
            peaks_height_dec_dff = noise_level_dec_dff * peaks_height_value

        # Get minimum distance between peaks from user-specified value
        min_distance_frames = self._peaks_distance_spin.value()

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(
            dec_dff,
            prominence=peaks_prominence_dec_dff,
            height=peaks_height_dec_dff,
            distance=min_distance_frames,
        )
        peaks_dec_dff = cast(np.ndarray, peaks_dec_dff)

        # get the amplitudes of the peaks in the dec_dff trace
        peaks_amplitudes_dec_dff = [float(dec_dff[p]) for p in peaks_dec_dff]

        # check if the roi is stimulated
        is_roi_stimulated = roi_stimulation_overlap_ratio > STIMULATION_AREA_THRESHOLD

        # if the experiment is evoked, store the stimulation metadata
        stimulation_frames_and_powers: dict[str, int] | None = None
        led_pulse_duration: str | None = None
        if evoked_exp and evoked_meta is not None:
            # get the stimulation info from the metadata (if any)
            stimulation_frames_and_powers = cast(
                dict, evoked_meta.get("pulse_on_frame", {})
            )
            led_pulse_duration = evoked_meta.get("led_pulse_duration", "unknown")

        # calculate the frequency of the peaks in the dec_dff trace
        frequency = (
            len(peaks_dec_dff) / tot_time_sec
            if tot_time_sec and len(peaks_dec_dff) > 0
            else None
        )

        # get the conditions for the well
        condition_1, condition_2 = self._get_conditions(fov_name)

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        burst_the, burst_min_dur, burst_gauss_sigma = self._burst_wdg.value().values()

        # store the data to the analysis dict as ROIData
        self._analysis_data[fov_name][str(label_value)] = ROIData(
            well_fov_position=fov_name,
            raw_trace=cast(list[float], roi_trace.tolist()),
            dff=cast(list[float], dff.tolist()),
            dec_dff=dec_dff.tolist(),
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            peaks_height_dec_dff=peaks_height_dec_dff,
            dec_dff_frequency=frequency or None,
            inferred_spikes=spikes.tolist(),
            inferred_spikes_threshold=spike_detection_threshold,
            cell_size=roi_size,
            cell_size_units="µm" if px_size is not None else "pixel",
            condition_1=condition_1,
            condition_2=condition_2,
            total_recording_time_sec=tot_time_sec,
            active=len(peaks_dec_dff) > 0,
            iei=iei,
            evoked_experiment=evoked_exp,
            stimulated=is_roi_stimulated,
            stimulations_frames_and_powers=stimulation_frames_and_powers,
            led_pulse_duration=led_pulse_duration,
            led_power_equation=self._led_power_equation_le.text(),
            calcium_sync_jitter_window=self._calcium_synchrony_jitter_spin.value(),
            spikes_sync_cross_corr_lag=self._spikes_sync_cross_corr_max_lag.value(),
            calcium_network_threshold=self._calcium_network_threshold_spin.value(),
            spikes_burst_threshold=cast(float, burst_the),
            spikes_burst_min_duration=cast(int, burst_min_dur),
            spikes_burst_gaussian_sigma=cast(float, burst_gauss_sigma),
        )

    def _get_conditions(self, pos_name: str) -> tuple[str | None, str | None]:
        """Get the conditions for the well if any."""
        condition_1 = condition_2 = None
        if self._plate_map_data:
            well_name = pos_name.split("_")[0]
            if well_name in self._plate_map_data:
                condition_1 = self._plate_map_data[well_name].get(COND1)
                condition_2 = self._plate_map_data[well_name].get(COND2)
            else:
                condition_1 = condition_2 = None
        return condition_1, condition_2

    def _on_worker_finished(self) -> None:
        """Called when the data extraction is finished."""
        LOGGER.info("Traces Analysis Finished.")
        self._enable(True)
        self._elapsed_timer.stop()
        self._cancel_waiting_bar.stop()

        # update the analysis data of the plate viewer
        if self._plate_viewer is not None:
            self._plate_viewer.analysis_data = self._analysis_data

            # automatically set combo boxes to first valid option when analysis
            # data is available - ensures graphs refresh after analysis completion
            for sgh in self._plate_viewer.SW_GRAPHS:
                if sgh._combo.currentText() != "None":
                    # Force refresh for already selected options
                    sgh._on_combo_changed(sgh._combo.currentText())

            for mgh in self._plate_viewer.MW_GRAPHS:
                if mgh._combo.currentText() != "None":
                    # Force refresh for already selected options
                    mgh._on_combo_changed(mgh._combo.currentText())

        # save the analysis data to a JSON file
        if self._analysis_path:
            save_trace_data_to_csv(self._analysis_path, self._analysis_data)
            save_analysis_data_to_csv(self._analysis_path, self._analysis_data)

        # show a message box if there are failed labels
        if self._failed_labels:
            msg = (
                "The following labels were not found during the analysis:\n\n"
                + "\n".join(self._failed_labels)
            )
            self._show_and_log_error(msg)

    def _on_worker_errored(self) -> None:
        """Called when the worker encounters an error."""
        LOGGER.info("Extraction of traces terminated with an error.")
        self._enable(True)
        self._elapsed_timer.stop()
        self._cancel_waiting_bar.stop()

    def _save_plate_map(self, path: Path, data: list[PlateMapData]) -> None:
        """Save the plate map data to a JSON file."""
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    def _save_analysis_data(self, pos_name: str) -> None:
        """Save analysis data to a JSON file."""
        LOGGER.info("Saving JSON file for Well %s.", pos_name)
        if not self._analysis_path:
            return
        path = Path(self._analysis_path) / f"{pos_name}.json"
        with path.open("w") as f:
            json.dump(
                self._analysis_data[pos_name],
                f,
                default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                indent=2,
            )

    # WIDGET -----------------------------------------------------------------------

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(a0)

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._cancel_waiting_bar.setEnabled(True)
        self._plate_map_wdg.setEnabled(enable)
        self._experiment_type_wdg.setEnabled(enable)
        self._stimulation_area_path.setEnabled(enable)
        self._dff_wdg.setEnabled(enable)
        self._dec_wdg.setEnabled(enable)
        self._peaks_distance_wdg.setEnabled(enable)
        self._peaks_height_wdg.setEnabled(enable)
        self._spike_threshold_wdg.setEnabled(enable)
        self._peaks_prominence_wdg.setEnabled(enable)
        self._spike_synchrony_wdg.setEnabled(enable)
        self._calcium_synchrony_wdg.setEnabled(enable)
        self._calcium_network_wdg.setEnabled(enable)
        self._burst_wdg.setEnabled(enable)
        self._pos_wdg.setEnabled(enable)
        self._run_btn.setEnabled(enable)
        if self._plate_viewer is None:
            return
        self._plate_viewer._segmentation_wdg.setEnabled(enable)
        # disable graphs tabs
        self._plate_viewer._tab.setTabEnabled(1, enable)
        self._plate_viewer._tab.setTabEnabled(2, enable)

    def _save_led_equation_to_json_settings(self, eq: str) -> None:
        """Save the LED power equation to a JSON file."""
        if not self.analysis_path:
            return

        settings_json_file = Path(self.analysis_path) / SETTINGS_PATH

        try:
            # Read existing settings if file exists
            settings = {}
            if settings_json_file.exists():
                with open(settings_json_file) as f:
                    settings = json.load(f)

            # Update the LED power equation
            settings[LED_POWER_EQUATION] = eq

            # Write back the complete settings
            with open(settings_json_file, "w") as f:
                json.dump(
                    settings,
                    f,
                    indent=2,
                )
        except Exception as e:
            LOGGER.error(f"Failed to save LED power equation: {e}")

    def _save_settings_as_json(self) -> None:
        """Save the noise multiplier to a JSON file."""
        if not self.analysis_path:
            return

        settings_json_file = Path(self.analysis_path) / SETTINGS_PATH

        try:
            # Read existing settings if file exists
            settings = {}
            if settings_json_file.exists():
                with open(settings_json_file) as f:
                    settings = json.load(f)

            settings[DFF_WINDOW] = self._dff_window_size_spin.value()
            settings[DECAY_CONSTANT] = self._decay_constant_spin.value()
            peaks_h_data = self._peaks_height_wdg.value()
            settings[PEAKS_HEIGHT_VALUE] = peaks_h_data.get("value", DEFAULT_HEIGHT)
            settings[PEAKS_HEIGHT_MODE] = peaks_h_data.get("mode", GLOBAL_HEIGHT)
            spike_thresh_data = self._spike_threshold_wdg.value()
            settings[SPIKE_THRESHOLD_VALUE] = spike_thresh_data.get(
                "value", DEFAULT_SPIKE_THRESHOLD
            )
            settings[SPIKE_THRESHOLD_MODE] = spike_thresh_data.get("mode", MULTIPLIER)
            settings[PEAKS_DISTANCE] = self._peaks_distance_spin.value()
            prom = self._peaks_prominence_multiplier_spin.value()
            settings[PEAKS_PROMINENCE_MULTIPLIER] = prom
            burst_the, burst_d, burst_g = self._burst_wdg.value().values()
            settings[BURST_THRESHOLD] = burst_the
            settings[BURST_MIN_DURATION] = burst_d
            settings[BURST_GAUSSIAN_SIGMA] = burst_g

            settings[SPIKES_SYNC_CROSS_CORR_MAX_LAG] = (
                self._spikes_sync_cross_corr_max_lag.value()
            )
            settings[CALCIUM_SYNC_JITTER_WINDOW] = (
                self._calcium_synchrony_jitter_spin.value()
            )
            settings[CALCIUM_NETWORK_THRESHOLD] = (
                self._calcium_network_threshold_spin.value()
            )

            # Write back the complete settings
            with open(settings_json_file, "w") as f:
                json.dump(
                    settings,
                    f,
                    indent=2,
                )
        except Exception as e:
            LOGGER.error(f"Failed to save noise multiplier: {e}")

    def _on_activity_changed(self, text: str) -> None:
        """Show or hide the stimulation area path and LED power widgets."""
        if text == EVOKED:
            self._stimulation_area_path.show()
            self._led_power_wdg.show()
        else:
            self._stimulation_area_path.hide()
            self._led_power_wdg.hide()

    def _reset_progress_bar(self) -> None:
        """Reset the progress bar and elapsed time label."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def _show_and_log_error(self, msg: str) -> None:
        """Log and display an error message."""
        LOGGER.error(msg)
        show_error_dialog(self, msg)

    def _plate_map_msgbox(self, msg: str) -> Any:
        """Show a message box to ask the user if wants to overwrite the labels."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setText(msg)
        msg_box.setWindowTitle("Plate Map")
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        return msg_box.exec()

    def _update_progress_label(self, time_str: str) -> None:
        """Update the progress label with elapsed time."""
        self._elapsed_time_label.setText(time_str)

    def _update_progress_bar(self) -> None:
        """Update the progress bar value."""
        if self._check_for_abort_requested():
            return
        value = self._progress_bar.value() + 1
        self._progress_bar.setValue(value)
        self._progress_pos_label.setText(f"[{value}/{self._progress_bar.maximum()}]")

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

    def _load_plate_map(self, plate: useq.WellPlate | None) -> None:
        """Load the plate map from the given file."""
        if plate is None:
            return
        # clear the plate map data
        self._plate_map_genotype.clear()
        self._plate_map_treatment.clear()
        # set the plate type
        self._plate_map_genotype.setPlate(plate)
        self._plate_map_treatment.setPlate(plate)
        # load plate map if exists
        if not self._analysis_path:
            return
        gen_path = Path(self._analysis_path) / GENOTYPE_MAP
        if gen_path.exists():
            self._plate_map_genotype.setValue(gen_path)
        treat_path = Path(self._analysis_path) / TREATMENT_MAP
        if treat_path.exists():
            self._plate_map_treatment.setValue(treat_path)
