from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from skimage import filters, morphology

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER

if TYPE_CHECKING:
    from pathlib import Path

# Define a type variable for the BaseClass
T = TypeVar("T", bound="BaseClass")

RED = "#C33"
GREEN = "#00FF00"
GENOTYPE_MAP = "genotype_plate_map.json"
TREATMENT_MAP = "treatment_plate_map.json"
COND1 = "condition_1"
COND2 = "condition_2"
STIMULATION_MASK = "stimulation_mask.tif"
MWCM = "mW/cm²"
SETTINGS_PATH = "settings.json"
PLATE_PLAN = "plate_plan"
LED_POWER_EQUATION = "led_power_equation"
PEAKS_HEIGHT_VALUE = "peaks_height_value"
PEAKS_HEIGHT_MODE = "peaks_height_mode"
SPIKE_THRESHOLD_VALUE = "spike_threshold_value"
SPIKE_THRESHOLD_MODE = "spike_threshold_mode"
PEAKS_PROMINENCE_MULTIPLIER = "peaks_prominence_multiplier"
PEAKS_DISTANCE = "peaks_distance"
DFF_WINDOW = "dff_window"
BURST_THRESHOLD = "burst_threshold"
BURST_MIN_DURATION = "burst_min_duration"
BURST_GAUSSIAN_SIGMA = "burst_gaussian_sigma"
EVK_STIM = "evk_stim"
EVK_NON_STIM = "evk_non_stim"
MEAN_SUFFIX = "_Mean"
SEM_SUFFIX = "_SEM"
N_SUFFIX = "_N"
EVENT_KEY = "mda_event"
DECAY_CONSTANT = "decay constant"
SPIKE_SYNCHRONY_METHOD = "cross_correlation"
SPIKES_SYNC_CROSS_CORR_MAX_LAG = "spikes_sync_cross_corr_lag"
CALCIUM_PEAKS_SYNCHRONY_METHOD = "jitter_window"
CALCIUM_SYNC_JITTER_WINDOW = "calcium_sync_jitter_window"
CALCIUM_NETWORK_THRESHOLD = "calcium_network_threshold"

MAX_FRAMES_AFTER_STIMULATION = 5
DEFAULT_BURST_THRESHOLD = 30.0
DEFAULT_MIN_BURST_DURATION = 3
DEFAULT_BURST_GAUSS_SIGMA = 2.0
DEFAULT_DFF_WINDOW = 30
DEFAULT_HEIGHT = 3
DEFAULT_SPIKE_THRESHOLD = 1
DEFAULT_SPIKE_SYNCHRONY_MAX_LAG = 5
DEFAULT_CALCIUM_SYNC_JITTER_WINDOW = 2
DEFAULT_CALCIUM_NETWORK_THRESHOLD = 90.0


@dataclass
class BaseClass:
    """Base class for all classes in the package."""

    def replace(self: T, **kwargs: Any) -> T:
        """Replace the values of the dataclass with the given keyword arguments."""
        return replace(self, **kwargs)


@dataclass
class ROIData(BaseClass):
    """NamedTuple to store ROI data."""

    well_fov_position: str = ""
    raw_trace: list[float] | None = None
    dff: list[float] | None = None
    dec_dff: list[float] | None = None  # deconvolved dff with oasis package
    peaks_dec_dff: list[float] | None = None
    peaks_amplitudes_dec_dff: list[float] | None = None
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    inferred_spikes: list[float] | None = None
    inferred_spikes_threshold: float | None = None
    dec_dff_frequency: float | None = None  # Hz
    condition_1: str | None = None
    condition_2: str | None = None
    cell_size: float | None = None
    cell_size_units: str | None = None
    elapsed_time_list_ms: list[float] | None = None  # in ms
    total_recording_time_sec: float | None = None  # in seconds
    active: bool | None = None
    iei: list[float] | None = None  # interevent interval
    evoked_experiment: bool = False
    stimulated: bool = False
    stimulations_frames_and_powers: dict[str, int] | None = None
    led_pulse_duration: str | None = None
    led_power_equation: str | None = None  # equation for LED power
    calcium_sync_jitter_window: int | None = None  # in frames
    spikes_sync_cross_corr_lag: int | None = None  # in frames
    calcium_network_threshold: float | None = None  # percentile (0-100)
    spikes_burst_threshold: float | None = None  # in percent
    spikes_burst_min_duration: int | None = None  # in seconds
    spikes_burst_gaussian_sigma: float | None = None  # in seconds

    # ... add whatever other data we need


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


class _BrowseWidget(QWidget):
    pathSet = Signal(str)
    filePathSet = Signal(str)

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
        import os

        path_text = self._path.text()
        return str(os.path.normpath(path_text)) if path_text else ""

    def setValue(self, path: str | Path) -> None:
        self._path.setText(str(path))

    def _on_browse(self) -> None:
        if self._is_dir:
            if path := QFileDialog.getExistingDirectory(
                self, f"Select the {self._label_text}.", self._current_path
            ):
                self._path.setText(path)
                self.pathSet.emit(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select the {self._label_text}.",
                "",
                "JSON (*.json); IMAGES (*.tif *.tiff)",
            )
            if path:
                self._path.setText(path)
                self.filePathSet.emit(path)


class _ElapsedTimer(QObject):
    """A timer to keep track of the elapsed time."""

    elapsed_time_updated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_timer = QElapsedTimer()
        self._time_timer = QTimer()
        self._time_timer.timeout.connect(self._update_elapsed_time)

    def start(self) -> None:
        self._elapsed_timer.start()
        self._time_timer.start(1000)

    def stop(self) -> None:
        self._elapsed_timer.invalidate()
        self._time_timer.stop()

    def _update_elapsed_time(self) -> None:
        elapsed_ms = self._elapsed_timer.elapsed()
        elapsed_time_str = self._format_elapsed_time(elapsed_ms)
        self.elapsed_time_updated.emit(elapsed_time_str)

    @staticmethod
    def _format_elapsed_time(milliseconds: int) -> str:
        seconds = milliseconds // 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


class _ProgressBarWidget(QDialog):
    """A progress bar that oscillates between 0 and a given range."""

    def __init__(self, parent: QWidget | None = None, *, text: str = "") -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Sheet)

        self._label = QLabel(text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumWidth(200)
        self._progress_bar.setValue(0)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._label)
        layout.addWidget(self._progress_bar)

    def setText(self, text: str) -> None:
        """Set the text of the progress bar."""
        self._label.setText(text)

    def setValue(self, value: int) -> None:
        """Set the progress bar value."""
        self._progress_bar.setValue(value)

    def setRange(self, min: int, max: int) -> None:
        """Set the progress bar range."""
        self._progress_bar.setRange(min, max)

    def showPercentage(self, visible: bool) -> None:
        """Show or hide the percentage display on the progress bar."""
        self._progress_bar.setTextVisible(visible)


class _WaitingProgressBarWidget(QDialog):
    """A progress bar that oscillates between 0 and a given range."""

    def __init__(
        self, parent: QWidget | None = None, *, range: int = 50, text: str = ""
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

        self._range = range

        self._text = text
        label = QLabel(self._text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumWidth(200)
        self._progress_bar.setRange(0, self._range)
        self._progress_bar.setValue(0)

        self._direction = 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_progress)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(label)
        layout.addWidget(self._progress_bar)

    def start(self) -> None:
        """Start the progress bar."""
        self.show()
        self._timer.start(50)

    def stop(self) -> None:
        """Stop the progress bar."""
        self.hide()
        self._timer.stop()

    def _update_progress(self) -> None:
        """Update the progress bar value.

        The progress bar value will oscillate between 0 and the range and back.
        """
        value = self._progress_bar.value()
        value += self._direction
        if value >= self._range:
            value = self._range
            self._direction = -1
        elif value <= 0:
            value = 0
            self._direction = 1
        self._progress_bar.setValue(value)


def parse_lineedit_text(input_str: str) -> list[int]:
    """Parse the input string and return a list of numbers."""
    parts = input_str.split(",")
    numbers: list[int] = []
    for part in parts:
        part = part.strip()  # remove any leading/trailing whitespace
        if "-" in part:
            with contextlib.suppress(ValueError):
                start, end = map(int, part.split("-"))
                numbers.extend(range(start, end + 1))
        else:
            with contextlib.suppress(ValueError):
                numbers.append(int(part))
    return numbers


def calculate_dff(
    data: np.ndarray, window: int = 100, percentile: int = 10, plot: bool = False
) -> np.ndarray:
    """Calculate the delta F/F using a sliding window and a percentile.

    Parameters
    ----------
    data : np.ndarray
        Array representing the fluorescence trace.
    window : int
        Size of the moving window for the background calculation. Default is 100.
    percentile : int
        Percentile to use for the background calculation. Default is 10.
    plot : bool
        Whether to show a plot of the background and trace. Default is False.

    Returns
    -------
    np.ndarray
        Array representing the delta F/F.
    """
    dff: np.ndarray = np.array([])
    bg: np.ndarray = _calculate_bg(data, window, percentile)
    dff = (data - bg) / bg
    dff -= np.min(dff)

    # plot background and trace
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(bg, label="background", color="black")
        plt.plot(data, label="trace", color="green")
        plt.legend()
        plt.show()

    return dff


def _calculate_bg(data: np.ndarray, window: int, percentile: int = 10) -> np.ndarray:
    """
    Calculate the background using a moving window and a specified percentile.

    Parameters
    ----------
    data : np.ndarray
        Array representing the fluorescence trace.
    window : int
        Size of the moving window.
    percentile : int
        Percentile to use for the background calculation. Default is 10.

    Returns
    -------
    np.ndarray
        Array representing the background.
    """
    # Initialize background array
    background: np.ndarray = np.zeros_like(data)

    # use the lower percentile (e.g., 10th percentile)
    for y in range(len(data)):
        x = max(0, y - window // 2)
        lower_percentile = np.percentile(data[x : y + 1], percentile)
        background[y] = lower_percentile

    # center the window around the current index
    # for y in range(len(data)):
    #     start = max(0, y - window // 2)
    #     end = min(len(data), y + window // 2 + 1)
    #     lower_percentile = np.percentile(data[start:end], percentile)
    #     background[y] = lower_percentile

    return background


def get_iei(peaks: np.ndarray, elapsed_time_list_ms: list[float]) -> list[float] | None:
    """Calculate the interevent interval."""
    # if less than 2 peaks or framerate is negative
    if len(peaks) < 2 or len(elapsed_time_list_ms) <= 1:
        return None

    peaks_time_stamps = [elapsed_time_list_ms[i] for i in peaks]  # ms

    # calculate the difference in time between two consecutive peaks
    iei_ms = np.diff(np.array(peaks_time_stamps))  # ms

    return [float(iei_peak / 1000) for iei_peak in iei_ms]


def create_stimulation_mask(stimulation_file: str) -> np.ndarray:
    """Create a binary mask from an input image.

    We use this to create a mask of the stimulated area. If the input image is a
    mask image already, simply return it.

    Parameters
    ----------
    stimulation_file : str
        Path to the stimulation image.
    """
    # load grayscale image
    blue_img = tifffile.imread(stimulation_file)

    # check if the image is already a binary mask
    unique = np.unique(blue_img)
    # if only pne values which is 1 (full fov illumination)
    if unique.size == 1 and unique[0] == 1:
        return blue_img  # type: ignore
    # if only two values which are 0 and 1 (binary mask)
    elif unique.size == 2:
        # if the image is already a binary mask, return it
        if unique[0] == 0 and unique[1] == 1:
            return blue_img  # type: ignore

    # apply Gaussian Blur to reduce noise
    blur = filters.gaussian(blue_img, sigma=2)

    # set the threshold to otsu's threshold and apply thresholding
    th = blur > filters.threshold_otsu(blur)

    # morphological operations
    selem_small = morphology.disk(2)
    selem_large = morphology.disk(5)

    # closing operation (removes small holes)
    closed = morphology.closing(th, selem_small)

    # erosion (removes small noise)
    eroded = morphology.erosion(closed, selem_small)

    # final closing with a larger structuring element
    final_mask = morphology.closing(eroded, selem_large)

    return final_mask.astype(np.uint8)  # type: ignore


def get_overlap_roi_with_stimulated_area(
    stimulation_mask: np.ndarray, roi_mask: np.ndarray
) -> float:
    """Compute the fraction of the ROI that overlaps with the stimulated area."""
    if roi_mask.shape != stimulation_mask.shape:
        raise ValueError("roi_mask and st_area must have the same dimensions.")

    # count nonzero pixels in the ROI mask
    cell_pixels = np.count_nonzero(roi_mask)

    # if the ROI mask has no pixels, return 0
    if cell_pixels == 0:
        return 0.0

    # count overlapping pixels (logical AND operation)
    overlapping_pixels = np.count_nonzero(roi_mask & stimulation_mask)

    return overlapping_pixels / cell_pixels


def _get_spikes_over_threshold(
    roi_data: ROIData, raw: bool = False
) -> list[float] | None:
    """Get spikes over threshold from ROI data."""
    if not roi_data.inferred_spikes or roi_data.inferred_spikes_threshold is None:
        return None
    if raw:
        # Return raw inferred spikes
        return roi_data.inferred_spikes
    spikes_thresholded = []
    for spike in roi_data.inferred_spikes:
        if spike > roi_data.inferred_spikes_threshold:
            spikes_thresholded.append(spike)
        else:
            spikes_thresholded.append(0.0)
    return spikes_thresholded


def equation_from_str(equation: str) -> Callable | None:
    """Parse various equation formats and return a callable function.

    Supported formats:
    - Linear: y = m*x + q  (e.g. "y = 2*x + 3")
    - Quadratic: y = a*x^2 + b*x + c  (e.g. "y = 0.5*x^2 + 2*x + 1")
    - Exponential: y = a*exp(b*x) + c  (e.g. "y = 2*exp(0.1*x) + 1")
    - Power: y = a*x^b + c  (e.g. "y = 2*x^0.5 + 1")
    - Logarithmic: y = a*log(x) + b  (e.g. "y = 2*log(x) + 1")
    """
    if not equation:
        return None

    # Remove all whitespace for easier parsing
    eq = equation.replace(" ", "").lower()

    try:
        if linear_match := re.match(r"y=([+-]?\d*\.?\d+)\*x([+-]\d*\.?\d+)", eq):
            m = float(linear_match[1])
            q = float(linear_match[2])
            return lambda x: m * x + q

        if quad_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*x\^2([+-]\d*\.?\d+)\*x([+-]\d*\.?\d+)", eq
        ):
            a = float(quad_match[1])
            b = float(quad_match[2])
            c = float(quad_match[3])
            return lambda x: a * x**2 + b * x + c

        if exp_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*exp\(([+-]?\d*\.?\d+)\*x\)([+-]\d*\.?\d+)",
            eq,
        ):
            a = float(exp_match[1])
            b = float(exp_match[2])
            c = float(exp_match[3])
            return lambda x: a * np.exp(b * x) + c

        if power_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*x\^([+-]?\d*\.?\d+)([+-]\d*\.?\d+)", eq
        ):
            a = float(power_match[1])
            b = float(power_match[2])
            c = float(power_match[3])
            return lambda x: a * (x**b) + c

        if log_match := re.match(r"y=([+-]?\d*\.?\d+)\*log\(x\)([+-]\d*\.?\d+)", eq):
            a = float(log_match[1])
            b = float(log_match[2])
            return lambda x: a * np.log(x) + b

        # If no pattern matches, show error
        msg = (
            "Invalid equation format! Using values from the metadata.\n"
            "Only Linear, Quadratic, Exponential, Power, and Logarithmic equations "
            "are supported."
        )
        LOGGER.error(msg)
        return None

    except ValueError as e:
        msg = (
            f"Error parsing equation coefficients: {e}\n"
            "Using values from the metadata."
        )
        LOGGER.error(msg)
        return None


# SYNCHRONY FUNCTIONS -----------------------------------------------------------------


def _get_calcium_peaks_events_from_rois(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract binary peak event trains from ROI data.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all

    Returns
    -------
        Dictionary mapping ROI names to binary peak event arrays
    """
    peak_trains: dict[str, np.ndarray] = {}

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None

    max_frames = 0
    for roi_key, roi_data in roi_data_dict.items():
        try:
            roi_id = int(roi_key)
            if roi_id not in rois or not roi_data.active:
                continue
        except ValueError:
            continue

        max_frames = len(roi_data.raw_trace) if roi_data.raw_trace else 0
        if max_frames == 0:
            return None

        if (
            roi_data.dec_dff
            and roi_data.peaks_dec_dff
            and len(roi_data.peaks_dec_dff) > 0
        ):
            # Create binary peak event train
            peak_train = np.zeros(max_frames, dtype=np.float32)
            for peak_frame in roi_data.peaks_dec_dff:
                if 0 <= int(peak_frame) < max_frames:
                    peak_train[int(peak_frame)] = 1.0

            if np.sum(peak_train) > 0:  # Only include ROIs with at least one peak
                peak_trains[roi_key] = peak_train

    return peak_trains if len(peak_trains) >= 2 else None


def _get_calcium_peaks_event_synchrony_matrix(
    peak_event_dict: dict[str, list[float]],
    method: str = "correlation",
    jitter_window: int = 2,
    max_lag: int = 5,
) -> np.ndarray | None:
    """Compute pairwise peak event synchrony using robust methods.

    Handles timing jitter better than simple correlation.

    Parameters
    ----------
    peak_event_dict : dict
        Dictionary mapping ROI names to binary peak event arrays
    method : str
        Method to use - "jitter_window", "cross_correlation", or "correlation"
    jitter_window : int
        Tolerance window for peak coincidence (frames)
    max_lag : int
        Maximum lag for cross-correlation method (frames)

    Returns
    -------
    np.ndarray or None
        Synchrony matrix robust to small temporal shifts
    """
    active_rois = list(peak_event_dict.keys())
    if len(active_rois) < 2:
        return None

    try:
        # Convert peak event data into a NumPy array of shape (#ROIs, #Timepoints)
        peak_array = np.array(
            [peak_event_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None

    if peak_array.shape[0] < 2:
        return None

    n_rois = peak_array.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(n_rois):
            if i == j:
                synchrony_matrix[i, j] = 1.0  # Perfect self-synchrony
            else:
                events_i = peak_array[i]
                events_j = peak_array[j]

                # Handle case where one or both ROIs have no peaks
                if np.sum(events_i) == 0 or np.sum(events_j) == 0:
                    synchrony_matrix[i, j] = 0.0
                else:
                    if method == "jitter_window":
                        sync_value = _calculate_jitter_window_synchrony(
                            events_i, events_j, jitter_window
                        )
                    elif method == "cross_correlation":
                        sync_value = _calculate_cross_correlation_synchrony(
                            events_i, events_j, max_lag
                        )
                    else:
                        # Fallback to original correlation method (default)
                        correlation = np.corrcoef(events_i, events_j)[0, 1]
                        sync_value = 0.0 if np.isnan(correlation) else abs(correlation)

                    synchrony_matrix[i, j] = sync_value

    return synchrony_matrix


def _get_calcium_peaks_event_synchrony(
    peak_event_synchrony_matrix: np.ndarray | None,
) -> float | None:
    """Calculate global peak event synchrony score from a peak event synchrony matrix.

    This function reuses the same approach as spike synchrony.
    """
    if peak_event_synchrony_matrix is None or peak_event_synchrony_matrix.size == 0:
        return None
    # Ensure the matrix is at least 2x2 and square
    if (
        peak_event_synchrony_matrix.shape[0] < 2
        or peak_event_synchrony_matrix.shape[0] != peak_event_synchrony_matrix.shape[1]
    ):
        return None

    # Calculate the sum of each row, excluding the diagonal
    n_rois = peak_event_synchrony_matrix.shape[0]
    off_diagonal_sum = np.sum(peak_event_synchrony_matrix, axis=1) - np.diag(
        peak_event_synchrony_matrix
    )

    # Normalize by the number of off-diagonal elements per row
    mean_synchrony_per_roi = off_diagonal_sum / (n_rois - 1)

    # Return the median synchrony across all ROIs
    return float(np.median(mean_synchrony_per_roi))


def _get_spike_synchrony_matrix(
    spike_data_dict: dict[str, list[float]],
    method: str = "correlation",
    jitter_window: int = 2,
    max_lag: int = 5,
) -> np.ndarray | None:
    """Compute pairwise spike synchrony from spike amplitude data.

    Parameters
    ----------
    spike_data_dict : dict
        Dictionary mapping ROI names to spike amplitude arrays
    method : str
        Method to use - "jitter_window", "cross_correlation", or "correlation"
    jitter_window : int
        Tolerance window for spike coincidence (frames)
    max_lag : int
        Maximum lag for cross-correlation method (frames)

    Returns
    -------
    np.ndarray or None
        Synchrony matrix robust to small temporal shifts
    """
    active_rois = list(spike_data_dict.keys())
    if len(active_rois) < 2:
        return None

    try:
        # Convert spike data into a NumPy array of shape (#ROIs, #Timepoints)
        spike_array = np.array(
            [spike_data_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None

    if spike_array.shape[0] < 2:
        return None

    # Create binary spike matrices (1 where spike > 0, 0 otherwise)
    binary_spikes = (spike_array > 0).astype(np.float32)

    # Calculate pairwise synchrony using correlation of binary spike trains
    n_rois = binary_spikes.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(n_rois):
            if i == j:
                synchrony_matrix[i, j] = 1.0  # Perfect self-synchrony
            else:
                # Calculate correlation between binary spike trains
                spikes_i = binary_spikes[i]
                spikes_j = binary_spikes[j]

                # Handle case where one or both ROIs have no spikes
                if np.sum(spikes_i) == 0 or np.sum(spikes_j) == 0:
                    synchrony_matrix[i, j] = 0.0
                else:
                    if method == "jitter_window":
                        sync_value = _calculate_jitter_window_synchrony(
                            spikes_i, spikes_j, jitter_window
                        )
                    elif method == "cross_correlation":
                        sync_value = _calculate_cross_correlation_synchrony(
                            spikes_i, spikes_j, max_lag
                        )
                    else:
                        # Fallback to original correlation method (default)
                        correlation = np.corrcoef(spikes_i, spikes_j)[0, 1]
                        sync_value = 0.0 if np.isnan(correlation) else abs(correlation)

                    synchrony_matrix[i, j] = sync_value

    return synchrony_matrix


def _get_spike_synchrony(spike_synchrony_matrix: np.ndarray | None) -> float | None:
    """Calculate global spike synchrony score from a spike synchrony matrix."""
    if spike_synchrony_matrix is None or spike_synchrony_matrix.size == 0:
        return None
    # Ensure the matrix is at least 2x2 and square
    if (
        spike_synchrony_matrix.shape[0] < 2
        or spike_synchrony_matrix.shape[0] != spike_synchrony_matrix.shape[1]
    ):
        return None

    # Calculate the sum of each row, excluding the diagonal
    n_rois = spike_synchrony_matrix.shape[0]
    off_diagonal_sum = np.sum(spike_synchrony_matrix, axis=1) - np.diag(
        spike_synchrony_matrix
    )

    # Normalize by the number of off-diagonal elements per row
    mean_synchrony_per_roi = off_diagonal_sum / (n_rois - 1)

    # Return the median synchrony across all ROIs
    return float(np.median(mean_synchrony_per_roi))


def _calculate_jitter_window_synchrony(
    events_i: np.ndarray, events_j: np.ndarray, jitter_window: int
) -> float:
    """Calculate synchrony allowing for temporal jitter within a window.

    For each peak in ROI i, check if there's a peak in ROI j within ±jitter_window.
    """
    peaks_i = np.where(events_i > 0)[0]
    peaks_j = np.where(events_j > 0)[0]

    if len(peaks_i) == 0 or len(peaks_j) == 0:
        return 0.0

    # Count coincident peaks (bidirectional)
    coincidences_i_to_j = 0
    for peak_i in peaks_i:
        # Check if any peak in j is within jitter window of peak_i
        distances = np.abs(peaks_j - peak_i)
        if np.any(distances <= jitter_window):
            coincidences_i_to_j += 1

    coincidences_j_to_i = 0
    for peak_j in peaks_j:
        # Check if any peak in i is within jitter window of peak_j
        distances = np.abs(peaks_i - peak_j)
        if np.any(distances <= jitter_window):
            coincidences_j_to_i += 1

    # Calculate symmetric synchrony measure
    total_peaks = len(peaks_i) + len(peaks_j)
    total_coincidences = coincidences_i_to_j + coincidences_j_to_i

    return total_coincidences / total_peaks if total_peaks > 0 else 0.0


def _calculate_cross_correlation_synchrony(
    events_i: np.ndarray, events_j: np.ndarray, max_lag: int
) -> float:
    """Calculate synchrony using maximum cross-correlation within lag range."""
    from scipy.signal import correlate

    # Cross-correlation
    xcorr = correlate(events_i, events_j, mode="full")

    # Get the center (zero-lag) position
    center = len(events_i) - 1

    # Extract correlations within max_lag range
    start_idx = max(0, center - max_lag)
    end_idx = min(len(xcorr), center + max_lag + 1)

    local_xcorr = xcorr[start_idx:end_idx]

    # Normalize by the geometric mean of autocorrelations
    auto_i = np.sum(events_i * events_i)
    auto_j = np.sum(events_j * events_j)

    if auto_i > 0 and auto_j > 0:
        normalization = np.sqrt(auto_i * auto_j)
        max_correlation = np.max(local_xcorr) / normalization
        return float(np.clip(max_correlation, 0, 1))
    else:
        return 0.0


def separate_stimulated_vs_non_stimulated_peaks(
    dec_dff: np.ndarray,
    peaks_dec_dff: np.ndarray,
    pulse_on_frames_and_powers: dict[str, int],
    is_roi_stimulated: bool,
    led_pulse_duration: str = "unknown",
    led_power_equation: Callable | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Separate peak amplitudes into stimulated and non-stimulated categories.

    Args:
        dec_dff: Deconvolved dF/F signal
        peaks_dec_dff: Array of peak indices
        pulse_on_frames_and_powers: Dict mapping frame numbers to power values
        is_roi_stimulated: Whether this ROI is in a stimulated area
        led_pulse_duration: Duration of LED pulse (for labeling)
        led_power_equation: Optional function to convert power percentage to mW/cm²

    Returns
    -------
        Tuple of (amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks)
        Each is a dict mapping power_duration strings to lists of amplitudes
    """
    import bisect

    amplitudes_stimulated_peaks: dict[str, list[float]] = {}
    amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}

    sorted_peaks_dec_dff = sorted(peaks_dec_dff)

    for frame, power in pulse_on_frames_and_powers.items():
        stim_frame = int(frame)
        # Find index of first peak >= stim_frame
        i = bisect.bisect_left(sorted_peaks_dec_dff, stim_frame)

        # Check if index is valid
        if i >= len(sorted_peaks_dec_dff):
            continue

        peak_idx = sorted_peaks_dec_dff[i]

        # Check if peak is within stimulation window
        if (
            peak_idx >= stim_frame
            and peak_idx <= stim_frame + MAX_FRAMES_AFTER_STIMULATION
        ):
            amplitude = float(dec_dff[peak_idx])

            # Format power value
            if led_power_equation is not None:
                power_val = led_power_equation(power)
                power_str = f"{power_val:.3f}{MWCM}"
            else:
                power_str = f"{power}%"

            # Create column key
            col = f"{power_str}_{led_pulse_duration}"

            # Categorize based on stimulation status
            if is_roi_stimulated:
                amplitudes_stimulated_peaks.setdefault(col, []).append(amplitude)
            else:
                amplitudes_non_stimulated_peaks.setdefault(col, []).append(amplitude)

    return amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks


def get_stimulated_amplitudes_from_roi_data(
    roi_data: ROIData,
    led_power_equation: Callable | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Get stimulated and non-stimulated amplitudes from ROIData on-demand.

    Args:
        roi_data: ROIData object containing the necessary data
        led_power_equation: Optional function to convert power percentage to mW/cm²

    Returns
    -------
        Tuple of (amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks)
    """
    if (
        not roi_data.evoked_experiment
        or roi_data.dec_dff is None
        or roi_data.peaks_dec_dff is None
        or roi_data.stimulations_frames_and_powers is None
    ):
        return {}, {}

    return separate_stimulated_vs_non_stimulated_peaks(
        dec_dff=np.array(roi_data.dec_dff),
        peaks_dec_dff=np.array(roi_data.peaks_dec_dff),
        pulse_on_frames_and_powers=roi_data.stimulations_frames_and_powers,
        is_roi_stimulated=roi_data.stimulated,
        led_pulse_duration=roi_data.led_pulse_duration or "unknown",
        led_power_equation=led_power_equation,
    )


def create_divider_line(text: str | None = None) -> QWidget:
    """Create a horizontal divider line, optionally with text.

    Parameters
    ----------
    text : str | None
        Optional text to display in front of the divider line

    Returns
    -------
    QWidget
        Widget containing the divider line and optional text
    """
    if text is None:
        return _create_line()
    # Create container widget for text + line
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    # Add text label
    label = QLabel(text)
    # make bold and increase font size
    label.setStyleSheet("font-weight: bold; font-size: 14px; color: rgb(0, 183, 0);")
    layout.addWidget(label)

    line = _create_line()
    layout.addWidget(line, 1)  # Give line stretch factor of 1

    return container


def _create_line() -> QFrame:
    """Create a horizontal line frame for use as a divider."""
    result = QFrame()
    # set color
    # result.setStyleSheet("color: rgb(0, 183, 0);")
    result.setFrameShape(QFrame.Shape.HLine)
    result.setFrameShadow(QFrame.Shadow.Plain)
    return result
