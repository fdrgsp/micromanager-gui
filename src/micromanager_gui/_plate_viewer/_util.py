from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
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
from scipy.interpolate import CubicSpline
from skimage import filters, morphology

# Define a type variable for the BaseClass
T = TypeVar("T", bound="BaseClass")

RED = "#C33"
GREEN = "#00FF00"
GENOTYPE_MAP = "genotype_plate_map.json"
TREATMENT_MAP = "treatment_plate_map.json"
COND1 = "condition_1"
COND2 = "condition_2"
STIMULATION_MASK = "stimulation_mask.tif"

# -----------------------------------GRAPH PLOTTING-----------------------------------
# Anything added here will appear in the dropdown menu in the graph widget.
# Modify the plot_traces function in _plot_methods.py to add the corresponding plotting
# logic for the new options.

RAW_TRACES = "Raw Traces"
NORMALIZED_TRACES = "Normalized Traces [0, 1]"
DFF = "DeltaF/F0"
DFF_NORMALIZED = "DeltaF/F0 Normalized [0, 1]"
DEC_DFF = "Deconvolved DeltaF/F0"
DEC_DFF_WITH_PEAKS = "Deconvolved DeltaF/F0 with Peaks"
DEC_DFF_NORMALIZED = "Deconvolved DeltaF/F0 Normalized [0, 1]"
DEC_DFF_NORMALIZED_WITH_PEAKS = "Deconvolved DeltaF/F0 Normalized [0, 1] with Peaks"
DEC_DFF_AMPLITUDE = "Deconvolved DeltaF/F0 Amplitudes"
DEC_DFF_FREQUENCY = "Deconvolved DeltaF/F0 Frequencies"
DEC_DFF_AMPLITUDE_VS_FREQUENCY = "Deconvolved DeltaF/F0 Amplitudes vs Frequencies"
DEC_DFF_IEI = "Deconvolved DeltaF/F0 Inter-event Interval"

DEC_DFF_AMPLITUDE_VS_FREQUENCY_ALL = "Deconvolved DeltaF/F0 Amplitudes vs Frequencies"
DEC_DFF_AMPLITUDE_ALL = "Deconvolved DeltaF/F0 Amplitudes"
DEC_DFF_FREQUENCY_ALL = "Deconvolved DeltaF/F0 Frequencies"
RASTER_PLOT = "Raster plot Colored by ROI"
RASTER_PLOT_AMP = "Raster plot Colored by Amplitude"
RASTER_PLOT_AMP_WITH_COLORBAR = "Raster plot Colored by Amplitude with Colorbar"
DEC_DFF_IEI_ALL = "Deconvolved DeltaF/F0 Inter-event Interval"
STIMULATED_AREA = "Stimulated Area"
STIMULATED_ROIS = "Stimulated vs Non-Stimulated ROIs"
STIMULATED_ROIS_WITH_STIMULATED_AREA = (
    "Stimulated vs Non-Stimulated ROIs with Stimulated Area"
)
GLOBAL_CONNECTIVITY_CUBIC = "Global connectivity-Cubic"
GLOBAL_CONNECTIVITY_LINEAR = "Global connectivity-Linear"

SINGLE_WELL_COMBO_OPTIONS = [
    RAW_TRACES,
    NORMALIZED_TRACES,
    DFF,
    DFF_NORMALIZED,
    DEC_DFF,
    DEC_DFF_WITH_PEAKS,
    DEC_DFF_NORMALIZED,
    DEC_DFF_NORMALIZED_WITH_PEAKS,
    DEC_DFF_AMPLITUDE,
    DEC_DFF_FREQUENCY,
    DEC_DFF_AMPLITUDE_VS_FREQUENCY,
    DEC_DFF_IEI,
    RASTER_PLOT,
    RASTER_PLOT_AMP,
    RASTER_PLOT_AMP_WITH_COLORBAR,
    STIMULATED_AREA,
    STIMULATED_ROIS,
    STIMULATED_ROIS_WITH_STIMULATED_AREA,
    GLOBAL_CONNECTIVITY_CUBIC,
    GLOBAL_CONNECTIVITY_LINEAR,
]

MULTI_WELL_COMBO_OPTIONS = [
    DEC_DFF_AMPLITUDE_VS_FREQUENCY_ALL,
    DEC_DFF_AMPLITUDE_ALL,
    DEC_DFF_FREQUENCY_ALL,
    DEC_DFF_IEI_ALL,
]
# ------------------------------------------------------------------------------------


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
    inferred_spikes: list[float] | None = None
    dec_dff_frequency: float | None = None
    condition_1: str | None = None
    condition_2: str | None = None
    cell_size: float | None = None
    cell_size_units: str | None = None
    total_recording_time_in_sec: float | None = None
    active: bool | None = None
    linear_phase: list[float] | None = None
    cubic_phase: list[float] | None = None
    iei: list[float] | None = None  # interevent interval
    stimulated: bool = False
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
        return self._path.text()  # type: ignore

    def setValue(self, path: str) -> None:
        self._path.setText(path)

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

    # Use the lower percentile (e.g., 10th percentile)
    for y in range(len(data)):
        x = max(0, y - window // 2)
        lower_percentile = np.percentile(data[x : y + 1], percentile)
        background[y] = lower_percentile

    return background


def get_linear_phase(frames: int, peaks: np.ndarray) -> list[float]:
    """Calculate the linear phase progression."""
    peaks_list = [int(peak) for peak in peaks]

    if any(p < 0 or p >= frames for p in peaks):
        raise ValueError("All peaks must be within the range of frames.")

    if peaks_list[0] != 0:
        peaks_list.insert(0, 0)
    if peaks_list[-1] != (frames - 1):
        peaks_list.append(frames - 1)

    phase = [0.0] * frames

    for k in range(len(peaks_list) - 1):
        start, end = peaks_list[k], peaks_list[k + 1]

        if start == end:
            continue

        for t in range(start, end):
            phase[t] = (2 * np.pi) * ((t - start) / (end - start)) + (2 * np.pi * k)

    phase[frames - 1] = 2 * np.pi * (len(peaks_list) - 1)

    return phase


def get_cubic_phase(total_frames: int, peaks: np.ndarray) -> list[float]:
    """Calculate the instantaneous phase with smooth interpolation and handle negative values."""  # noqa: E501
    peaks_list = [int(peak) for peak in peaks]

    if peaks_list[0] != 0:
        peaks_list.insert(0, 0)

    if peaks_list[-1] != (total_frames - 1):
        peaks_list.append(total_frames - 1)

    num_cycles = len(peaks_list) - 1

    peak_phases = np.arange(num_cycles + 1) * 2 * np.pi

    cubic_spline = CubicSpline(peaks_list, peak_phases, bc_type="clamped")

    frames = np.arange(total_frames)
    phases = cubic_spline(frames)

    phases = np.clip(phases, 0, None)
    phases = np.mod(phases, 2 * np.pi)

    return [float(phase) for phase in phases]


def get_iei(peaks: list[int], elapsed_time_list: list[float]) -> list[float] | None:
    """Calculate the interevent interval."""
    # if less than 2 peaks or framerate is negative
    if len(peaks) < 2 or len(elapsed_time_list) <= 1:
        return None

    peaks_time_stamps = [elapsed_time_list[i] for i in peaks]  # ms

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
    if np.unique(blue_img).size == 2:
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
