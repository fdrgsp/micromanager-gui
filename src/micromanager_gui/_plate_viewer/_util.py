from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import CubicSpline

# Define a type variable for the BaseClass
T = TypeVar("T", bound="BaseClass")

RED = "#C33"
GREEN = "#00FF00"
GENOTYPE_MAP = "genotype_plate_map.json"
TREATMENT_MAP = "treatment_plate_map.json"
COND1 = "condition_1"
COND2 = "condition_2"


@dataclass
class BaseClass:
    """Base class for all classes in the package."""

    def replace(self: T, **kwargs: Any) -> T:
        """Replace the values of the dataclass with the given keyword arguments."""
        return replace(self, **kwargs)


@dataclass
class ROIData(BaseClass):
    """NamedTuple to store ROI data."""

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
    average_time_interval: float | None = None
    active: bool | None = None
    # ... add whatever other data we need


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


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
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

        self._label = text
        label = QLabel(self._label)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumWidth(200)
        self._progress_bar.setValue(0)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(label)
        layout.addWidget(self._progress_bar)

    def setValue(self, value: int) -> None:
        """Set the progress bar value."""
        self._progress_bar.setValue(value)

    def setRange(self, min: int, max: int) -> None:
        """Set the progress bar range."""
        self._progress_bar.setRange(min, max)


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

def get_linear_phase(frames: int, peaks: list[int]) -> np.ndarray:
    """Calculate the linear phase progression."""
    peaks_copy = peaks.tolist().copy()

    if len(peaks_copy) == 0:
        return None

    if peaks_copy[0] != 0:
        peaks_copy.insert(0, 0)
    if peaks_copy[-1] != (frames-1):
        peaks_copy.append(frames-1)

    phase = [0.0] * frames
    for k in range(len(peaks_copy) - 1):
        start = peaks_copy[k]
        end = peaks_copy[k+1]

        for t in range(start, end):
            instant_phase = (2 * np.pi) * ((t - start)/\
                                            (end - start)) + \
                                            (2 * np.pi * k)
            phase[t] = instant_phase
        # phase.append(2 * np.pi * (len(peaks_copy) - 1))

    return phase

def get_cubic_phase(total_frames: int, peaks: list[int]) -> np.ndarray | None:
    """Calculate the instantaneous phase with smooth interpolation and handle negative values."""
    peaks_copy = peaks.tolist().copy()

    if len(peaks_copy) == 0:
        return None

    # Ensure first peak starts at frame 0
    if peaks_copy[0] != 0:
        peaks_copy.insert(0, 0)

    # Ensure last peak is at the final frame
    if peaks_copy[-1] != total_frames - 1:
        peaks_copy.append(total_frames - 1)

    num_cycles = len(peaks_copy) - 1  # Number of peak-to-peak cycles

    # Define phase values at the peak positions (increments by 2π per cycle)
    peak_phases = np.arange(num_cycles + 1) * 2 * np.pi

    # Use Clamped Cubic Spline to reduce overshooting
    cubic_spline = CubicSpline(peaks_copy, peak_phases, bc_type='clamped')

    # Generate phase values for all frames
    frames = np.arange(total_frames)
    phase = cubic_spline(frames)

    # Handle potential negative values
    phase = np.clip(phase, 0, None)  # Remove negatives
    phase = np.mod(phase, 2 * np.pi)  # Keep phase in range [0, 2π]

    return phase

def get_connectivity(connection_matrix: np.ndarray):
    """Calculate the connection matrix."""
    if connection_matrix:
        if len(connection_matrix) > 1:
            mean_connect = np.median(np.sum(connection_matrix, axis=0) - 1) /\
                (len(connection_matrix) - 1)
        else:
            mean_connect = 'N/A - Only one active ROI'
    else:
        mean_connect = 'No calcium events detected'

    return mean_connect

def get_connectivity_matrix(phase_dict: dict[str, list[float]],
                          path: str, interpolation: str) -> np.ndarray:
    """Calculate global connectivity using vectorized operations."""
    active_rois = list(phase_dict.keys())  # ROI names

    # Convert phase_dict values into a NumPy array of shape (N, T)
    phase_array = np.array([phase_dict[roi] for roi in active_rois])  # Shape (N, T)

    # Compute pairwise phase difference using broadcasting (Shape: (N, N, T))
    phase_diff = np.expand_dims(phase_array, axis=1) \
        - np.expand_dims(phase_array, axis=0)

    # Ensure phase difference is within valid range [0, 2π]
    phase_diff = np.mod(np.abs(phase_diff), 2 * np.pi)

    # Compute cosine and sine of the phase differences
    cos_mean = np.mean(np.cos(phase_diff), axis=2)  # Shape: (N, N)
    sin_mean = np.mean(np.sin(phase_diff), axis=2)  # Shape: (N, N)

    # Compute synchronization index (vectorized)
    connect_matrix = np.sqrt(cos_mean**2 + sin_mean**2)

    fig_path = path + "_" + interpolation
    _plot_connection(connect_matrix, fig_path, active_rois)

    return connect_matrix

def _plot_connection(connect_matrix: np.ndarray, path: str,
                        roi_labels: list[str]) -> None:
    """Plot the connection matrix."""
    fig, ax = plt.subplots()
    im = ax.imshow(connect_matrix)
    ax.figure.colorbar(im, ax=ax)
    # ax.set_xticks(range(connect_matrix.shape[1]), labels="Neuron ID")
    # ax.set_yticks(range(connect_matrix.shape[0]), labels="Neuron ID")
    # ax.spines[:].set_visible(False)
    ax.set_xticks(range(connect_matrix.shape[1]), labels=roi_labels)
    ax.set_yticks(range(connect_matrix.shape[0]), labels=roi_labels)
    ax.set_xlabel("Neuron ID")
    ax.set_ylabel("Neuron ID")
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig(path)
    plt.close(fig)
