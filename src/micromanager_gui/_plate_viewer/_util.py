from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
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
BLUE = "#3776A1"
GENOTYPE_MAP = "genotype_plate_map.json"
TREATMENT_MAP = "treatment_plate_map.json"
COND1 = "condition_1"
COND2 = "condition_2"

# ----------------------------Measurement to compile in CSV---------------------------
# Each metric here will be pulled/calculated from the analysis data and compile into
# a CSV at the end of the analysis.
COMPILE_METRICS = [
    "amplitude",
    "frequency",
    "cell_size",
    "linear_connectivity",
    "cubic_connectivity",
    "iei",
    "percentage_active",
]

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

DEC_DFF_AMPLITUDE_VS_FREQUENCY_ALL = "Deconvolved DeltaF/F0 Amplitudes vs Frequencies"
DEC_DFF_AMPLITUDE_ALL = "Deconvolved DeltaF/F0 Amplitudes"
DEC_DFF_FREQUENCY_ALL = "Deconvolved DeltaF/F0 Frequencies"

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
]

MULTI_WELL_COMBO_OPTIONS = [
    DEC_DFF_AMPLITUDE_VS_FREQUENCY_ALL,
    DEC_DFF_AMPLITUDE_ALL,
    DEC_DFF_FREQUENCY_ALL,
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
    iei: list[float] | None = None
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

    phase_list = [float(phase) for phase in phases]

    return phase_list


def get_connectivity(phase_dict: dict[str, list[float]]) -> float | None:
    """Calculate the connection matrix."""
    connection_matrix = _get_connectivity_matrix(phase_dict)

    if connection_matrix is None or connection_matrix.size == 0:
        return None

    # Ensure the matrix is at least 2x2 and square
    if connection_matrix.shape[0] < 2 or (
        connection_matrix.shape[0] != connection_matrix.shape[1]
    ):
        return None

    # Compute mean connectivity
    mean_connect = float(
        np.median(np.sum(connection_matrix, axis=0) - 1)
        / (connection_matrix.shape[0] - 1)
    )

    return mean_connect


def _get_connectivity_matrix(phase_dict: dict[str, list[float]]) -> np.ndarray | None:
    """Calculate global connectivity using vectorized operations."""
    active_rois = list(phase_dict.keys())  # ROI names

    if len(active_rois) < 2:
        return None

    # Convert phase_dict values into a NumPy array of shape (N, T)
    phase_array = np.array([phase_dict[roi] for roi in active_rois])  # Shape (N, T)

    # Compute pairwise phase difference using broadcasting (Shape: (N, N, T))
    phase_diff = np.expand_dims(phase_array, axis=1) - np.expand_dims(
        phase_array, axis=0
    )

    # Ensure phase difference is within valid range [0, 2π]
    phase_diff = np.mod(np.abs(phase_diff), 2 * np.pi)

    # Compute cosine and sine of the phase differences
    cos_mean = np.mean(np.cos(phase_diff), axis=2)  # Shape: (N, N)
    sin_mean = np.mean(np.sin(phase_diff), axis=2)  # Shape: (N, N)

    # Compute synchronization index (vectorized)
    connect_matrix = np.array(np.sqrt(cos_mean**2 + sin_mean**2))

    return connect_matrix


def get_iei(peaks: list[int], exposure_time: float) -> list[float] | None:
    """Calculate the interevent interval."""
    # calculate framerate in Hz
    framerate = 1 / (exposure_time / 1000)

    # if less than 2 peaks or framerate is negative
    if len(peaks) < 2 or framerate <= 0:
        return None

    # calculate the difference in time between two consecutive peaks
    iei_frames = np.diff(np.array(peaks))
    iei = [float(iei_frame / framerate) for iei_frame in iei_frames]  # s

    return iei


# ----------------------------code to compile data------------------------------------
def compile_data_to_csv(
    analysis_data: dict[str, dict],
    plate_map: dict[str, dict[str, str]],
    save_path: str,
    col_per_treatment: int = 12,
) -> None:
    """Compile the data from analysis data into a CSV."""
    condition_1_list, condition_2_list = _compile_conditions(plate_map)
    fov_data_by_metric, cell_size_unit = _compile_per_metric(analysis_data, plate_map)
    _output_csv(
        fov_data_by_metric,
        condition_1_list,
        condition_2_list,
        cell_size_unit,
        save_path,
    )


def _compile_per_metric(
    analysis_data: dict[str, dict], plate_map: dict[str, dict[str, str]]
) -> tuple[list[dict[str, dict[str, list[float]]]], str]:
    """Group the FOV data of all the output parameter into a giant list."""
    # data_by_metrics: list[dict[str, dict[str, dict[str, list[float]]]]] = []
    data_by_metrics: list[dict[str, dict[str, list[float]]]] = []
    #               data metrix = []

    for output in COMPILE_METRICS:  # noqa: B007
        data_by_metrics.append({})

    wells_in_plate_map = list(plate_map.keys())

    # TODO: if no plate map:

    for fov_name, fov_dict in analysis_data.items():
        well = fov_name.split("_")[0]

        # TODO: if well not in the plate map
        if well not in wells_in_plate_map:
            continue

        condition_1 = plate_map[well][COND1]
        condition_2 = plate_map[well][COND2]

        for output_dict in data_by_metrics:
            if condition_1 not in output_dict:
                output_dict[condition_1] = {}
            if condition_2 not in output_dict[condition_1]:
                output_dict[condition_1][condition_2] = []

        data_per_fov_dict, cell_size_unit = _compile_data_per_fov(fov_dict)

        for i, output in enumerate(COMPILE_METRICS):
            output_value = data_per_fov_dict[output]
            data_by_metrics[i][condition_1][condition_2].append(output_value)

    return data_by_metrics, cell_size_unit


def _compile_data_per_fov(
    fov_dict: dict[str, ROIData],
) -> tuple[dict[str, float], str]:
    """Compile FOV data from all ROI data."""
    # compile data for one fov
    data_per_fov_dict: dict[str, float] = {}

    for measurement in COMPILE_METRICS:
        if measurement not in data_per_fov_dict:
            data_per_fov_dict[measurement] = 0

    amplitude_list_fov: list[float | None] = []
    cell_size_list_fov: list[float | None] = []
    frequency_list_fov: list[float | None] = []
    iei_list_fov: list[float | None] = []
    active_cells: int = 0
    cell_size_unit: str = ""
    cubic_phases: dict[str, list[float]] = {}
    linear_phases: dict[str, list[float]] = {}

    for roi_name, roiData in fov_dict.items():
        if isinstance(roiData, ROIData) and roiData.active:
            # cell size
            cell_size_list_fov.append(roiData.cell_size)
            if len(cell_size_unit) < 1:
                cell_size_unit = (
                    roiData.cell_size_units
                    if isinstance(roiData.cell_size_units, str)
                    else ""
                )

            # amplitude
            avg_amp_roi = (
                np.mean(roiData.peaks_amplitudes_dec_dff, dtype=np.float64)
                if isinstance(roiData.peaks_amplitudes_dec_dff, list)
                else np.nan
            )
            amplitude_list_fov.append(avg_amp_roi)

            # frequency
            frequency_list_fov.append(roiData.dec_dff_frequency)

            # iei
            avg_iei_roi = (
                np.mean(roiData.iei, dtype=np.float64)
                if isinstance(roiData.iei, list)
                else np.nan
            )
            iei_list_fov.append(avg_iei_roi)

            # global connectivity
            if roiData.cubic_phase and len(roiData.cubic_phase) > 0:
                cubic_phases[roi_name] = roiData.cubic_phase

            if roiData.linear_phase and len(roiData.linear_phase) > 0:
                linear_phases[roi_name] = roiData.linear_phase

            # activity
            active_cells += 1

    avg_amp_fov = _safe_mean(amplitude_list_fov)
    avg_cell_size_fov = _safe_mean(cell_size_list_fov)
    avg_frequency_fov = _safe_mean(frequency_list_fov)
    avg_iei_fov = _safe_mean(iei_list_fov)

    cubic_connectivity = get_connectivity(cubic_phases)
    linear_connectivity = get_connectivity(linear_phases)
    percentage_active = float(active_cells / len(list(fov_dict.keys())) * 100)

    # NOTE: if adding more output measurements,
    # make sure to check that the keys are in the COMPILED_METRICS
    data_per_fov_dict["amplitude"] = avg_amp_fov
    data_per_fov_dict["frequency"] = avg_frequency_fov
    data_per_fov_dict["cell_size"] = avg_cell_size_fov
    data_per_fov_dict["iei"] = avg_iei_fov
    data_per_fov_dict["percentage_active"] = percentage_active
    data_per_fov_dict["cubic_connectivity"] = (
        cubic_connectivity if isinstance(cubic_connectivity, list) else np.nan
    )
    data_per_fov_dict["linear_connectivity"] = (
        linear_connectivity if isinstance(linear_connectivity, list) else np.nan
    )

    return data_per_fov_dict, cell_size_unit


def _compile_conditions(
    plate_map_data: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    # TODO: what if one of the condition is missing?
    condition_1_list = list({value[COND1] for value in plate_map_data.values()})
    condition_2_list = list({value[COND2] for value in plate_map_data.values()})
    return condition_1_list, condition_2_list


def _output_csv(
    compiled_data_list: list,
    condition_1_list: list,
    condition_2_list: list,
    cell_size_unit: str,
    save_path: str,
    col_per_treatment: int = 12,
) -> None:
    """Save csv files of the data."""
    # logger.info("Saving output files")
    exp_name = Path(save_path).parent.name

    if compiled_data_list is None:
        return None

    for readout, readout_data in zip(COMPILE_METRICS, compiled_data_list):
        if readout == "cell_size":
            file_path = Path(save_path) / f"{exp_name}_{readout}_{cell_size_unit}.xlsx"
        else:
            file_path = Path(save_path) / f"{exp_name}_{readout}.xlsx"
        with xlsxwriter.Workbook(file_path, {"nan_inf_to_errors": True}) as wkbk:
            wkst = wkbk.add_worksheet(readout)
            num_format = wkbk.add_format({"num_format": "0.00"})
            wkst.write(0, 0, readout)

            if len(condition_2_list) > 0:
                # write conditions
                for i, condition in enumerate(condition_2_list):
                    for repeat in range(col_per_treatment):
                        wkst.write(0, i * col_per_treatment + repeat + 1, condition)

            # write genotypes
            for i, condition_1 in enumerate(condition_1_list):
                cond1 = condition_1
                if condition_1.lower() == "crispr":
                    cond1 = "+/+"
                elif condition_1.lower() == "patient":
                    cond1 = "+/-"
                elif condition_1.lower() == "null":
                    cond1 = "-/-"

                wkst.write(i + 1, 0, cond1)

            for condition1, cond_data in readout_data.items():
                for condition2, data_list in cond_data.items():
                    for i in range(col_per_treatment):
                        try:
                            start = condition_2_list.index(condition2)
                            row = condition_1_list.index(condition1) + 1
                        except ValueError:
                            start = 0
                            row = 5

                        if i < len(data_list):
                            entry = data_list[i]
                            if entry == "N/A":
                                wkst.write(
                                    row, start * col_per_treatment + i + 1, entry
                                )
                            else:
                                wkst.write_number(
                                    row,
                                    start * col_per_treatment + i + 1,
                                    float(entry),
                                    num_format,
                                )
                        else:
                            entry = "N/A"
                            wkst.write(row, start * col_per_treatment + i + 1, entry)


def _safe_mean(data_list: list) -> float:
    if len(data_list) < 1:
        return np.nan

    try:
        data_list_cleaned = [float(v) for v in data_list if v is not None]
        return (
            np.mean(data_list_cleaned, dtype=np.float64)
            if data_list_cleaned
            else np.nan
        )
    except (ValueError, TypeError):
        return np.nan
