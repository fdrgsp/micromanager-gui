from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from fonticon_mdi6 import MDI6
from oasis.functions import deconvolve
from qtpy.QtCore import QSize, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QGroupBox,
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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._util import (
    COND1,
    COND2,
    GENOTYPE_MAP,
    GREEN,
    RED,
    TREATMENT_MAP,
    Peaks,
    ROIData,
    _ElapsedTimer,
    _WaitingProgressBarWidget,
    parse_lineedit_text,
    show_error_dialog,
)

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
DFF_WINDOW = 100
R_SQUARED_THRESHOLD = 0.98


logger = logging.getLogger("analysis_logger")
logger.setLevel(logging.DEBUG)
log_file = Path(__file__).parent / "analysis_logger.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def single_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.array(a * np.exp(-b * x) + c)


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

        self._plate_map_data: dict[str, dict[str, str]] = {}

        self._data: TensorstoreZarrReader | OMEZarrReader | None = data

        self._labels_path: str | None = labels_path

        self._analysis_data: dict[str, dict[str, ROIData]] = {}

        self._bleach_error_path: Path | None = None

        self._worker: GeneratorWorker | None = None

        self._cancelled: bool = False

        pos_wdg = QWidget(self)
        pos_wdg.setToolTip(
            "Select the Positions to analyze. Leave blank to analyze all Positions. "
            "You can input single Positions (e.g. 30, 33) a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65). "
            "NOTE: The Positions are 0-indexed."
        )
        pos_wdg_layout = QHBoxLayout(pos_wdg)
        pos_wdg_layout.setContentsMargins(0, 0, 0, 0)
        pos_wdg_layout.setSpacing(5)
        pos_lbl = QLabel("Analyze Positions:")
        pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit()
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33")
        pos_wdg_layout.addWidget(pos_lbl)
        pos_wdg_layout.addWidget(self._pos_le)

        self._output_path = _BrowseWidget(
            self,
            "Analysis Output Path",
            "",
            "Select the output path for the Analysis Data.",
            is_dir=True,
        )
        pos_lbl.setFixedWidth(self._output_path._label.sizeHint().width())

        progress_wdg = QWidget(self)
        progress_wdg_layout = QHBoxLayout(progress_wdg)
        progress_wdg_layout.setContentsMargins(0, 0, 0, 0)

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        self._run_btn.setIconSize(QSize(25, 25))
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))
        self._cancel_btn.setIconSize(QSize(25, 25))
        self._cancel_btn.clicked.connect(self.cancel)

        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        progress_wdg_layout.addWidget(self._run_btn)
        progress_wdg_layout.addWidget(self._cancel_btn)
        progress_wdg_layout.addWidget(self._progress_bar)
        progress_wdg_layout.addWidget(self._progress_pos_label)
        progress_wdg_layout.addWidget(self._elapsed_time_label)

        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        self.progress_bar_updated.connect(self._update_progress_bar)

        self.groupbox = QGroupBox("Extract Traces", self)
        wdg_layout = QVBoxLayout(self.groupbox)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(self._output_path)
        wdg_layout.addWidget(pos_wdg)
        wdg_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

        self._cancel_waiting_bar = _WaitingProgressBarWidget(
            text="Stopping all the Tasks..."
        )

    @property
    def data(
        self,
    ) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader) -> None:
        self._data = data

    @property
    def labels_path(self) -> str | None:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, labels_path: str) -> None:
        self._labels_path = labels_path

    @property
    def analysis_data(self) -> dict[str, dict[str, ROIData]]:
        return self._analysis_data

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(event)

    def run(self) -> None:
        """Extract the roi traces in a separate thread."""
        pos = self._prepare_for_running()

        if pos is None:
            return

        logger.info("Number of positions: %s", len(pos))

        self._progress_bar.reset()
        self._progress_bar.setRange(0, len(pos))
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText(f"[0/{self._progress_bar.maximum()}]")

        # start elapsed timer
        self._elapsed_timer.start()

        self._cancelled = False

        self._enable(False)

        self._worker = create_worker(
            self._extract_traces,
            positions=pos,
            _start_thread=True,
            _connect={
                "finished": self._on_worker_finished,
                "errored": self._on_worker_finished,
            },
        )

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is None or not self._worker.is_running:
            return

        self._cancelled = True

        self._worker.quit()

        self._cancel_waiting_bar.start()

        # stop the elapsed timer
        self._elapsed_timer.stop()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def _prepare_for_running(self) -> list[int] | None:
        """Prepare the widget for running.

        Returns the number of positions or None if an error occurred.
        """
        if self._worker is not None and self._worker.is_running:
            return None

        if self.data is None or self._labels_path is None:
            logger.error("No data or labels path provided!")
            show_error_dialog(self, "No data or labels path provided!")
            return None

        sequence = self.data.sequence
        if sequence is None:
            logger.error("No useq.MDAsequence found!")
            show_error_dialog(self, "No useq.MDAsequence found!")
            return None

        if self._plate_viewer is not None and (
            not self._plate_viewer._plate_map_genotype.value()
            or not self._plate_viewer._plate_map_treatment.value()
        ):
            response = self._no_plate_map_msgbox()
            if response == QMessageBox.StandardButton.No:
                return None

        if path := self._output_path.value():
            save_path = Path(path)
            if not save_path.is_dir():
                logger.error("Output Path is not a directory!")
                show_error_dialog(self, "Output Path is not a directory!")
                return None
            # create the save path if it does not exist
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
                # create bleach correction error path
                self._bleach_error_path = save_path / "bleach_correction_error"
                self._bleach_error_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.error("No Output Path provided!")
            show_error_dialog(self, "No Output Path provided!")
            return None

        # return all positions if the input is empty
        if not self._pos_le.text():
            return list(range(len(sequence.stage_positions)))
        # parse the input positions
        positions = parse_lineedit_text(self._pos_le.text())
        if not positions:
            show_error_dialog(self, "Invalid Positions provided!")
            return None
        if max(positions) >= len(sequence.stage_positions):
            show_error_dialog(self, "Input Positions out of range!")
            return None
        return positions

    def _no_plate_map_msgbox(self) -> Any:
        """Show a message box to ask the user if wants to overwrite the labels."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("The Plate Map is not set!\n\nDo you want to continue?")
        msg.setWindowTitle("Plate Map")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        return msg.exec()

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._cancel_waiting_bar.setEnabled(True)
        self._pos_le.setEnabled(enable)
        self._output_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)
        if self._plate_viewer is None:
            return
        self._plate_viewer._plate_map_group.setEnabled(enable)
        self._plate_viewer._segmentation_wdg.setEnabled(enable)

    def _on_worker_finished(self) -> None:
        """Called when the extraction is finished."""
        logger.info("Extraction of traces finished.")

        self._enable(True)

        self._elapsed_timer.stop()
        self._cancel_waiting_bar.stop()

        self._bleach_error_path = None

        # update the analysis data of the plate viewer
        if self._plate_viewer is not None:
            self._plate_viewer.analysis_data = self._analysis_data
            self._plate_viewer._analysis_file_path = self._output_path.value()

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

    def _get_labels_file(self, label_name: str) -> str | None:
        """Get the labels file for the given name."""
        if self._labels_path is None:
            return None
        for label_file in Path(self._labels_path).glob("*.tif"):
            if label_file.name.endswith(label_name):
                return str(label_file)
        return None

    def _check_for_abort_requested(self) -> bool:
        return bool(self._worker is not None and self._worker.abort_requested)

    def _handle_plate_map(self) -> None:
        if self._plate_viewer is None:
            return

        conition_1_plate_map = self._plate_viewer._plate_map_genotype.value()
        conition_2_plate_map = self._plate_viewer._plate_map_treatment.value()

        # save plate map
        logger.info("Saving Plate Maps.")
        if conition_1_plate_map:
            path = Path(self._output_path.value()) / GENOTYPE_MAP
            with path.open("w") as f:
                json.dump(
                    self._plate_viewer._plate_map_genotype.value(),
                    f,
                    indent=2,
                )
        if conition_2_plate_map:
            path = Path(self._output_path.value()) / TREATMENT_MAP
            with path.open("w") as f:
                json.dump(
                    self._plate_viewer._plate_map_treatment.value(),
                    f,
                    indent=2,
                )

        self._plate_map_data.clear()

        # update the stored _plate_map_data dict so we have the condition for each well
        # name as the kek. eg.g:
        # {"A1": {"condition_1": "condition_1", "condition_2": "condition_2"}}
        for data in conition_1_plate_map:
            self._plate_map_data[data.name] = {COND1: data.condition[0]}

        for data in conition_2_plate_map:
            if data.name in self._plate_map_data:
                self._plate_map_data[data.name][COND2] = data.condition[0]
            else:
                self._plate_map_data[data.name] = {COND2: data.condition[0]}

    def _extract_traces(self, positions: list[int]) -> None:
        """Extract the roi traces in multiple threads."""
        logger.info("Starting traces extraction...")

        # save plate maps and update the stored _plate_map_data dict
        self._handle_plate_map()

        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)  # leave a couple of cores for the system
        pos = len(positions)
        chunk_size = max(1, pos // cpu_count)

        logger.info("CPU count: %s", cpu_count)
        logger.info("Chunk size: %s", chunk_size)

        try:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = [
                    executor.submit(
                        self._extract_trace_for_chunk,
                        positions,
                        start,
                        min(start + chunk_size, pos),
                    )
                    for start in range(0, pos, chunk_size)
                ]

                for idx, future in enumerate(as_completed(futures)):
                    if self._check_for_abort_requested():
                        logger.info("Abort requested, cancelling all futures.")
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                        logger.info(f"Chunk {idx + 1} completed.")
                    except Exception as e:
                        logger.error("An error occurred in a chunk: %s", e)
                        show_error_dialog(self, f"An error occurred in a chunk: {e}")
                        break

            logger.info("All tasks completed.")

        except Exception as e:
            logger.error("An error occurred: %s", e)
            show_error_dialog(self, f"An error occurred: {e}")

    def _extract_trace_for_chunk(
        self, positions: list[int], start: int, end: int
    ) -> None:
        """Extract the roi traces for the given chunk."""
        for p in range(start, end):
            if self._check_for_abort_requested():
                break
            self._extract_trace_per_position(positions[p])

    def _extract_trace_per_position(self, p: int) -> None:
        """Extract the roi traces for the given position."""
        if self.data is None or self._check_for_abort_requested():
            return

        data, meta = self.data.isel(p=p, metadata=True)

        # get position name from metadata
        # the "Event" key was used in the old metadata format
        event_key = "mda_event" if "mda_event" in meta[0] else "Event"
        well = meta[0].get(event_key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")

        # create the dict for the well
        if well not in self._analysis_data:
            self._analysis_data[well] = {}

        # matching label name
        labels_name = f"{well}_p{p}.tif"

        # get the labels file
        labels = tifffile.imread(self._get_labels_file(labels_name))
        if labels is None:
            logger.error("No labels found for %s!", labels_name)
            show_error_dialog(self, f"No labels found for {labels_name}!")
            return

        # get the range of labels
        labels_range = range(1, labels.max())

        # create masks for each label
        masks = {label_value: (labels == label_value) for label_value in labels_range}

        logger.info("Processing Well %s", well)

        # calculate exponential decay for photobleaching correction:
        # first get the average trace for the position, fit an exponential decay
        # and use the fitted curve for photobleaching correction for each roi
        logger.info(f"Calculating Exponential Decay for well {well}.")
        average_trace = data.mean(axis=(1, 2))
        exponential_decay = self._get_exponential_decay(average_trace, well)
        if exponential_decay is None:
            logger.error(
                f"Exponential Decay could not be calculated for Well {well} - pos{p}."
            )
        else:
            average_fitted_curve, average_popts, _ = exponential_decay

        roi_trace: np.ndarray | list[float] | None

        # extract roi traces
        logger.info(f"Extracting Traces from Well {well}.")
        for label_value, mask in tqdm(
            masks.items(), desc=f"Extracting Traces from Well {well}"
        ):
            if self._check_for_abort_requested():
                break

            # calculate the mean trace for the roi
            masked_data = data[:, mask]

            # compute the mean for each frame
            roi_trace = cast(np.ndarray, masked_data.mean(axis=1))

            # get the size of the roi in µm or px if µm is not available
            roi_size_pixel = masked_data.shape[1]  # area
            px_size = meta[0].get("PixelSizeUm", None)
            # calculate the size of the roi in µm if px_size is available or not 0,
            # otherwise use the size is in pixels
            roi_size = roi_size_pixel * px_size if px_size else roi_size_pixel

            # get the conditions for the roi if available
            condition_1 = condition_2 = None
            if self._plate_map_data:
                well_name = well.split("_")[0]
                if well_name in self._plate_map_data:
                    condition_1 = self._plate_map_data[well_name].get(COND1)
                    condition_2 = self._plate_map_data[well_name].get(COND2)
                else:
                    condition_1 = condition_2 = None

            # store the analysis data
            self._analysis_data[well][str(label_value)] = ROIData(
                raw_trace=roi_trace.tolist(),
                use_for_bleach_correction=exponential_decay,
                cell_size=roi_size,
                cell_size_units="µm" if px_size is not None else "pixel",
                condition_1=condition_1,
                condition_2=condition_2,
            )

        # perform photobleaching correction
        logger.info(f"Performing Belaching Correction for Well {well}.")
        for label_value in tqdm(
            labels_range, desc=f"Performing Belaching Correction for Well {well}"
        ):
            if self._check_for_abort_requested():
                break

            data = self._analysis_data[well][str(label_value)]

            roi_trace = data.raw_trace

            if roi_trace is None:
                continue

            # calculate the bleach corrected trace
            if exponential_decay is not None:
                bleach_corrected = (
                    np.array(roi_trace) - average_fitted_curve + average_popts[2]
                )
            # if no exponential decay was calculated, use the raw trace
            else:
                bleach_corrected = np.array(roi_trace)

            # calculate the dF/F
            dff = self._calculate_dff(bleach_corrected, window=DFF_WINDOW)
            d_dff, _, _, _, _ = deconvolve(dff, g=(None, None), penalty=1)

            # find the peaks in the bleach corrected trace
            prominence = np.mean(d_dff) * 0.2  # 20% of the mean of the derivative
            peaks = self._find_peaks(d_dff, prominence=prominence)

            # store the analysis data
            update = data.replace(
                average_photobleaching_fitted_curve=average_fitted_curve,
                average_popts=average_popts,
                bleach_corrected_trace=bleach_corrected.tolist(),
                peaks=[Peaks(peak=peak) for peak in peaks] if len(peaks) > 2 else None,
                dff=dff.tolist(),
                d_dff=d_dff.tolist(),
            )
            self._analysis_data[well][str(label_value)] = update

        # save json file
        logger.info("Saving JSON file for Well %s.", well)
        path = Path(self._output_path.value()) / f"{well}.json"
        with path.open("w") as f:
            json.dump(
                self._analysis_data[well],
                f,
                default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                indent=2,
            )

        # update the progress bar
        self.progress_bar_updated.emit()

    def _smooth_and_normalize(self, trace: np.ndarray) -> np.ndarray:
        """Smooth and normalize the trace between 0 and 1."""
        # smoothing that preserves the peaks
        smoothed = savgol_filter(trace, window_length=5, polyorder=2)
        # normalize the smoothed trace from 0 to 1
        return cast(
            np.ndarray,
            (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed)),
        )

    def _get_exponential_decay(
        self, trace: np.ndarray, well: str = ""
    ) -> tuple[list[float], list[float], float] | None:
        """Fit an exponential decay to the trace.

        Returns None if the R squared value is less than 0.9.
        """
        time_points = np.arange(len(trace))
        initial_guess = [max(trace), 0.01, min(trace)]
        try:
            popt, _ = curve_fit(
                single_exponential, time_points, trace, p0=initial_guess, maxfev=2000
            )
            fitted_curve = single_exponential(time_points, *popt)
            residuals = trace - fitted_curve
            r, _ = pearsonr(trace, fitted_curve)
            ss_total = np.sum((trace - np.mean(trace)) ** 2)
            ss_res = np.sum(residuals**2)
            r_squared = 1 - (ss_res / ss_total)
        except Exception as e:
            logger.error("Error fitting curve: %s", e)
            return None

        # save the fitted curve if the R squared value is less than 0.98
        if r_squared <= 0.98 and self._bleach_error_path is not None and well:
            plt.plot(fitted_curve, "black", "--")
            plt.plot(trace, "green")
            plt.savefig(
                self._bleach_error_path
                / f"failed_fitted_curve_r2{r_squared}_{well}.png"
            )

        return (
            None
            if r_squared <= R_SQUARED_THRESHOLD
            else (fitted_curve.tolist(), popt.tolist(), float(r_squared))
        )

    def _calculate_dff(self, trace: np.ndarray, window: int = 100) -> np.ndarray:
        """Calculate the ΔF/F (delta F over F) signal for a given calcium trace.

        Parameters
        ----------
        trace : np.ndarray
            The input signal as a 1D numpy array.
        window : int, optional
            The size of the sliding window for background calculation, by default 100.
        """
        background, _ = self._calculate_bg(trace, window)
        dff = (trace - background) / background
        dff -= np.min(dff)  # Shift the signal to ensure non-negative values.
        return np.array(dff)

    def _calculate_bg(
        self, trace: np.ndarray, window: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the background and median trace for a given signal...

        ...using a sliding window approach.

        Parameters
        ----------
        trace : np.ndarray
            The input signal as a 1D numpy array.
        window : int
            The size of the sliding window for background calculation.
        """
        n = len(trace)
        background = np.zeros_like(trace)
        median = np.zeros_like(trace)

        for i in range(n):
            start = max(0, i - window)
            window_slice = trace[start : i + 1]
            median_value = np.median(window_slice)
            lower_quantile = window_slice <= median_value
            background[i] = np.mean(window_slice[lower_quantile])
            median[i] = median_value

        return background, median

    def _find_peaks(
        self, trace: np.ndarray, prominence: float | None = None
    ) -> list[int]:
        """Smooth the trace and find the peaks."""
        smoothed_normalized = self._smooth_and_normalize(trace)
        # find the peaks # TODO: find the necessary parameters to use
        peaks, _ = find_peaks(smoothed_normalized, width=3, prominence=prominence)
        peaks = cast(np.ndarray, peaks)
        return cast(list[int], peaks.tolist())
