from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._util import (
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

    from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
    from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


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

        self._labels_path: str | None = labels_path

        self._analysis_data: dict[str, dict[str, ROIData]] = {}

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

        # self._output_path = _SelectAnalysisPath(
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
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
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
        self.groupbox.setCheckable(True)
        self.groupbox.setChecked(False)
        wdg_layout = QVBoxLayout(self.groupbox)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(self._output_path)
        wdg_layout.addWidget(pos_wdg)
        wdg_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

        self._cancel_waiting_bar = _WaitingProgressBarWidget(
            text="Stopping all the Tasks..."
        )

    @property
    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
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

        if self._data is None or self._labels_path is None:
            logger.error("No data or labels path provided!")
            show_error_dialog(self, "No data or labels path provided!")
            return None

        sequence = self._data.sequence
        if sequence is None:
            logger.error("No useq.MDAsequence found!")
            show_error_dialog(self, "No useq.MDAsequence found!")
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

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._pos_le.setEnabled(enable)
        self._output_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)

    def _on_worker_finished(self) -> None:
        """Called when the extraction is finished."""
        logger.info("Extraction of traces finished.")

        self._enable(True)

        self._elapsed_timer.stop()
        self._cancel_waiting_bar.stop()

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

    def _extract_traces(self, positions: list[int]) -> None:
        """Extract the roi traces in multiple threads."""
        logger.info("Starting traces extraction...")

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
        if self._data is None or self._check_for_abort_requested():
            return

        data, meta = self._data.isel(p=p, metadata=True)

        # get position name from metadata
        well = meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")
        
        total_frames = data.shape[0]
        binning, magnification, pixel_size, objective, exposure, framerate = self._extract_metadata(meta) 

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

        logger.info("Processing well %s", well)

        # temporary storage for trace to use for photobleaching correction
        fitted_curves: list[tuple[list[float], list[float], float]] = []

        roi_trace: np.ndarray | list[float] | None
        roi_size_um: float | None

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
            # print(f"            the shape of masked_data is: {masked_data.shape}")
            roi_size_pixel = masked_data.shape[1]
            roi_size_um = self._cell_size_in_um(roi_size_pixel, binning, pixel_size, objective, magnification)

            # calculate the exponential decay for photobleaching correction
            exponential_decay = self._get_exponential_decay(roi_trace)
            if exponential_decay is not None:
                fitted_curves.append(exponential_decay)

            # store the analysis data
            self._analysis_data[well][str(label_value)] = ROIData(
                raw_trace=roi_trace.tolist(),
                use_for_bleach_correction=exponential_decay,
                cell_size=roi_size_um
            )

        # average the fitted curves
        logger.info(f"Averaging the fitted curves well {well}.")
        popts = np.array([popt for _, popt, _ in fitted_curves])
        average_popts = np.mean(popts, axis=0)
        time_points = np.arange(data.shape[0])
        average_fitted_curve = single_exponential(time_points, *average_popts)

        # perform photobleaching correction
        logger.info(f"Performing Bleaching Correction for Well {well}.")
        for label_value in tqdm(
            labels_range, desc=f"Performing Bleaching Correction for Well {well}"
        ):
            if self._check_for_abort_requested():
                break

            data = self._analysis_data[well][str(label_value)] # for one ROI

            roi_trace = data.raw_trace

            if roi_trace is None:
                continue

            # calculate the bleach corrected trace
            bleach_corrected = (
                np.array(roi_trace) - average_fitted_curve + average_popts[2]
            )

            # calculate the dF/F TODO: how to calculate F0?
            F0 = np.min(bleach_corrected)
            dff = (bleach_corrected - F0) / F0

            prominence = np.mean(dff) * 0.35
            # find the peaks in the bleach corrected trace
            peaks = self._find_peaks(dff, prominence=prominence) # for one ROI

            # Peaks
            amplitudes, start, end, new_peaks = self._get_amplitude(dff, peaks)
            max_slopes = self._get_max_slope(dff, new_peaks, start)
            raise_time = self._get_raise_time(new_peaks, start, framerate)
            decay_time = self._get_decay_time(new_peaks, end, framerate)

            #ROIData
            iei = self._get_iei(start, framerate)
            mean_iei = np.mean(iei)
            mean_iei_stdev = np.std(iei)         
            mean_amplitude = np.mean(amplitudes)
            mean_amplitude_stdev = np.std(amplitudes)
            frequency = len(peaks) / (total_frames/framerate)
            mean_raise_time = np.mean(raise_time)
            mean_raise_time_stdev = np.std(raise_time)  
            mean_decay_time = np.mean(decay_time)
            mean_decay_time_stdev = np.std(decay_time)

            # store the analysis data
            update = data.replace(
                average_photobleaching_fitted_curve=average_fitted_curve.tolist(),
                average_popts=average_popts.tolist(),
                bleach_corrected_trace=bleach_corrected.tolist(),
                peaks=[Peaks(peak=new_peaks[i], 
                             amplitude=amplitudes[i],
                             max_slope=max_slopes[i],
                             raise_time=raise_time[i],
                             decay_time=decay_time[i]
                             ) for i in range(len(new_peaks))],
                mean_amplitude=mean_amplitude,
                mean_amplitude_stdev=mean_amplitude_stdev,
                frequency=frequency,
                mean_raise_time=mean_raise_time,
                mean_raise_time_stdev=mean_raise_time_stdev,
                mean_decay_time=mean_decay_time,
                mean_decay_time_stdev=mean_decay_time_stdev,
                mean_iei=mean_iei,
                mean_iei_stdev=mean_iei_stdev,
                dff=dff.tolist(),
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
        self, trace: np.ndarray
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

        return (
            None
            if r_squared <= 0.98
            else (fitted_curve.tolist(), popt.tolist(), float(r_squared))
        )

    def _find_peaks(
        self, trace: np.ndarray, prominence: float | None = None
    ) -> list[int]:
        """Smooth the trace and find the peaks."""
        smoothed_normalized = self._smooth_and_normalize(trace)
        peaks, _ = find_peaks(smoothed_normalized, width=3, prominence=prominence)
        peaks = cast(np.ndarray, peaks)
        return cast(list[int], peaks.tolist())

    # TODO: are there better ways to find the start and end of a peak?
    def old_get_amplitude(self, dff: list[float], peaks: list[Peaks], deriv_threhold=0.01,
                        reset_num=17, neg_reset_num=2, total_dist=40, min_dist=5
                        ) -> tuple[list[float], list[int], list[int], list[int]]:
        """Calculate amplitudes, peak indices, and base_indices of each ROI."""
        amplitudes = []
        start_indices = []
        end_indices = []
        remove_peaks = []
        
        if len(peaks) > 0:
            dff_deriv = np.diff(dff) # the difference of dff across frames

            for i in range(len(peaks)):
                searching = True
                under_thresh_count = 0
                total_count = 0
                start_index = peaks[i] # the frame for the first spike

                if start_index > 0:
                    while searching:
                        start_index -= 1
                        total_count += 1

                        # If collide with a new spike
                        if start_index in peaks:
                            subsearching = True
                            negative_count = 0
                            while subsearching:
                                start_index += 1
                                if start_index < len(dff_deriv):
                                    if dff_deriv[start_index] < 0:
                                        negative_count += 1
                                    elif peaks[i] - start_index <= min_dist:
                                        subsearching = False
                                    else:
                                        negative_count = 0
                                    if negative_count == neg_reset_num:
                                        subsearching = False
                                else:
                                    subsearching = False
                            break

                        # if the difference is below threshold
                        if dff_deriv[start_index] < deriv_threhold:
                            under_thresh_count += 1
                        else:
                            under_thresh_count = 0

                        # stop searching for starting index
                        if under_thresh_count >= reset_num or start_index == 0 or total_count == total_dist:
                            searching = False

                # Search for ending index for current spike
                searching = True
                under_thresh_count = 0
                total_count = 0
                end_index = peaks[i]

                if end_index < (len(dff_deriv) - 1):
                    while searching:
                        end_index += 1
                        total_count += 1

                        # If collide with a new spike
                        if end_index in peaks:
                            subsearching = True
                            negative_count = 0
                            while subsearching:
                                end_index -= 1
                                if dff_deriv[end_index] > 0:
                                    negative_count += 1
                                elif end_index - peaks[i] <= min_dist:
                                    subsearching = False
                                else:
                                    negative_count = 0
                                if negative_count == neg_reset_num:
                                    subsearching = False
                            break
                        if dff_deriv[end_index] < deriv_threhold:
                            under_thresh_count += 1
                        else:
                            under_thresh_count = 0

                        # NOTE: changed the operator from == to >=
                        if under_thresh_count >= reset_num or end_index == (len(dff_deriv) - 1) or \
                                total_count == total_dist:
                            searching = False

                # Save data
                # print(f"-----------------------------------------")

                spk_to_end = dff[peaks[i]:(end_index + 1)]
                start_to_spk = dff[start_index:peaks[i]]
                # print(f"            spk_to_end: {spk_to_end}")
                # print(f"        start to spk: {start_to_spk}")


                f_start_index = int(peaks[i] -(len(start_to_spk) - (np.argmin(start_to_spk) + 1)))
                f_end_index = int(peaks[i] + np.argmin(spk_to_end))
                amplitude = dff[peaks[i]]-dff[f_start_index]
                # print(f"    start_index: {f_start_index} of {dff[f_start_index]}")
                # print(f"            peak: {peaks[i]} of {dff[peaks[i]]}")                
                # print(f"       end_index: {f_end_index} of {dff[f_end_index]}")

                if amplitude > 0:
                    start_indices.append(f_start_index)
                    end_indices.append(f_end_index)
                    amplitudes.append(amplitude)
                else:
                    remove_peaks.append(peaks[i])
                    # print(f"            REMOVING {peaks[i]} because the amplitude is {amplitude}")
                
                # print(f"        amplitude: {amplitudes[-1]} from {start_indices[-1]} to {end_indices[-1]}")
        
        new_peaks = [peak for peak in peaks if (peak not in remove_peaks)]

        return amplitudes, start_indices, end_indices, new_peaks

    # ChatGPT-improved
    def _get_amplitude(self, dff: list[float], peaks: list[int], deriv_threshold=0.01,
                   reset_num=17, neg_reset_num=2, total_dist=40, min_dist=5
                   ) -> tuple[list[float], list[int], list[int], list[int]]:
        """Calculate amplitudes, peak indices, and base indices of each ROI."""
        amplitudes = []
        start_indices = []
        end_indices = []
        remove_peaks = []

        if peaks:
            dff_deriv = np.diff(dff)
            len_dff = len(dff)
            len_dff_deriv = len(dff_deriv)
            
            for peak in peaks:
                start_index = peak
                end_index = peak
                under_thresh_count = 0
                total_count = 0

                if start_index > 0:
                    while start_index > 0 and total_count < total_dist:
                        start_index -= 1
                        total_count += 1
                        if start_index in peaks:
                            negative_count = 0
                            while start_index < len_dff_deriv and dff_deriv[start_index] < 0 and negative_count < neg_reset_num:
                                start_index += 1
                                if dff_deriv[start_index] < 0:
                                    negative_count += 1
                                else:
                                    negative_count = 0
                            break
                        if dff_deriv[start_index] < deriv_threshold:
                            under_thresh_count += 1
                        else:
                            under_thresh_count = 0
                        if under_thresh_count >= reset_num:
                            break

                under_thresh_count = 0
                total_count = 0

                if end_index < len_dff_deriv - 1:
                    while end_index < len_dff_deriv - 1 and total_count < total_dist:
                        end_index += 1
                        total_count += 1
                        if end_index in peaks:
                            negative_count = 0
                            while end_index >= 0 and dff_deriv[end_index] > 0 and negative_count < neg_reset_num:
                                end_index -= 1
                                if dff_deriv[end_index] > 0:
                                    negative_count += 1
                                else:
                                    negative_count = 0
                            break
                        if dff_deriv[end_index] < deriv_threshold:
                            under_thresh_count += 1
                        else:
                            under_thresh_count = 0
                        if under_thresh_count >= reset_num:
                            break

                spk_to_end = dff[peak:(end_index + 1)]
                start_to_spk = dff[start_index:peak]
                f_start_index = int(peak - (len(start_to_spk) - (np.argmin(start_to_spk) + 1)))
                f_end_index = int(peak + np.argmin(spk_to_end))
                amplitude = dff[peak] - dff[f_start_index]

                if amplitude > 0:
                    start_indices.append(f_start_index)
                    end_indices.append(f_end_index)
                    amplitudes.append(amplitude)
                else:
                    remove_peaks.append(peak)

        new_peaks = [peak for peak in peaks if peak not in remove_peaks]

        return amplitudes, start_indices, end_indices, new_peaks

    def _get_max_slope(self, dff: list[float], peaks: list[int], bases: list[int]) -> list[float]:
        """Get max slope of each peak in one ROI."""
        max_slopes = []

        if len(peaks) > 0:
            for i in range(len(peaks)):
                peak_index = peaks[i]
                base_index = bases[i]

                # print(f"            peak index: {peak_index}, base_index: {base_index}")
                slope_window = dff[base_index:(peak_index + 1)]
                slope_window_a = slope_window[:-1]
                slope_window_b = slope_window[1:]
                max_slope = max([b-a for a,b in zip(slope_window_a, slope_window_b)])
                max_slopes.append(max_slope)
                # print(f"           max slope is {max_slope}")

        return max_slopes

    def _cell_size_in_um(self, cell_size_pixel: int, binning: int,
                         pixel_size: float, objective: int, magnification: float
                         )-> int:
        """Convert the cell size in pixel to um."""
        cell_size_um = cell_size_pixel * binning * pixel_size / (
            objective * magnification
        )

        return cell_size_um
    
    # NOTE: iei here is the time between the start index of each peak
    # the old is between each peak
    def _get_iei(self, start_indices: list[int], framerate: float) -> list[float]:
        """Calculate the interevent interval."""
        iei = []

        iei_frames = np.diff(np.array(start_indices))

        if framerate:
            iei.append(iei_frames/framerate)

        else:
            iei.append(iei_frames)

        return iei
    
    # NOTE: raise time here is from base to peak; FluoroSNNAO uses from base to max_slope point
    def _get_raise_time(self, peaks: list[int], start: list[int], framerate: float) -> list[float]:
        """Get Raise Time for each peak."""
        raise_time = [((peaks[i] - start[i] + 1)/framerate) for i in range(len(peaks))]

        return raise_time
    
    def _get_decay_time(self, peaks: list[int], end: list[int], framerate: float) -> list[float]:
        """Get decay time for each peak."""
        decay_time = [((end[i] - peaks[i] + 1)/ framerate) for i in range(len(peaks))]

        return decay_time
    
    def _extract_metadata(self, meta):
        """Extract information from metadata."""
        binning = int(meta[0].get('pco_camera-Binning'))
        magnification = float(meta[0].get('IntermediateMagnification-Magnification')[:-1])
        pixel_size = float(meta[0].get('PixelSizeUm'))
        objective = int(meta[0].get('Nosepiece-Label').split(' ')[-1][:-1])
        exposure = float(meta[0].get('Event').get('exposure'))
        framerate = 1 / exposure

        return binning, magnification, pixel_size, objective, exposure, framerate


