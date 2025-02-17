from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tifffile
import useq
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
from scipy.signal import find_peaks
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._logger import LOGGER
from ._util import (
    COND1,
    COND2,
    GENOTYPE_MAP,
    GREEN,
    RED,
    TREATMENT_MAP,
    ROIData,
    _ElapsedTimer,
    _WaitingProgressBarWidget,
    calculate_dff,
    get_connectivity,
    get_connectivity_matrix,
    get_cubic_phase,
    get_linear_phase,
    parse_lineedit_text,
    show_error_dialog,
)

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

ELAPSED_TIME_KEY = "ElapsedTime-ms"
CAMERA_KEY = "camera_metadata"


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

        LOGGER.info("Number of positions: %s", len(pos))

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
            LOGGER.error("No data or labels path provided!")
            show_error_dialog(self, "No data or labels path provided!")
            return None

        sequence = self.data.sequence
        if sequence is None:
            LOGGER.error("No useq.MDAsequence found!")
            show_error_dialog(self, "No useq.MDAsequence found!")
            return None

        if self._plate_viewer is not None:
            tr_map = self._plate_viewer._plate_map_treatment.value()
            gen_map = self._plate_viewer._plate_map_genotype.value()
            # if both plate map genotype and treatment are not set, ask the user if
            # they want to continue without the plate map
            if not gen_map and not tr_map:
                msg = "The Plate Map is not set!\n\nDo you want to continue?"
                response = self._plate_map_msgbox(msg)
                if response == QMessageBox.StandardButton.No:
                    return None
            # if only one of the plate map genotype or treatment is set, ask the user
            # if they want to continue without both the plate maps
            elif (gen_map and not tr_map) or not gen_map:
                map_type = "Genotype" if gen_map else "Treatment"
                msg = (
                    f"Only the '{map_type}' Plate Map is "
                    "set!\n\nDo you want to continue without both the Plate "
                    "Maps?"
                )
                response = self._plate_map_msgbox(msg)
                if response == QMessageBox.StandardButton.No:
                    return None

        if path := self._output_path.value():
            save_path = Path(path)
            if not save_path.is_dir():
                LOGGER.error("Output Path is not a directory!")
                show_error_dialog(self, "Output Path is not a directory!")
                return None
            # create the save path if it does not exist
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
                # create bleach correction error path
                self._bleach_error_path = save_path / "bleach_correction_error"
                self._bleach_error_path.mkdir(parents=True, exist_ok=True)
        else:
            LOGGER.error("No Output Path provided!")
            show_error_dialog(self, "No Output Path provided!")
            return None

        # if the input is empty, return all positions that have labels. this can speed
        # up analysis since we will be sure that the labels exist for the positions.
        if not self._pos_le.text():
            positions: list[int] = []
            pos = list(sequence.stage_positions)
            for i, p in enumerate(pos):
                well = p.name or f"pos_{str(i).zfill(4)}"
                label = f"{well}_p{i}.tif"
                if self._get_labels_file(label) is None:
                    continue
                positions.append(i)
        else:
            # parse the input positions
            positions = parse_lineedit_text(self._pos_le.text())
            if not positions:
                show_error_dialog(self, "Invalid Positions provided!")
                return None
            if max(positions) >= len(sequence.stage_positions):
                show_error_dialog(self, "Input Positions out of range!")
                return None
        LOGGER.info("Positions to analyze: %s", positions)
        return positions

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
        LOGGER.info("Extraction of traces finished.")

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

        condition_1_plate_map = self._plate_viewer._plate_map_genotype.value()
        conition_2_plate_map = self._plate_viewer._plate_map_treatment.value()

        # save plate map
        LOGGER.info("Saving Plate Maps.")
        if condition_1_plate_map:
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
        for data in condition_1_plate_map:
            self._plate_map_data[data.name] = {COND1: data.condition[0]}

        for data in conition_2_plate_map:
            if data.name in self._plate_map_data:
                self._plate_map_data[data.name][COND2] = data.condition[0]
            else:
                self._plate_map_data[data.name] = {COND2: data.condition[0]}

    def _extract_traces(self, positions: list[int]) -> None:
        """Extract the roi traces in multiple threads."""
        LOGGER.info("Starting traces extraction...")

        # save plate maps and update the stored _plate_map_data dict
        self._handle_plate_map()

        cpu_count = os.cpu_count() or 1
        cpu_count = max(1, cpu_count - 2)  # leave a couple of cores for the system
        pos = len(positions)
        chunk_size = max(1, pos // cpu_count)

        LOGGER.info("CPU count: %s", cpu_count)
        LOGGER.info("Chunk size: %s", chunk_size)

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
                        LOGGER.info("Abort requested, cancelling all futures.")
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        future.result()
                        LOGGER.info(f"Chunk {idx + 1} completed.")
                    except Exception as e:
                        LOGGER.error("An error occurred in a chunk: %s", e)
                        show_error_dialog(self, f"An error occurred in a chunk: {e}")
                        break

            LOGGER.info("All tasks completed.")

        except Exception as e:
            LOGGER.error("An error occurred: %s", e)
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
        labels_path = self._get_labels_file(labels_name)
        if labels_path is None:
            LOGGER.error("No labels found for %s!", labels_name)
            print(f"No labels found for {labels_name}!")
            return
        labels = tifffile.imread(labels_path)

        # get the range of labels
        labels_range = range(1, labels.max())

        # create masks for each label
        masks = {label_value: (labels == label_value) for label_value in labels_range}

        LOGGER.info("Processing Well %s", well)

        seq = cast(useq.MDASequence, self.data.sequence)
        timepoints = seq.sizes["t"]
        exp_time = meta[0][event_key].get("exposure")
        elapsed_time: list = []

        # get the elapsed time for each timepoint to calculate tot_time_sec
        if (cam_key := CAMERA_KEY) in meta[0]:  # new metadata format
            for m in meta:
                et = m[cam_key].get(ELAPSED_TIME_KEY)
                if et is not None:
                    elapsed_time.append(float(et))
        else:  # old metadata format
            for m in meta:
                et = m.get(ELAPSED_TIME_KEY)
                if et is not None:
                    elapsed_time.append(float(et))
        # if the len of elapsed time is not equal to the number of timepoints,
        # use exposure time and the number of timepoints to calculate tot_time_sec
        if len(elapsed_time) != timepoints:
            tot_time_sec = exp_time * timepoints / 1000
        else:
            # otherwise, calculate the total time in seconds using the elapsed time.
            # NOTE: adding the exposure time to consider the first frame
            tot_time_sec = (elapsed_time[-1] - elapsed_time[0] + exp_time) / 1000

        roi_trace: np.ndarray
        linear_phase_dict: dict[str, list[float]] | None = {}
        cubic_phase_dict: dict[str, list[float]] | None = {}

        # extract roi traces
        LOGGER.info(f"Extracting Traces from Well {well}.")
        for label_value, mask in tqdm(
            masks.items(), desc=f"Extracting Traces from Well {well}"
        ):
            if self._check_for_abort_requested():
                break

            # calculate the mean trace for the roi
            masked_data = data[:, mask]

            # get the size of the roi in µm or px if µm is not available
            roi_size_pixel = masked_data.shape[1]  # area
            px_size = meta[0].get("PixelSizeUm", None)
            # calculate the size of the roi in µm if px_size is available or not 0,
            # otherwise use the size is in pixels
            roi_size = roi_size_pixel * px_size if px_size else roi_size_pixel

            # exclude small rois, might not be necessary if trained cellpose performs
            # better
            if px_size and roi_size < 10:
                continue

            # compute the mean for each frame
            roi_trace = cast(np.ndarray, masked_data.mean(axis=1))

            # CALCULATE DELATA F OVER F ----------------------------------------------
            dff = calculate_dff(roi_trace, window=10, plot=False)

            # DECONVOLVE DFF --------------------------------------------------------
            dec_dff, spikes, _, k, _ = deconvolve(dff, penalty=1)

            # FIND PEAKS ------------------------------------------------------------
            # Get the prominence:
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
            noise_level_dec_dff = (
                np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
            )
            peaks_prominence_dec_dff = noise_level_dec_dff * 2

            # find the peaks in the dec_dff trace
            peaks_dec_dff, _ = find_peaks(dec_dff, prominence=peaks_prominence_dec_dff)

            # get the amplitudes of the peaks in the dec_dff trace
            peaks_amplitudes_dec_dff = [dec_dff[p] for p in peaks_dec_dff]

            # calculate the frequency of the peaks in the dec_dff trace
            try:
                frequency = len(peaks_dec_dff) / tot_time_sec  # in Hz
            except ZeroDivisionError:
                frequency = 0

            # get the conditions for the roi if available
            condition_1 = condition_2 = None
            if self._plate_map_data:
                well_name = well.split("_")[0]
                if well_name in self._plate_map_data:
                    condition_1 = self._plate_map_data[well_name].get(COND1)
                    condition_2 = self._plate_map_data[well_name].get(COND2)
                else:
                    condition_1 = condition_2 = None

            # get the linear and cubic phase if there are at least 2 peaks
            if len(peaks_dec_dff) >= 2:
                linear_phase = get_linear_phase(timepoints, peaks_dec_dff)
                cubic_phase = get_cubic_phase(timepoints, peaks_dec_dff)
                if linear_phase is not None:
                    linear_phase_dict[str(label_value)] = linear_phase
                if cubic_phase is not None:
                    cubic_phase_dict[str(label_value)] = cubic_phase

            # store the analysis data
            self._analysis_data[well][str(label_value)] = ROIData(
                raw_trace=cast(list[float], roi_trace.tolist()),
                dff=cast(list[float], dff.tolist()),
                dec_dff=cast(list[float], dec_dff.tolist()),
                peaks_dec_dff=cast(list[float], peaks_dec_dff.tolist()),
                peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
                peaks_prominence_dec_dff=peaks_prominence_dec_dff,
                dec_dff_frequency=frequency,  # in Hz
                inferred_spikes=cast(list[float], spikes.tolist()),
                cell_size=roi_size,
                cell_size_units="µm" if px_size is not None else "pixel",
                condition_1=condition_1,
                condition_2=condition_2,
                total_recording_time_in_sec=tot_time_sec,
                active=len(peaks_dec_dff) > 0,
                linear_phase=linear_phase or None,
                cubic_phase=cubic_phase or None,
            )

        # calculate connectivity
        cubic_fig_path = Path(self._output_path.value()) / f"{well}_cubic.png"
        linear_fig_path = Path(self._output_path.value()) / f"{well}_linear.png"
        cubic_connectivity_matrix = get_connectivity_matrix(
            cubic_phase_dict, cubic_fig_path
        )
        linear_connectivity_matrix = get_connectivity_matrix(
            linear_phase_dict, linear_fig_path
        )
        cubic_mean_global_connectivity = get_connectivity(cubic_connectivity_matrix)
        linear_mean_global_connectivity = get_connectivity(linear_connectivity_matrix)
        self._analysis_data[well]["cubic global connectivity"] = (
            cubic_mean_global_connectivity
        )
        self._analysis_data[well]["linear global connectivity"] = (
            linear_mean_global_connectivity
        )

        # save json file
        LOGGER.info("Saving JSON file for Well %s.", well)
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
