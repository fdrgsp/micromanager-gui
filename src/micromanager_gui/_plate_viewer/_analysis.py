from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tifffile
import useq
from fonticon_mdi6 import MDI6
from oasis.functions import deconvolve
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from ._analysis_gui import (
    AnalysisSettingsData,
    CalciumPeaksData,
    ExperimentTypeData,
    SpikeData,
    TraceExtractionData,
    _AnalysisProgressBarWidget,
    _CalciumAnalysisGUI,
)
from ._logger import LOGGER
from ._plate_map import PlateMapData
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
    DEFAULT_PEAKS_DISTANCE,
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
    _ElapsedTimer,
    _WaitingProgressBarWidget,
    calculate_dff,
    create_stimulation_mask,
    get_iei,
    get_overlap_roi_with_stimulated_area,
    mask_to_coordinates,
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


# Type alias for analysis settings where all fields are guaranteed to be non-None
# used in _get_validated_settings
@dataclass(frozen=True)
class ValidatedAnalysisSettings:
    """Analysis settings with all fields guaranteed to be non-None."""

    plate_map_data: tuple[list[PlateMapData], list[PlateMapData]]
    experiment_type_data: ExperimentTypeData
    trace_extraction_data: TraceExtractionData
    calcium_peaks_data: CalciumPeaksData
    spikes_data: SpikeData
    positions: str


class _AnalyseCalciumTraces(QWidget):
    """Widget to extract the roi traces from the data."""

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

        # MAIN GUI ----------------------------------------------------------------

        self._analysis_settings_gui = _CalciumAnalysisGUI(self)

        # PROGRESS BAR AND RUN/CANCEL BUTTONS -------------------------------------
        self._progress_bar_wdg = _AnalysisProgressBarWidget(self)
        self._pbar = self._progress_bar_wdg  # for easier access in the GUI

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))
        pbar_layout = cast(QHBoxLayout, self._progress_bar_wdg.layout())
        pbar_layout.insertWidget(0, self._run_btn)
        pbar_layout.insertWidget(1, self._cancel_btn)
        # add them to the analysis settings GUI layout
        gui_layout = cast(QVBoxLayout, self._analysis_settings_gui.layout())
        gui_layout.addSpacing(5)
        gui_layout.addWidget(self._progress_bar_wdg)

        # LAYOUT ----------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._analysis_settings_gui)

        # ELAPSED TIME TIMER ------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._pbar.set_time_label)

        # WAITING PROGRESS BAR -----------------------------------------------------
        self._cancel_waiting_bar = _WaitingProgressBarWidget(
            text="Stopping all the Tasks..."
        )

        # CONNECTIONS --------------------------------------------------------------
        self._progress_bar_wdg.updated.connect(self._progress_bar_wdg.auto_update)
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn.clicked.connect(self.cancel)

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
        value = self._analysis_settings_gui._experiment_type_wdg.value()
        return value.stimulation_area_path

    @stimulation_area_path.setter
    def stimulation_area_path(self, path: str | None) -> None:
        path = path or ""
        self._analysis_settings_gui.setValue(
            AnalysisSettingsData(
                experiment_type_data=ExperimentTypeData(stimulation_area_path=path)
            )
        )

    # PUBLIC METHODS ---------------------------------------------------------------

    def run(self) -> None:
        """Extract the roi traces in a separate thread."""
        # the connection to this method is done in _CalciumAnalysisGUI
        self._failed_labels.clear()

        pos = self._prepare_for_running()

        if pos is None:
            return

        LOGGER.info("Number of positions: %s", len(pos))

        self._pbar.reset()
        self._pbar.set_range(0, len(pos))
        self._pbar.set_time_label(f"[0/{self._pbar.maximum()}]")

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
        self._pbar.reset()
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

        value = self._get_validated_settings()
        tr_map, gen_map = value.plate_map_data

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
        value = self._get_validated_settings()
        exp_type = value.experiment_type_data.experiment_type
        return exp_type == EVOKED if exp_type else False

    def _prepare_stimulation_mask(self, analysis_path: Path) -> bool:
        """Generate the stimulation mask if the experiment involves evoked activity."""
        value = self._get_validated_settings()
        if stim_area_file := value.experiment_type_data.stimulation_area_path:
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

        pos = self._get_validated_settings().positions
        if not pos:
            positions = [
                i
                for i, p in enumerate(sequence.stage_positions)
                if self._get_labels_file(
                    f"{p.name or f'pos_{str(i).zfill(4)}'}_p{i}.tif"
                )
            ]
        else:
            positions = parse_lineedit_text(pos)
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

        value = self._get_validated_settings()
        condition_1_plate_map, condition_2_plate_map = value.plate_map_data
        # save plate map
        LOGGER.info("Saving Plate Maps.")
        path = Path(self._analysis_path) / GENOTYPE_MAP
        self._save_plate_map(path, condition_1_plate_map)
        path = Path(self._analysis_path) / TREATMENT_MAP
        self._save_plate_map(path, condition_2_plate_map)

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
        self._pbar.updated.emit()

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
        value = self._get_validated_settings()

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
        win = value.trace_extraction_data.dff_window_size
        # calculate the dff of the roi trace
        dff = calculate_dff(roi_trace, window=win, plot=False)

        # compute the decay constant
        tau = value.trace_extraction_data.decay_constant
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

        # Use the spike threshold widget to get the spike detection threshold
        spike_threshold_value = value.spikes_data.spike_threshold
        spike_threshold_mode = value.spikes_data.spike_threshold_mode

        if spike_threshold_mode == GLOBAL_SPIKE_THRESHOLD:
            spike_detection_threshold = spike_threshold_value
        else:  # MULTIPLIER
            # for spike amp use percentile-based approach to determine noise level
            non_zero_spikes = spikes[spikes > 0]
            # need sufficient data for reliable percentile
            if len(non_zero_spikes) > 5:
                spike_noise_reference = float(np.percentile(non_zero_spikes, 5))
            else:
                LOGGER.warning(
                    "Not enough data to determine spike noise reference "
                    "(< 5 non-zero spikes), using fallback value of 0.01."
                )
                spike_noise_reference = 0.01  # fallback value if not enough data
            spike_detection_threshold = spike_noise_reference * spike_threshold_value

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
        prom_multiplier = value.calcium_peaks_data.peaks_prominence_multiplier
        peaks_prominence_dec_dff: float = noise_level_dec_dff * prom_multiplier

        # use the peaks height widget to get the height threshold
        # if the mode is GLOBAL_HEIGHT, use the value directly, otherwise
        # use the value as a multiplier of the noise level
        peaks_height_value = value.calcium_peaks_data.peaks_height
        peaks_height_mode = value.calcium_peaks_data.peaks_height_mode
        if peaks_height_mode == GLOBAL_HEIGHT:
            peaks_height_dec_dff = peaks_height_value
        else:  # MULTIPLIER
            peaks_height_dec_dff = noise_level_dec_dff * peaks_height_value

        # Get minimum distance between peaks from user-specified value
        min_distance_frames = value.calcium_peaks_data.peaks_distance

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

        # get mask coords and shape for the ROI
        mask_coords, mask_shape = mask_to_coordinates(label_mask)

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
            led_power_equation=value.experiment_type_data.led_power_equation,
            calcium_sync_jitter_window=value.calcium_peaks_data.calcium_synchrony_jitter,
            spikes_sync_cross_corr_lag=value.spikes_data.synchrony_lag,
            calcium_network_threshold=value.calcium_peaks_data.calcium_network_threshold,
            spikes_burst_threshold=value.spikes_data.burst_threshold,
            spikes_burst_min_duration=value.spikes_data.burst_min_duration,
            spikes_burst_gaussian_sigma=value.spikes_data.burst_blur_sigma,
            mask_coord_and_shape=(mask_coords, mask_shape),
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
        self._analysis_settings_gui.enable(enable)
        self._run_btn.setEnabled(enable)
        if self._plate_viewer is None:
            return
        self._plate_viewer._segmentation_wdg.setEnabled(enable)
        # disable graphs tabs
        self._plate_viewer._tab.setTabEnabled(1, enable)
        self._plate_viewer._tab.setTabEnabled(2, enable)

    def _update_form_settings(self, f: Any) -> None:
        """Update the widget form from the JSON settings file."""
        # fmt: off
        # load the settings from the JSON file
        settings = cast(dict, json.load(f))
        # led power equation
        led_eq = cast(str, settings.get(LED_POWER_EQUATION, ""))
        # trace extraction data
        dff_window = cast(int, settings.get(DFF_WINDOW, DEFAULT_DFF_WINDOW))
        decay = cast(float, settings.get(DECAY_CONSTANT, 0.0))
        # calcium peaks data
        h_val = cast(float, settings.get(PEAKS_HEIGHT_VALUE, DEFAULT_HEIGHT))
        h_mode = cast(str, settings.get(PEAKS_HEIGHT_MODE, GLOBAL_HEIGHT))
        dist = cast(int, settings.get(PEAKS_DISTANCE, DEFAULT_PEAKS_DISTANCE))
        prom_mult = cast(float, settings.get(PEAKS_PROMINENCE_MULTIPLIER, 1.0))
        jit = cast(int,settings.get(CALCIUM_SYNC_JITTER_WINDOW, DEFAULT_CALCIUM_SYNC_JITTER_WINDOW))  # noqa: E501
        network_threshold = cast(float,settings.get(CALCIUM_NETWORK_THRESHOLD, DEFAULT_CALCIUM_NETWORK_THRESHOLD))  # noqa: E501
        # spikes data
        spike_thresh_val = cast(float, settings.get(SPIKE_THRESHOLD_VALUE, DEFAULT_SPIKE_THRESHOLD))  # noqa: E501
        spike_thresh_mode = cast(str, settings.get(SPIKE_THRESHOLD_MODE, MULTIPLIER))
        burst_the = cast(float, settings.get(BURST_THRESHOLD, DEFAULT_BURST_THRESHOLD))
        burst_d = cast(int, settings.get(BURST_MIN_DURATION, DEFAULT_MIN_BURST_DURATION))  # noqa: E501
        burst_g = cast(float, settings.get(BURST_GAUSSIAN_SIGMA, DEFAULT_BURST_GAUSS_SIGMA))  # noqa: E501
        lag = cast(int,settings.get(SPIKES_SYNC_CROSS_CORR_MAX_LAG, DEFAULT_SPIKE_SYNCHRONY_MAX_LAG))  # noqa: E501
        # fmt: on

        value = AnalysisSettingsData(
            experiment_type_data=ExperimentTypeData(led_power_equation=led_eq),
            trace_extraction_data=TraceExtractionData(
                dff_window_size=dff_window, decay_constant=decay
            ),
            calcium_peaks_data=CalciumPeaksData(
                peaks_height=h_val,
                peaks_height_mode=h_mode,
                peaks_distance=dist,
                peaks_prominence_multiplier=prom_mult,
                calcium_synchrony_jitter=jit,
                calcium_network_threshold=network_threshold,
            ),
            spikes_data=SpikeData(
                spike_threshold=spike_thresh_val,
                spike_threshold_mode=spike_thresh_mode,
                burst_threshold=burst_the,
                burst_min_duration=burst_d,
                burst_blur_sigma=burst_g,
                synchrony_lag=lag,
            ),
        )
        self._analysis_settings_gui.setValue(value)

    def _get_validated_settings(self) -> ValidatedAnalysisSettings:
        """Get settings from GUI.

        This is just for Type Checking purposes. In practice these are never None
        from the GUI so we cast them to ValidatedAnalysisSettings. This ensures that all
        fields are guaranteed to be non-None.
        """
        settings = self._analysis_settings_gui.value()

        # Assert all nested dataclass fields are non-None
        # since the GUI always provides complete values
        assert settings.experiment_type_data is not None
        assert settings.trace_extraction_data is not None
        assert settings.calcium_peaks_data is not None
        assert settings.spikes_data is not None
        assert settings.plate_map_data is not None
        assert settings.positions is not None

        return ValidatedAnalysisSettings(
            experiment_type_data=settings.experiment_type_data,
            trace_extraction_data=settings.trace_extraction_data,
            calcium_peaks_data=settings.calcium_peaks_data,
            spikes_data=settings.spikes_data,
            plate_map_data=settings.plate_map_data,
            positions=settings.positions,
        )

    def _save_settings_as_json(self) -> None:
        """Save the noise multiplier to a JSON file."""
        if not self.analysis_path:
            return

        settings_json_file = Path(self.analysis_path) / SETTINGS_PATH

        # fmt: off
        try:
            # Read existing settings if file exists
            settings = {}
            if settings_json_file.exists():
                with open(settings_json_file) as f:
                    settings = json.load(f)

            values = self._get_validated_settings()

            # GUI always provides these values so we can safely access them
            settings[DFF_WINDOW] = values.trace_extraction_data.dff_window_size
            settings[DECAY_CONSTANT] = values.trace_extraction_data.decay_constant

            settings[PEAKS_HEIGHT_VALUE] = values.calcium_peaks_data.peaks_height
            settings[PEAKS_HEIGHT_MODE] = values.calcium_peaks_data.peaks_height_mode
            settings[PEAKS_DISTANCE] = values.calcium_peaks_data.peaks_distance
            settings[PEAKS_PROMINENCE_MULTIPLIER] = values.calcium_peaks_data.peaks_prominence_multiplier  # noqa: E501
            settings[CALCIUM_NETWORK_THRESHOLD] = values.calcium_peaks_data.calcium_network_threshold  # noqa: E501

            settings[SPIKE_THRESHOLD_VALUE] = values.spikes_data.spike_threshold
            settings[SPIKE_THRESHOLD_MODE] = values.spikes_data.spike_threshold_mode
            settings[BURST_THRESHOLD] = values.spikes_data.burst_threshold
            settings[BURST_MIN_DURATION] = values.spikes_data.burst_min_duration
            settings[BURST_GAUSSIAN_SIGMA] = values.spikes_data.burst_blur_sigma
            settings[SPIKES_SYNC_CROSS_CORR_MAX_LAG] = values.spikes_data.synchrony_lag

            led_eq = values.experiment_type_data.led_power_equation or ""
            self._save_led_equation_to_json_settings(led_eq)

            # Write back the complete settings
            with open(settings_json_file, "w") as f:
                json.dump(
                    settings,
                    f,
                    indent=2,
                )
        except Exception as e:
            LOGGER.error(f"Failed to save noise multiplier: {e}")
        # fmt: on

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

    def _load_plate_map(self, plate: useq.WellPlate | None) -> None:
        """Load the plate map from the given file."""
        if plate is None:
            return
        plate_map_wdg = self._analysis_settings_gui._plate_map_wdg
        # clear the plate map data
        plate_map_wdg.clear()
        # set the plate type
        plate_map_wdg.setPlate(plate)
        # load plate map if exists
        if not self._analysis_path:
            return
        gen_path = Path(self._analysis_path) / GENOTYPE_MAP
        treat_path = Path(self._analysis_path) / TREATMENT_MAP
        plate_map_wdg.setValue(
            gen_path if gen_path.exists() else [],
            treat_path if treat_path.exists() else [],
        )
