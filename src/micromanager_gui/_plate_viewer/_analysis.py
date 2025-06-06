from __future__ import annotations

import bisect
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import tifffile
import useq
from fonticon_mdi6 import MDI6
from oasis.functions import deconvolve
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QSize, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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

from ._logger import LOGGER
from ._to_csv import save_to_csv
from ._util import (
    COND1,
    COND2,
    GENOTYPE_MAP,
    GREEN,
    LED_POWER_EQUATION,
    MWCM,
    RED,
    SETTINGS_PATH,
    STIMULATION_MASK,
    TREATMENT_MAP,
    ROIData,
    _BrowseWidget,
    _ElapsedTimer,
    _WaitingProgressBarWidget,
    calculate_dff,
    create_stimulation_mask,
    get_iei,
    get_linear_phase,
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

ELAPSED_TIME_KEY = "runner_time_ms"
CAMERA_KEY = "camera_metadata"
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"
EXCLUDE_AREA_SIZE_THRESHOLD = 10
STIMULATION_AREA_THRESHOLD = 0.1  # 10%
MAX_FRAMES_AFTER_STIMULATION = 5


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

        self._data: TensorstoreZarrReader | OMEZarrReader | None = data

        self._plate_map_data: dict[str, dict[str, str]] = {}
        self._stimulated_area_mask: np.ndarray | None = None
        self._labels_path: str | None = labels_path
        self._analysis_data: dict[str, dict[str, ROIData]] = {}
        self._min_peaks_height: float = 0.0

        self._led_power_equation: Callable | None = None

        self._worker: GeneratorWorker | None = None
        self._cancelled: bool = False

        # list to store the failed labels if they will not be found during the
        # analysis. used to show at the end of the analysis to the user which labels
        # are failed to be found.
        self._failed_labels: list[str] = []

        # ELAPSED TIME TIMER ---------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        # WIDGET TO SELECT THE EXPERIMENT TYPE ---------------------------------------
        experiment_type_wdg = QWidget(self)
        experiment_type_wdg_layout = QHBoxLayout(experiment_type_wdg)
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
            "• Linear: y = m*x + q (e.g., y = 2*x + 3)\n"
            "• Quadratic: y = a*x^2 + b*x + c (e.g., y = 0.5*x^2 + 2*x + 1)\n"
            "• Exponential: y = a*exp(b*x) + c (e.g., y = 2*exp(0.1*x) + 1)\n"
            "• Power: y = a*x^b + c (e.g., y = 2*x^0.5 + 1)\n"
            "• Logarithmic: y = a*log(x) + b (e.g., y = 2*log(x) + 1)\n"
            "Leave empty to use values from metadata."
        )
        led_lbl = QLabel("LED Power Equation:")
        led_lbl.setSizePolicy(*FIXED)
        self._led_power_equation_le = QLineEdit(self)
        # self._led_power_equation_le.setText("y = 11.07 * x - 6.63")
        self._led_power_equation_le.setPlaceholderText(
            "e.g., y = 2*x + 3 (Leave empty for metadata)"
        )
        led_layout = QHBoxLayout(self._led_power_wdg)
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.setSpacing(5)
        led_layout.addWidget(led_lbl)
        led_layout.addWidget(self._led_power_equation_le)
        self._led_power_wdg.hide()

        # WIDGET TO SELECT THE OUTPUT PATH -------------------------------------------
        self._analysis_path = _BrowseWidget(
            self,
            "Analysis Output Path",
            "",
            "Select the output path for the Analysis Data.",
            is_dir=True,
        )
        self._analysis_path.pathSet.connect(self._update_plate_viewer_analysis_path)

        # PEAKS SETTINGS -------------------------------------------------------------
        min_peaks_lbl_wdg = QWidget(self)
        min_peaks_lbl_wdg.setToolTip(
            "Set the min height for the peaks (used by the scipy find_peaks method)."
        )
        min_peaks_lbl = QLabel("Min Peaks Height:")
        min_peaks_lbl.setSizePolicy(*FIXED)
        self._min_peaks_height_spin = QDoubleSpinBox(self)
        self._min_peaks_height_spin.setDecimals(4)
        self._min_peaks_height_spin.setRange(0.0, 100000.0)
        self._min_peaks_height_spin.setSingleStep(0.01)
        self._min_peaks_height_spin.setValue(0.0075)
        min_peaks_layout = QHBoxLayout(min_peaks_lbl_wdg)
        min_peaks_layout.setContentsMargins(0, 0, 0, 0)
        min_peaks_layout.setSpacing(5)
        min_peaks_layout.addWidget(min_peaks_lbl)
        min_peaks_layout.addWidget(self._min_peaks_height_spin)

        # WIDGET TO SELECT THE POSITIONS TO ANALYZE ----------------------------------
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

        # PROGRESS BAR -------------------------------------------
        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        # RUN AND CANCEL BUTTONS ----------------------------------------------------
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

        # STYLING --------------------------------------------------------------------
        fixed_width = self._analysis_path._label.sizeHint().width()
        activity_combo_label.setFixedWidth(fixed_width)
        self._stimulation_area_path._label.setFixedWidth(fixed_width)
        led_lbl.setFixedWidth(fixed_width)
        pos_lbl.setFixedWidth(fixed_width)
        min_peaks_lbl.setFixedWidth(fixed_width)

        # LAYOUT ---------------------------------------------------------------------
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
        wdg_layout.addWidget(experiment_type_wdg)
        wdg_layout.addWidget(self._led_power_wdg)
        wdg_layout.addWidget(self._stimulation_area_path)
        wdg_layout.addWidget(min_peaks_lbl_wdg)
        wdg_layout.addWidget(self._analysis_path)
        wdg_layout.addWidget(pos_wdg)
        wdg_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

        self._cancel_waiting_bar = _WaitingProgressBarWidget(
            text="Stopping all the Tasks..."
        )

        # CONNECTIONS ---------------------------------------------------------------
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
        return self._analysis_path.value()

    @analysis_path.setter
    def analysis_path(self, analysis_path: str | None) -> None:
        self._analysis_path.setValue(analysis_path or "")

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

    def update_led_power_equation_form_settings(self) -> None:
        """Update the LED power equation line edit."""
        if not self.analysis_path:
            return None

        settings_json_file = Path(self.analysis_path) / SETTINGS_PATH
        if not settings_json_file.exists():
            return None

        try:
            with open(settings_json_file) as f:
                settings = cast(dict, json.load(f))
                pp = cast(str, settings.get(LED_POWER_EQUATION, ""))
                self._led_power_equation_le.setText(pp)
        except Exception as e:
            LOGGER.warning(f"Failed to load settings from {settings_json_file}: {e}")
            return None

    # PRIVATE METHODS -----------------------------------------------------------------

    # PREPARATION FOR RUNNING ---------------------------------------------------------

    def _prepare_for_running(self) -> list[int] | None:
        """Prepare the widget for running.

        Returns the number of positions to analyze or None if an error occurred.
        """
        if self._worker is not None and self._worker.is_running:
            return None

        if not self._validate_input_data():
            return None

        if self._plate_viewer and not self._validate_plate_map():
            return None

        if not (analysis_path := self._get_valid_output_path()):
            return None

        if self._is_evoked_experiment() and not self._prepare_stimulation_mask(
            analysis_path
        ):
            return None

        self._min_peaks_height = self._min_peaks_height_spin.value()

        # get the LED power equation from the line edit
        eq = self._led_power_equation_le.text()
        self._led_power_equation = self.equation_from_str(eq)
        if self._led_power_equation:
            self._save_led_equation_as_json(eq)

        return self._get_positions_to_analyze()

    def _validate_input_data(self) -> bool:
        """Check if required input data is available."""
        if self._data is None or self._labels_path is None:
            self._show_and_log_error("No data or labels path provided!")
            return False

        if self._data.sequence is None:
            self._show_and_log_error("No useq.MDAsequence found!")
            return False

        return True

    def _validate_plate_map(self) -> bool:
        """Validate plate map settings and prompt the user if needed."""
        if self._plate_viewer is None:
            return False

        tr_map = self._plate_viewer._plate_map_treatment.value()
        gen_map = self._plate_viewer._plate_map_genotype.value()

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
        if path := self._analysis_path.value():
            analysis_path = Path(path)
            if not analysis_path.is_dir():
                self._show_and_log_error("Output Path is not a directory!")
                return None
            return analysis_path

        self._show_and_log_error("No Output Path provided!")
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

    def equation_from_str(self, equation: str) -> Callable | None:
        """Parse various equation formats and return a callable function.

        Supported formats:
        - Linear: y = m*x + q  (e.g., "y = 2*x + 3")
        - Quadratic: y = a*x^2 + b*x + c  (e.g., "y = 0.5*x^2 + 2*x + 1")
        - Exponential: y = a*exp(b*x) + c  (e.g., "y = 2*exp(0.1*x) + 1")
        - Power: y = a*x^b + c  (e.g., "y = 2*x^0.5 + 1")
        - Logarithmic: y = a*log(x) + b  (e.g., "y = 2*log(x) + 1")
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

            if log_match := re.match(
                r"y=([+-]?\d*\.?\d+)\*log\(x\)([+-]\d*\.?\d+)", eq
            ):
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
            show_error_dialog(self, msg)
            return None

        except ValueError as e:
            msg = (
                f"Error parsing equation coefficients: {e}\n"
                "Using values from the metadata."
            )
            LOGGER.error(msg)
            show_error_dialog(self, msg)
            return None

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
        if self._plate_viewer is None:
            return

        condition_1_plate_map = self._plate_viewer._plate_map_genotype.value()
        conition_2_plate_map = self._plate_viewer._plate_map_treatment.value()

        # save plate map
        LOGGER.info("Saving Plate Maps.")
        if condition_1_plate_map:
            path = Path(self._analysis_path.value()) / GENOTYPE_MAP
            self._save_plate_map(path, self._plate_viewer._plate_map_genotype.value())
        if conition_2_plate_map:
            path = Path(self._analysis_path.value()) / TREATMENT_MAP
            self._save_plate_map(path, self._plate_viewer._plate_map_treatment.value())

        # update the stored _plate_map_data dict so we have the condition for each well
        # name as the key. e.g.:
        # {"A1": {"condition_1": "condition_1", "condition_2": "condition_2"}}
        self._plate_map_data.clear()
        for data in condition_1_plate_map:
            self._plate_map_data[data.name] = {COND1: data.condition[0]}

        for data in conition_2_plate_map:
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
        event_key = "mda_event" if "mda_event" in meta[0] else "Event"

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

        # get the elapsed time from the metadata to calculate the total time in seconds
        elapsed_time_list = self.get_elapsed_time_list(meta)

        # get the exposure time from the metadata
        exp_time = meta[0][event_key].get("exposure", 0.0)

        # get timepoints
        timepoints = sequence.sizes["t"]

        # get the total time in seconds for the recording
        tot_time_sec = self._calculate_total_time(
            elapsed_time_list, exp_time, timepoints
        )

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
                timepoints,
                exp_time,
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
            print(f"No labels found for {labels_name}!")
        return labels_path

    def _create_label_masks_dict(self, labels: np.ndarray) -> dict:
        """Create masks for each label in the labels image."""
        # get the range of labels and remove the background (0)
        labels_range = np.unique(labels[labels != 0])
        return {label_value: (labels == label_value) for label_value in labels_range}

    def get_elapsed_time_list(self, meta: list[dict]) -> list[float]:
        elapsed_time_list: list[float] = []
        # get the elapsed time for each timepoint to calculate tot_time_sec
        if (cam_key := CAMERA_KEY) in meta[0]:  # new metadata format
            for m in meta:
                et = m[cam_key].get(ELAPSED_TIME_KEY)
                if et is not None:
                    elapsed_time_list.append(float(et))
        return elapsed_time_list

    def _calculate_total_time(
        self,
        elapsed_time_list: list[float],
        exp_time: float,
        timepoints: int,
    ) -> float:
        """Calculate total time in seconds for the recording."""
        # if the len of elapsed time is not equal to the number of timepoints,
        # use exposure time and the number of timepoints to calculate tot_time_sec
        if len(elapsed_time_list) != timepoints:
            tot_time_sec = exp_time * timepoints / 1000
        # otherwise, calculate the total time in seconds using the elapsed time.
        # NOTE: adding the exposure time to consider the first frame
        else:
            tot_time_sec = (
                elapsed_time_list[-1] - elapsed_time_list[0] + exp_time
            ) / 1000
        return tot_time_sec

    def _process_roi_trace(
        self,
        data: np.ndarray,
        meta: list[dict],
        evoked_meta: dict[str, Any] | None,
        fov_name: str,
        label_value: int,
        label_mask: np.ndarray,
        timepoints: int,
        exp_time: float,
        tot_time_sec: float,
        evoked_exp: bool,
        elapsed_time_list: list[float],
    ) -> None:
        """Process individual ROI traces."""
        # get the data for the current label
        masked_data = data[:, label_mask]

        # get the size of the roi in µm or px if µm is not available
        roi_size_pixel = masked_data.shape[1]  # area
        px_size = meta[0].get("PixelSizeUm", None)
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

        # calculate the dff of the roi trace
        dff: np.ndarray = calculate_dff(roi_trace, window=10, plot=False)

        # deconvolve the dff trace
        dec_dff, spikes, _, _, _ = deconvolve(dff, penalty=1)

        # Get the prominence to find peaks in the deconvolved trace
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
        noise_level_dec_dff = np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
        peaks_prominence_dec_dff = noise_level_dec_dff  # * 2

        # find peaks in the deconvolved trace
        peaks_dec_dff, _ = find_peaks(
            dec_dff, prominence=peaks_prominence_dec_dff, height=self._min_peaks_height
        )

        # get the amplitudes of the peaks in the dec_dff trace
        peaks_amplitudes_dec_dff = [dec_dff[p] for p in peaks_dec_dff]

        # check if the roi is stimulated
        is_roi_stimulated = roi_stimulation_overlap_ratio > STIMULATION_AREA_THRESHOLD

        # to store the amplitudes as dict: {power_pulselength: [amplitude]}
        amplitudes_stimulated_peaks: dict[str, list[float]] = {}
        amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}
        stimulation_frames_and_powers: dict[str, int] = {}

        # if the experiment is evoked, get the amplitudes of the stimulated peaks
        if evoked_exp and evoked_meta is not None and len(peaks_dec_dff) > 0:
            # get the stimulation info from the metadata (if any)
            (
                amplitudes_stimulated_peaks,
                amplitudes_non_stimulated_peaks,
                stimulation_frames_and_powers,
            ) = self._update_stim_vs_non_stim(
                evoked_meta, dec_dff, peaks_dec_dff, is_roi_stimulated
            )

        # calculate the frequency of the peaks in the dec_dff trace
        frequency = (
            len(peaks_dec_dff) / tot_time_sec
            if tot_time_sec and len(peaks_dec_dff) > 0
            else None
        )

        # get the conditions for the well
        condition_1, condition_2 = self._get_conditions(fov_name)

        # calculate the linear phase of the peaks in the dec_dff trace
        instantaneous_phase = (
            get_linear_phase(timepoints, peaks_dec_dff)
            if len(peaks_dec_dff) > 0
            else None
        )

        # if the elapsed time is not available or for any reason is different from
        # the number of timepoints, set it as list of timepoints every exp_time
        if len(elapsed_time_list) != timepoints:
            elapsed_time_list = [i * exp_time for i in range(timepoints)]

        # calculate the inter-event interval (IEI) of the peaks in the dec_dff trace
        iei = get_iei(peaks_dec_dff, elapsed_time_list)

        # store the data to the analysis dict as ROIData
        self._analysis_data[fov_name][str(label_value)] = ROIData(
            well_fov_position=fov_name,
            raw_trace=cast(list[float], roi_trace.tolist()),
            dff=cast(list[float], dff.tolist()),
            dec_dff=dec_dff.tolist(),
            peaks_dec_dff=peaks_dec_dff.tolist(),
            peaks_amplitudes_dec_dff=peaks_amplitudes_dec_dff,
            peaks_prominence_dec_dff=peaks_prominence_dec_dff,
            dec_dff_frequency=frequency or None,
            inferred_spikes=spikes.tolist(),
            cell_size=roi_size,
            cell_size_units="µm" if px_size is not None else "pixel",
            condition_1=condition_1,
            condition_2=condition_2,
            total_recording_time_in_sec=tot_time_sec,
            active=len(peaks_dec_dff) > 0,
            instantaneous_phase=instantaneous_phase,
            iei=iei,
            evoked_experiment=evoked_exp,
            stimulated=is_roi_stimulated,
            amplitudes_stimulated_peaks=amplitudes_stimulated_peaks or None,
            amplitudes_non_stimulated_peaks=amplitudes_non_stimulated_peaks or None,
            stmulations_frames_and_powers=stimulation_frames_and_powers or None,
        )

    def _update_stim_vs_non_stim(
        self,
        evoked_experiment_meta: dict[str, Any],
        dec_dff: np.ndarray,
        peaks_dec_dff: np.ndarray,
        is_roi_stimulated: bool,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
        """Update the stimulated and non-stimulated peaks amplitude dict."""
        # to store the amplitudes as dict: {power_pulselength: [amplitude]}
        amplitudes_stimulated_peaks: dict[str, list[float]] = {}
        amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}

        pulse_on_frames_and_powers = cast(
            dict, evoked_experiment_meta.get("pulse_on_frame", {})
        )
        sorted_peaks_dec_dff = list(sorted(peaks_dec_dff))  # noqa: C413

        for frame, power in pulse_on_frames_and_powers.items():
            stim_frame = int(frame)
            # find index of first peak >= stim_frame.
            i = bisect.bisect_left(sorted_peaks_dec_dff, stim_frame)
            # Note that if the frame is not found, bisect_left returns the index
            # where it would be inserted and so the max index + 1. We need to check
            # if the index is valid and, if not, skip it.
            if i >= len(sorted_peaks_dec_dff):
                continue
            peak_idx = sorted_peaks_dec_dff[i]
            # check if the peak is on the stimulation frame or in the next 5 frames
            if (
                peak_idx >= stim_frame
                and peak_idx <= stim_frame + MAX_FRAMES_AFTER_STIMULATION
            ):
                amplitude = dec_dff[peak_idx]
                pulse_len = evoked_experiment_meta.get("led_pulse_duration", "unknown")
                if self._led_power_equation is not None:
                    power = self._led_power_equation(power)
                    power = f"{power:.3f}{MWCM}"
                col = f"{power}_{pulse_len}"
                if is_roi_stimulated:
                    amplitudes_stimulated_peaks.setdefault(col, []).append(amplitude)
                else:
                    amplitudes_non_stimulated_peaks.setdefault(col, []).append(
                        amplitude
                    )
        return (
            amplitudes_stimulated_peaks,
            amplitudes_non_stimulated_peaks,
            pulse_on_frames_and_powers,
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
            self._plate_viewer.pv_analysis_data = self._analysis_data

            # update the graphs with the new data
            if self._plate_viewer._tab.currentIndex() != 0:
                self._plate_viewer._on_tab_changed(1)
                for sgh in self._plate_viewer.SW_GRAPHS:
                    sgh._on_combo_changed(sgh._combo.currentText())
                for mgh in self._plate_viewer.MW_GRAPHS:
                    mgh._on_combo_changed(mgh._combo.currentText())

        # save the analysis data to a JSON file
        save_to_csv(self._analysis_path.value(), self._analysis_data)

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
        path = Path(self._analysis_path.value()) / f"{pos_name}.json"
        with path.open("w") as f:
            json.dump(
                self._analysis_data[pos_name],
                f,
                default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                indent=2,
            )

    # WIDGET --------------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(event)

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._cancel_waiting_bar.setEnabled(True)
        self._pos_le.setEnabled(enable)
        self._stimulation_area_path.setEnabled(enable)
        self._experiment_type_combo.setEnabled(enable)
        self._analysis_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)
        if self._plate_viewer is None:
            return
        self._plate_viewer._plate_map_group.setEnabled(enable)
        self._plate_viewer._segmentation_wdg.setEnabled(enable)
        # disable graphs tabs
        self._plate_viewer._tab.setTabEnabled(1, enable)
        self._plate_viewer._tab.setTabEnabled(2, enable)

    def _update_plate_viewer_analysis_path(self, path: str) -> None:
        """Update the analysis path of the plate viewer."""
        if self._plate_viewer is not None:
            self._plate_viewer._pv_analysis_path = path

    def _save_led_equation_as_json(self, eq: str) -> None:
        """Save the LED power equation to a JSON file."""
        if not self.analysis_path or not self._led_power_equation:
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
