from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from ._logger import LOGGER
from ._to_csv import save_trace_data_to_csv
from ._util import (
    DFF_WINDOW,
    GREEN,
    RED,
    SETTINGS_PATH,
    ROIData,
    _BrowseWidget,
    _ElapsedTimer,
    _WaitingProgressBarWidget,
    calculate_dff,
    parse_lineedit_text,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

DEFAULT_WINDOW = 50


def single_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return np.array(a * np.exp(-b * x) + c)


class _ExtractCalciumTraces(QWidget):
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

        # WIDGET TO SELECT THE OUTPUT PATH -------------------------------------------
        self._analysis_path = _BrowseWidget(
            self,
            "Analysis Output Path",
            "",
            "Select the output path for the Analysis Data.",
            is_dir=True,
        )
        self._analysis_path.pathSet.connect(self._update_plate_viewer_analysis_path)

        # DF/F SETTINGS --------------------------------------------------------
        dff_wdg = QWidget(self)
        dff_wdg.setToolTip(
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
            "• Too large (>1000): May not adapt to legitimate baseline shifts"
        )
        dff_lbl = QLabel("ΔF/F0 Window Size:")
        dff_lbl.setSizePolicy(*FIXED)
        self._dff_window_size_spin = QSpinBox(self)
        self._dff_window_size_spin.setRange(0, 10000)
        self._dff_window_size_spin.setSingleStep(1)
        self._dff_window_size_spin.setValue(DEFAULT_WINDOW)
        dff_layout = QHBoxLayout(dff_wdg)
        dff_layout.setContentsMargins(0, 0, 0, 0)
        dff_layout.setSpacing(5)
        dff_layout.addWidget(dff_lbl)
        dff_layout.addWidget(self._dff_window_size_spin)

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
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33. Leave empty for all")
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
        self._analysis_path._label.setFixedWidth(fixed_width)
        dff_lbl.setFixedWidth(fixed_width)
        pos_lbl.setFixedWidth(fixed_width)

        # LAYOUT ---------------------------------------------------------------------
        progress_wdg = QWidget(self)
        progress_wdg_layout = QHBoxLayout(progress_wdg)
        progress_wdg_layout.setContentsMargins(0, 0, 0, 0)
        progress_wdg_layout.addWidget(self._run_btn)
        progress_wdg_layout.addWidget(self._cancel_btn)
        progress_wdg_layout.addWidget(self._progress_bar)
        progress_wdg_layout.addWidget(self._progress_pos_label)
        progress_wdg_layout.addWidget(self._elapsed_time_label)

        self.groupbox = QGroupBox("Extract Calcium Traces", self)
        wdg_layout = QVBoxLayout(self.groupbox)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(self._analysis_path)
        wdg_layout.addWidget(dff_wdg)
        wdg_layout.addSpacing(10)
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

    # PUBLIC METHODS ---------------------------------------------------------------

    def run(self) -> None:
        """Extract the roi traces in a separate thread."""
        self._failed_labels.clear()

        print("Starting traces extraction...")

        pos = self._prepare_for_running()

        print("Positions to analyze:", pos)

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

    def update_widget_form_json_settings(self) -> None:
        """Update the widget form from the JSON settings."""
        if not self.analysis_path:
            return None

        settings_json_file = Path(self.analysis_path) / SETTINGS_PATH
        if not settings_json_file.exists():
            return None

        try:
            with open(settings_json_file) as f:
                settings = cast(dict, json.load(f))
                dff_window = cast(int, settings.get(DFF_WINDOW, DEFAULT_WINDOW))
                self._dff_window_size_spin.setValue(dff_window)
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

        if not self._plate_viewer:
            return None

        if not self._validate_input_data():
            LOGGER.error("Input data validation failed!")
            self._show_and_log_error("Input data validation failed!")
            return None

        if not self._get_valid_output_path():
            LOGGER.error("Output path validation failed!")
            self._show_and_log_error("Output path validation failed!")
            return None

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

    def _get_positions_to_analyze(self) -> list[int] | None:
        """Get the positions to analyze."""
        if self._data is None or (sequence:=self._data.sequence) is None:
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

    # RUN THE TRACES EXTRACTION -------------------------------------------------------

    def _extract_traces_data(self, positions: list[int]) -> Generator[str, None, None]:
        """Extract the roi traces in multiple threads."""
        LOGGER.info("Starting traces analysis...")

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

        msg = f"Extracting Traces Data from Well {fov_name}."
        LOGGER.info(msg)
        for label_value, label_mask in tqdm(labels_masks.items(), desc=msg):
            if self._check_for_abort_requested():
                break
            # extract the data
            self._process_roi_trace(data, meta, fov_name, label_value, label_mask)

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

    def _create_label_masks_dict(self, labels: np.ndarray) -> dict:
        """Create masks for each label in the labels image."""
        # get the range of labels and remove the background (0)
        labels_range = np.unique(labels[labels != 0])
        return {label_value: (labels == label_value) for label_value in labels_range}

    def _process_roi_trace(
        self,
        data: np.ndarray,
        meta: list[dict],
        fov_name: str,
        label_value: int,
        label_mask: np.ndarray,
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
        # compute the mean for each frame
        roi_trace: np.ndarray = masked_data.mean(axis=1)
        win = self._dff_window_size_spin.value()
        # calculate the dff of the roi trace
        dff: np.ndarray = calculate_dff(roi_trace, window=win, plot=False)

        # deconvolve the dff trace with adaptive penalty
        dec_dff, spikes, _, _, _ = deconvolve(dff, penalty=1)

        # store the data to the analysis dict as ROIData
        self._analysis_data[fov_name][str(label_value)] = ROIData(
            well_fov_position=fov_name,
            label_mask=label_mask.tolist(),
            raw_trace=cast(list[float], roi_trace.tolist()),
            dff=cast(list[float], dff.tolist()),
            dec_dff=dec_dff.tolist(),
            inferred_spikes=spikes.tolist(),
            cell_size=roi_size,
            cell_size_units="µm" if px_size is not None else "pixel",
        )

    def _on_worker_finished(self) -> None:
        """Called when the data extraction is finished."""
        LOGGER.info("Traces extraction completed!.")
        self._enable(True)
        self._elapsed_timer.stop()
        self._cancel_waiting_bar.stop()

        # update the analysis data of the plate viewer
        if self._plate_viewer is not None:
            self._plate_viewer.pv_analysis_data = self._analysis_data

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
        if self.analysis_path:
            save_trace_data_to_csv(self.analysis_path, self._analysis_data)

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

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(a0)

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._cancel_waiting_bar.setEnabled(True)
        self._pos_le.setEnabled(enable)
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
