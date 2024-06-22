from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tifffile
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._util import ElapsedTimer, ROIData, show_error_dialog

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
    from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

    from ._plate_viewer import PlateViewer

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


# def calculate_roi_trace(data: np.ndarray, mask: np.ndarray) -> list[float]:
#     roi_trace = []
#     for i in range(data.shape[0]):
#         roi = data[i][mask]
#         roi_trace.append(np.mean(roi))
#     return roi_trace


class _SelectAnalysisPath(_BrowseWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        path: str | None = None,
        tooltip: str = "",
    ) -> None:
        super().__init__(parent, label, path, tooltip)

    def _on_browse(self) -> None:
        dialog = QFileDialog(self, f"Select the {self._label_text}.")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dialog.setDirectory(self._current_path)

        if dialog.exec() == QFileDialog.Accepted:
            selected_path = dialog.selectedFiles()[0]
            self._path.setText(selected_path)


class _AnalyseCalciumTraces(QWidget):
    progress_bar_updated = Signal()
    elapsed_time_updated = Signal(str)

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

        self._output_path = _SelectAnalysisPath(
            self,
            "Analysis Output Path",
            "",
            "Select the output path for the Analysis Data.",
        )

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

        self._elapsed_timer = ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        self.progress_bar_updated.connect(self._update_progress_bar)

        self.groupbox = QGroupBox("Extract Traces", self)
        self.groupbox.setCheckable(True)
        self.groupbox.setChecked(False)
        wdg_layout = QVBoxLayout(self.groupbox)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(self._output_path)
        wdg_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

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

    def save_analysis_data(self, path: str) -> None:
        """Save the analysis data to a JSON file in a separate thread."""
        create_worker(self._save_analysis_data, path=path, _start_thread=True)

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is not None:
            self._worker.quit()
        # stop the elapsed timer
        self._elapsed_timer.stop()
        self._progress_bar.reset()
        self._progress_pos_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def run(self) -> None:
        """Extract the roi traces in a separate thread."""
        if self._worker is not None and self._worker.is_running:
            return

        if self._data is None or self._labels_path is None:
            show_error_dialog(self, "No data or labels path provided!")
            return

        sequence = self._data.sequence
        if sequence is None:
            show_error_dialog(self, "No useq.MDAsequence found!")
            return

        if not self._output_path.value():
            show_error_dialog(self, "No Output Path provided!")
            return

        # check if the provided json file is empty. If not, ask the user to overwrite it
        if Path(self._output_path.value()).is_file():
            with open(self._output_path.value()) as f:
                if f.read():
                    response = self._overwrite_msgbox()
                    if response == QMessageBox.StandardButton.No:
                        return

        pos = len(sequence.stage_positions)
        self._progress_bar.reset()
        self._progress_bar.setRange(0, pos)
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText(f"[0/{self._progress_bar.maximum()}]")

        # start elapsed timer
        self._elapsed_timer.start()

        self._worker = create_worker(
            self._extract_traces,
            positions=pos,
            _start_thread=True,
            _connect={"finished": self._on_finished},
        )

    def _get_save_name(self) -> str:
        """Generate a save name based on metadata."""
        name = "analysis_data"
        if self._data is not None:
            seq = self._data.sequence
            if seq is not None:
                meta = seq.metadata.get(PYMMCW_METADATA_KEY, {})
                name = meta.get("save_name", name)
                name = f"{name}_analysis_data"
        return name

    def _save_analysis_data(self, path: str) -> None:
        """Save the analysis data to a JSON file."""
        save_path = Path(path)

        if save_path.is_dir():
            name = self._get_save_name()
            save_path = save_path / f"{name}.json"
        elif not save_path.suffix:
            save_path = save_path.with_suffix(".json")

        try:
            with save_path.open("w") as f:
                json.dump(
                    self._analysis_data,
                    f,
                    default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                    indent=2,
                )
        except OSError as e:
            show_error_dialog(self, f"File system error occurred: {e}")
        except json.JSONDecodeError as e:
            show_error_dialog(self, f"JSON serialization error occurred: {e}")
        except Exception as e:
            show_error_dialog(self, f"An unexpected error occurred: {e}")

    def _overwrite_msgbox(self) -> Any:
        """Show a message box to ask the user if wants to overwrite the json file."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("The provided json file is not empty!")
        msg.setInformativeText("Do you want to overwrite it?")
        msg.setWindowTitle("Overwrite Analysis Data")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        return msg.exec()

    def _on_finished(self) -> None:
        """Called when the extraction is finished."""
        self._elapsed_timer.stop()
        self._progress_bar.setValue(self._progress_bar.maximum())

        # save the analysis data
        self.save_analysis_data(self._output_path.value())

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

    def _extract_traces(self, positions: int) -> None:
        """Extract the roi traces in multiple threads."""
        cpu_count = os.cpu_count() or 1
        chunk_size = max(1, positions // cpu_count)

        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(
                        self._extract_trace_for_chunk,
                        start,
                        min(start + chunk_size, positions),
                    )
                    for start in range(0, positions, chunk_size)
                ]
                for _ in as_completed(futures):
                    if self._check_for_abort_requested():
                        for f in futures:
                            f.cancel()
                        break

        except Exception as e:
            show_error_dialog(self, f"An error occurred: {e}")

    def _extract_trace_for_chunk(self, start: int, end: int) -> None:
        for p in range(start, end):
            if self._check_for_abort_requested():
                break
            self._extract_trace_per_position(p)

    def _check_for_abort_requested(self) -> bool:
        return bool(self._worker is not None and self._worker.abort_requested)

    def _extract_trace_per_position(self, p: int) -> None:
        if self._data is None or self._check_for_abort_requested():
            return

        data, meta = self._data.isel(p=p, metadata=True)

        # get position name from metadata
        well = meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")

        # create the dict for the well
        if well not in self._analysis_data:
            self._analysis_data[well] = {}

        # matching label name
        labels_name = f"{well}_p{p}.tif"

        # get the labels file
        labels = tifffile.imread(self._get_labels_file(labels_name))
        if labels is None:
            show_error_dialog(self, f"No labels found for {labels_name}!")
            return

        # get the range of labels
        labels_range = range(1, labels.max() + 1)

        # create masks for each label
        masks = {label_value: (labels == label_value) for label_value in labels_range}

        # extract roi traces
        for label_value, mask in tqdm(masks.items(), desc=f"Processing well {well}"):
            if self._check_for_abort_requested():
                break

            # calculate the mean trace for the roi
            masked_data = data[:, mask]

            # compute the mean for each frame
            roi_trace = cast(np.ndarray, masked_data.mean(axis=1))

            # store the roi trace
            self._analysis_data[well][str(label_value)] = ROIData(
                raw_trace=roi_trace.tolist()
            )

        # update the progress bar
        if not self._check_for_abort_requested():
            self.progress_bar_updated.emit()

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._output_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(event)
