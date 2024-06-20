from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
import tifffile
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._util import ElapsedTimer, show_error_dialog

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
    from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

    from ._plate_viewer import PlateViewer


@dataclass
class Peaks:
    """NamedTuple to store peak data."""

    peak: int | None = None
    amplitude: float | None = None
    raise_time: float | None = None
    decay_time: float | None = None
    # ... add whatever other data we need


@dataclass
class ROIData:
    """NamedTuple to store ROI data."""

    trace: list[float] | None = None
    peaks: list[Peaks] | None = None
    mean_frequency: float | None = None
    mean_amplitude: float | None = None
    # ... add whatever other data we need


def calculate_roi_trace(data: np.ndarray, mask: list[np.ndarray]) -> list[float]:
    roi_trace = []
    for i in range(data.shape[0]):
        # get the mean intensity of the roi
        roi = np.where(mask, data[i], 0)
        roi_trace.append(roi.mean())
    return roi_trace


class _AnalyseCalciumTraces(QWidget):
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

        self._extract_traces_btn = QPushButton("Extract Traces", self)
        self._extract_traces_btn.clicked.connect(self.extract_traces)
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self._cancel)

        self._output_path = _BrowseWidget(
            self,
            "Analysis Output Path",
            "",
            "Select the output path for the Analysis Data.",
            is_dir=True,
        )

        progress_wdg = QWidget(self)
        progress_wdg_layout = QHBoxLayout(progress_wdg)
        progress_wdg_layout.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_label = QLabel("00:00:00")
        progress_wdg_layout.addWidget(self._progress_bar)
        progress_wdg_layout.addWidget(self._progress_label)

        self._elapsed_timer = ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        wdg = QGroupBox("Extract Traces", self)
        wdg.setCheckable(True)
        wdg.setChecked(False)
        wdg_layout = QGridLayout(wdg)
        wdg_layout.setContentsMargins(10, 10, 10, 10)
        wdg_layout.setSpacing(5)
        wdg_layout.addWidget(self._output_path, 0, 0, 1, 3)
        wdg_layout.addWidget(self._extract_traces_btn, 1, 0)
        wdg_layout.addWidget(self._cancel_btn, 1, 1)
        wdg_layout.addWidget(progress_wdg, 1, 2)

        self._extract_traces_btn.setFixedWidth(
            self._output_path._label.minimumSizeHint().width()
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(wdg)
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

    @property
    def analysis_data(self) -> dict[str, dict[str, ROIData]]:
        return self._analysis_data

    def save_analysis_data(self, path: str) -> None:
        """Save the analysis data to a JSON file."""
        with open(path, "w") as f:
            json.dump(
                self._analysis_data,
                f,
                default=lambda o: asdict(o) if isinstance(o, ROIData) else o,
                indent=2,
            )

    def extract_traces(self) -> None:
        """Extract the roi traces in a separate thread."""
        if self._worker is not None and self._worker.is_running:
            return

        # start elapsed timer
        self._elapsed_timer.start()

        self._worker = create_worker(
            self._extract_traces,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress_bar,
                "finished": self._stop_timer,
            },
        )

    def _cancel(self) -> None:
        """Cancel the current extraction."""
        if self._worker is not None:
            self._worker.quit()
        # stop the elapsed timer
        self._stop_timer()
        self._progress_label.setText("00:00:00")

    def _stop_timer(self) -> None:
        """Stop the elapsed timer and the time timer."""
        self._elapsed_timer.stop()

    def _update_progress_label(self, time_str: str) -> None:
        """Update the progress label with elapsed time."""
        self._progress_label.setText(time_str)

    def _update_progress_bar(self, value: int) -> None:
        """Update the progress bar value."""
        self._progress_bar.setValue(value)

    def _extract_traces(self) -> Generator[int, None, None]:
        """Extract the roi traces in multiple threads."""
        if self.data is None or self.labels_path is None:
            show_error_dialog(self, "No data or labels path provided!")
            return

        sequence = self.data.sequence
        if sequence is None:
            show_error_dialog(self, "No useq.MDAsequence found!")
            return

        pos = len(sequence.stage_positions)

        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(self._extract_trace_per_position, p)
                    for p in range(pos)
                ]
                for idx, _ in enumerate(as_completed(futures)):
                    if self._worker is not None and self._worker.abort_requested:
                        break
                    yield int((idx / pos) * 100)

        except Exception as e:
            show_error_dialog(self, f"An error occurred: {e}")

    def _extract_trace_per_position(self, p: int) -> None:
        if self.data is None or (
            self._worker is not None and self._worker.abort_requested
        ):
            return

        data, meta = self.data.isel(p=p, metadata=True)
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
        labels_range = range(1, labels.max() + 1)
        for label_value in tqdm(labels_range, desc=f"Processing well {well}"):
            if self._worker is not None and self._worker.abort_requested:
                break
            mask = labels == label_value
            roi_trace = calculate_roi_trace(data, mask)
            # create the dict for the roi
            self._analysis_data[well][str(label_value)] = ROIData(trace=roi_trace)

    def _get_labels_file(self, label_name: str) -> str | None:
        """Get the labels file for the given name."""
        if self._labels_path is None:
            return None
        for label_file in Path(self._labels_path).glob("*.tif"):
            if label_file.name.endswith(label_name):
                return str(label_file)
        return None

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(event)
