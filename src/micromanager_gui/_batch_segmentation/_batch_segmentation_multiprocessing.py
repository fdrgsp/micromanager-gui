from __future__ import annotations

import concurrent.futures
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING

import tifffile
from cellpose import models
from fonticon_mdi6 import MDI6
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._plate_viewer._init_dialog import _BrowseWidget
from micromanager_gui._plate_viewer._util import GREEN, RED
from micromanager_gui._widgets._mda_widget._save_widget import (
    OME_ZARR,
    WRITERS,
    ZARR_TESNSORSTORE,
)
from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

if TYPE_CHECKING:
    from threading import Event

    from cellpose.models import CellposeModel
    from superqt.utils import FunctionWorker


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
MODEL = "cyto3"
EXT = (WRITERS[OME_ZARR][0], WRITERS[ZARR_TESNSORSTORE][0])


class CellposeBatchSegmentation(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._run_worker: FunctionWorker | None = None
        self._futures: list[concurrent.futures.Future] = []
        self._stop_event: Event = Manager().Event()

        self._input_path = _BrowseWidget(
            self,
            "Input Folder",
            "",
            "Choose the folder containing the files to segment.",
            is_dir=True,
        )

        buttons_wdg = QWidget(self)
        buttons_layout = QHBoxLayout(buttons_wdg)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(5)
        self._use_gpu = QCheckBox("Use GPU")
        self._use_gpu.setChecked(True)
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
        buttons_layout.addWidget(self._use_gpu)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self._run_btn)
        buttons_layout.addWidget(self._cancel_btn)

        self.groupbox = QGroupBox("Batch Cellpose Segmentation", self)
        settings_groupbox_layout = QVBoxLayout(self.groupbox)
        settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        settings_groupbox_layout.setSpacing(5)
        settings_groupbox_layout.addWidget(self._input_path)
        settings_groupbox_layout.addWidget(buttons_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

    def run(self) -> None:
        self._stop_event.clear()
        self._run_worker = create_worker(self._run, _start_thread=True)

    def cancel(self) -> None:
        """Cancel the current run."""
        self._stop_event.set()
        for future in self._futures:
            future.cancel()
        if self._run_worker is not None:
            self._run_worker.quit()

    def _run(self) -> None:
        """Run the Cellpose segmentation."""
        input_path = self._input_path.value()

        if not input_path:
            return

        # select only folders within the input path and, within them, only ome.zarr or
        # tensorstore.zarr files
        files = [
            str(f)
            for folder in Path(input_path).iterdir()
            if folder.is_dir()
            for f in folder.iterdir()
            if f.name.endswith(EXT)
        ]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._futures = [
                executor.submit(
                    _segment_data, f, self._stop_event, self._use_gpu.isChecked()
                )
                for f in files
            ]

            for future in tqdm(
                concurrent.futures.as_completed(self._futures),
                total=len(self._futures),
                desc="Processing files",
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")


def _segment_data(data_path: str, stop_event: Event, use_gpu: bool) -> None:
    """Segment the data with Cellpose."""
    if stop_event.is_set():
        print(f"Segmentation process stopped for {data_path}")
        return

    path = Path(data_path)
    data: OMEZarrReader | TensorstoreZarrReader
    if path.name.endswith(WRITERS[OME_ZARR][0]):
        data = OMEZarrReader(path)
    elif path.name.endswith(WRITERS[ZARR_TESNSORSTORE][0]):
        data = TensorstoreZarrReader(path)
    else:
        print(f"Unsupported file format: {path.name}, skipping...")
        return

    sequence = data.sequence
    if sequence is None:
        print(f"Skipping {data.path.name}, no sequence foundata.")
        return

    positions = list(range(len(sequence.stage_positions)))

    model = models.Cellpose(gpu=use_gpu, model_type=MODEL)

    file_name = data.path.name
    for ext in EXT:
        if file_name.endswith(ext):
            file_name = file_name[: -len(ext)]
            break

    path = data.path.parent / f"{file_name}_labels"
    if not path.exists():
        path.mkdir()

    _segment(
        data=data, path=path, model=model, positions=positions, stop_event=stop_event
    )


def _segment(
    data: OMEZarrReader | TensorstoreZarrReader,
    path: Path,
    model: CellposeModel,
    positions: list[int],
    stop_event: Event,
) -> None:
    """Perform the segmentation using Cellpose."""
    for p in tqdm(positions, desc="Processing positions"):
        if stop_event.is_set():
            print(f"Segmentation stopped at position {p}")
            break
        # get the data
        stack, meta = data.isel(p=p, metadata=True)
        # get position name from metadata
        pos_name = meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")
        # max projection from half to the end of the stack
        stack_half_to_end = stack[stack.shape[0] // 2 :, :, :]
        # perform cellpose on each time point
        cyto_frame = stack_half_to_end.max(axis=0)
        labels, _, _, _ = model.eval(cyto_frame)
        # save to disk
        tifffile.imsave(path / f"{pos_name}_p{p}.tif", labels)
