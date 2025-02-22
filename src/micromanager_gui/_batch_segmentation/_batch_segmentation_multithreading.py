from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tifffile
from cellpose import models
from fonticon_mdi6 import MDI6
from qtpy.QtCore import QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
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
    from collections.abc import Generator

    import numpy as np
    from cellpose.models import CellposeModel
    from superqt.utils import GeneratorWorker


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
MODEL = "cyto3"
EXT = (WRITERS[OME_ZARR][0], WRITERS[ZARR_TESNSORSTORE][0])


class CellposeBatchSegmentation(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._labels: dict[str, np.ndarray] = {}

        self._worker: GeneratorWorker | None = None

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
        settings_groupbox_layout = QGridLayout(self.groupbox)
        settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        settings_groupbox_layout.setSpacing(5)
        settings_groupbox_layout.addWidget(self._input_path, 0, 0, 1, 2)
        settings_groupbox_layout.addWidget(buttons_wdg, 1, 0, 2, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

    def run(self) -> None:
        """Run the Cellpose segmentation."""
        if self._worker is not None and self._worker.is_running:
            return

        input_path = self._input_path.value()

        if not input_path:
            return

        # select only folders within the input path and, within them, only ome.zarr or
        # tensorstore.zarr files
        files = [
            f
            for folder in Path(input_path).iterdir()
            if folder.is_dir()
            for f in folder.iterdir()
            if f.name.endswith(EXT)
        ]

        data: list[OMEZarrReader | TensorstoreZarrReader] = []
        for f in files:
            if f.name.endswith(WRITERS[OME_ZARR][0]):
                data.append(OMEZarrReader(f))
            elif f.name.endswith(WRITERS[ZARR_TESNSORSTORE][0]):
                data.append(TensorstoreZarrReader(f))
            else:
                print(f"Unsupported file format: {f.name}, skipping...")
                continue

        for d in tqdm(data, total=len(data), desc="Processing files"):
            sequence = d.sequence
            if sequence is None:
                print(f"Skipping {d.path.name}, no sequence found.")
                continue

            positions = list(range(len(sequence.stage_positions)))

            model = models.Cellpose(gpu=self._use_gpu.isChecked(), model_type=MODEL)

            file_name = d.path.name
            for ext in EXT:
                if file_name.endswith(ext):
                    file_name = file_name[: -len(ext)]
                    break

            path = d.path.parent / f"{file_name}_labels"
            if not path.exists():
                path.mkdir()

            self._worker = create_worker(
                self._segment,
                data=d,
                path=path,
                model=model,
                positions=positions,
                _start_thread=True,
            )

    def _segment(
        self,
        data: OMEZarrReader | TensorstoreZarrReader,
        path: str,
        model: CellposeModel,
        positions: list[int],
    ) -> Generator[str, None, None]:
        """Perform the segmentation using Cellpose."""
        for p in tqdm(positions, desc="Processing positions"):
            if self._worker is not None and self._worker.abort_requested:
                break
            # get the data
            stack, meta = data.isel(p=p, metadata=True)
            # get position name from metadata
            pos_name = (
                meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")
            )
            yield f"[Well {pos_name} (p{p}/{len(positions)-1})]"
            # max projection from half to the end of the stack
            stack_half_to_end = stack[stack.shape[0] // 2 :, :, :]
            # perform cellpose on each time point
            cyto_frame = stack_half_to_end.max(axis=0)
            labels, _, _, _ = model.eval(cyto_frame)
            self._labels[f"{pos_name}_p{p}"] = labels
            # save to disk
            tifffile.imwrite(Path(path) / f"{pos_name}_p{p}.tif", labels)

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is not None:
            self._worker.quit()
