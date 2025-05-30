from __future__ import annotations

import concurrent.futures
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING

import tifffile
from cellpose import core, models
from cellpose.models import CellposeModel
from fonticon_mdi6 import MDI6
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._plate_viewer._util import GREEN, RED, _BrowseWidget
from micromanager_gui._widgets._mda_widget._save_widget import (
    OME_ZARR,
    WRITERS,
    ZARR_TESNSORSTORE,
)
from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

if TYPE_CHECKING:
    from threading import Event

    from superqt.utils import FunctionWorker


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
EXT = (WRITERS[OME_ZARR][0], WRITERS[ZARR_TESNSORSTORE][0])


CUSTOM = "custom"
CYTO3 = "cyto3"
CUSTOM_MODEL_PATH = "cellpose_models/cp3_img8_epoch7000_py"


class _SelectModelPath(_BrowseWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "Custom Model",
        tooltip: str = "Choose the path to the custom Cellpose model.",
    ) -> None:
        super().__init__(parent, label, "", tooltip, is_dir=False)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select the {self._label_text}.",
            "",
            "",  # TODO: add model extension
        )
        if path:
            self._path.setText(path)


class CellposeBatchSegmentationMP(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Cellpose Segmentation")

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

        model_wdg = QWidget(self)
        model_wdg_layout = QHBoxLayout(model_wdg)
        model_wdg_layout.setContentsMargins(0, 0, 0, 0)
        model_wdg_layout.setSpacing(5)
        self._models_combo_label = QLabel("Model Type:")
        self._models_combo_label.setSizePolicy(*FIXED)
        self._models_combo = QComboBox()
        # self._models_combo.addItems(["nuclei", "cyto", "cyto2", "cyto3", "custom"])
        self._models_combo.addItems([CYTO3, CUSTOM])
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)
        model_wdg_layout.addWidget(self._models_combo_label)
        model_wdg_layout.addWidget(self._models_combo, 1)

        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # add icons to the buttons
        button_box.button(QDialogButtonBox.StandardButton.Ok).setIcon(
            icon(MDI6.play, color=GREEN)
        )
        button_box.button(QDialogButtonBox.StandardButton.Cancel).setIcon(
            icon(MDI6.stop, color=RED)
        )
        button_box.accepted.connect(self.run)
        button_box.rejected.connect(self.cancel)

        wdg = QWidget()
        settings_groupbox_layout = QVBoxLayout(wdg)
        settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        settings_groupbox_layout.setSpacing(5)
        settings_groupbox_layout.addWidget(model_wdg)
        settings_groupbox_layout.addWidget(self._browse_custom_model)
        settings_groupbox_layout.addWidget(self._input_path)
        settings_groupbox_layout.addWidget(button_box)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(wdg)
        main_layout.addStretch(1)

        # STYLING -------------------------------------------------------------
        fixed_width = self._browse_custom_model._label.sizeHint().width()
        self._input_path._label.setFixedWidth(fixed_width)
        self._models_combo_label.setFixedWidth(fixed_width)

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

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()

    def _get_moedel_type_and_path(self) -> tuple[str, str]:
        model_type = self._models_combo.currentText()
        model_path = self._browse_custom_model.value()
        return model_type, model_path

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

        model_type, model_path = self._get_moedel_type_and_path()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._futures = [
                executor.submit(
                    _segment_data, f, self._stop_event, model_type, model_path
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


def _segment_data(
    data_path: str, stop_event: Event, model_type: str, model_path: str
) -> None:
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

    # only cuda since per now cellpose does not work with gpu on mac
    use_gpu = core.use_gpu()
    if model_type == CUSTOM:
        model = CellposeModel(pretrained_model=model_path, gpu=use_gpu)
    else:  # model_type == CYTO3
        model = models.Cellpose(model_type=model_type, gpu=use_gpu)

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
        output = model.eval(cyto_frame)
        labels = output[0]
        # save to disk
        tifffile.imwrite(path / f"{pos_name}_p{p}.tif", labels)
