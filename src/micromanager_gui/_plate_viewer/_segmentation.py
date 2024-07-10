from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import tifffile
from cellpose import models
from cellpose.models import CellposeModel
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
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
from superqt.utils import create_worker
from tqdm import tqdm

from ._init_dialog import _BrowseWidget
from ._util import _ElapsedTimer, parse_lineedit_text, show_error_dialog

if TYPE_CHECKING:
    import numpy as np
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
    from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

    from ._plate_viewer import PlateViewer


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


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


class _CellposeSegmentation(QWidget):
    def __init__(
        self,
        parent: PlateViewer | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
    ) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer | None = parent

        self._data: TensorstoreZarrReader | OMEZarrReader | None = data

        self._labels: dict[str, np.ndarray] = {}

        self._worker: GeneratorWorker | None = None

        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.hide()

        model_wdg = QWidget(self)
        model_wdg_layout = QHBoxLayout(model_wdg)
        model_wdg_layout.setContentsMargins(0, 0, 0, 0)
        model_wdg_layout.setSpacing(5)
        self._models_combo_label = QLabel("Model Type:")
        self._models_combo_label.setSizePolicy(*FIXED)
        self._models_combo = QComboBox()
        # self._models_combo.addItems(["nuclei", "cyto", "cyto2", "cyto3", "custom"])
        self._models_combo.addItems(["cyto3", "custom"])
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)
        model_wdg_layout.addWidget(self._models_combo_label)
        model_wdg_layout.addWidget(self._models_combo)

        channel_wdg = QWidget(self)
        channel_layout = QHBoxLayout(channel_wdg)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(5)
        self._channel_combo_label = QLabel("Segment Channel:")
        self._channel_combo_label.setSizePolicy(*FIXED)
        self._channel_combo = QComboBox()
        self._channel_combo.setToolTip("Select the channel to segment.")
        if self._data is not None and self._data.sequence:
            chs = self._data.sequence.sizes.get("c")
            if chs is not None:
                self._channel_combo.addItems([str(i) for i in range(chs)])
        channel_layout.addWidget(self._channel_combo_label)
        channel_layout.addWidget(self._channel_combo)

        diameter_wdg = QWidget(self)
        diameter_layout = QHBoxLayout(diameter_wdg)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.setSpacing(5)
        self._diameter_label = QLabel("Diameter:")
        self._diameter_label.setSizePolicy(*FIXED)
        self._diameter_spin = QDoubleSpinBox(self)
        self._diameter_spin.setRange(0, 1000)
        self._diameter_spin.setValue(0)
        self._diameter_spin.setToolTip("Set the diameter of the cells.")
        diameter_layout.addWidget(self._diameter_label)
        diameter_layout.addWidget(self._diameter_spin)

        self._output_path = _BrowseWidget(
            self,
            "Labels Output Path",
            "",
            "Choose the path to save the labels.",
            is_dir=True,
        )

        pos_wdg = QWidget(self)
        pos_wdg.setToolTip(
            "Select the Positions to segment. Leave blank to segment all Positions. "
            "You can input single Positions (e.g. 30, 33) a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65). "
            "NOTE: The Positions are 0-indexed."
        )
        pos_wdg_layout = QHBoxLayout(pos_wdg)
        pos_wdg_layout.setContentsMargins(0, 0, 0, 0)
        pos_wdg_layout.setSpacing(5)
        pos_lbl = QLabel("Segment Positions:")
        pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit()
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33")
        pos_wdg_layout.addWidget(pos_lbl)
        pos_wdg_layout.addWidget(self._pos_le)

        # set the minimum width of the labels
        fixed_lbl_width = self._output_path._label.minimumSizeHint().width()
        self._models_combo_label.setMinimumWidth(fixed_lbl_width)
        self._channel_combo_label.setMinimumWidth(fixed_lbl_width)
        self._browse_custom_model._label.setMinimumWidth(fixed_lbl_width)
        self._diameter_label.setMinimumWidth(fixed_lbl_width)
        pos_lbl.setMinimumWidth(fixed_lbl_width)

        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        progress_wdg = QWidget(self)
        progress_layout = QHBoxLayout(progress_wdg)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(5)

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.clicked.connect(self.run)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.clicked.connect(self.cancel)

        self._progress_label = QLabel("[0/0]")
        self._progress_bar = QProgressBar(self)
        self._elapsed_time_label = QLabel("00:00:00")

        progress_layout.addWidget(self._run_btn)
        progress_layout.addWidget(self._cancel_btn)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._elapsed_time_label)

        self.groupbox = QGroupBox("Cellpose Segmentation", self)
        self.groupbox.setCheckable(True)
        self.groupbox.setChecked(False)
        settings_groupbox_layout = QGridLayout(self.groupbox)
        settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        settings_groupbox_layout.setSpacing(5)
        settings_groupbox_layout.addWidget(model_wdg, 0, 0, 1, 2)
        settings_groupbox_layout.addWidget(self._browse_custom_model, 1, 0, 1, 2)
        settings_groupbox_layout.addWidget(channel_wdg, 2, 0, 1, 2)
        settings_groupbox_layout.addWidget(diameter_wdg, 3, 0, 1, 2)
        settings_groupbox_layout.addWidget(self._output_path, 4, 0, 1, 2)
        settings_groupbox_layout.addWidget(pos_wdg, 5, 0, 1, 2)
        settings_groupbox_layout.addWidget(progress_wdg, 6, 0, 1, 2)

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
        if self._data.sequence is None:
            return
        chs = self._data.sequence.sizes.get("c")
        if chs is not None:
            self._channel_combo.addItems([str(i) for i in range(chs)])

    @property
    def labels(self) -> dict[str, np.ndarray]:
        return self._labels

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is not None:
            self._worker.quit()
        self._elapsed_timer.stop()
        self._progress_bar.reset()
        self._progress_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def run(self) -> None:
        """Perform the Cellpose segmentation in a separate thread."""
        if self._worker is not None and self._worker.is_running:
            return

        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_label.setText("[0/0]")

        if self._data is None:
            return

        path = self._output_path.value()
        if not path:
            show_error_dialog(self, "Please select a Labels Output Path.")
            return

        sequence = self._data.sequence
        if sequence is None:
            show_error_dialog(self, "No useq.MDAsequence found!")
            return

        # use all positions if the input is empty
        if not self._pos_le.text():
            positions = list(range(len(sequence.stage_positions)))
        else:
            # parse the input positions
            positions = parse_lineedit_text(self._pos_le.text())

            if not positions:
                show_error_dialog(self, "Invalid Positions provided!")
                return

            if max(positions) >= len(sequence.stage_positions):
                show_error_dialog(self, "Input Positions out of range!")
                return

        self._progress_bar.setRange(0, len(positions))

        # ask the user if wants to overwrite the labels if they already exist
        if list(Path(path).glob("*.tif")):
            response = self._overwrite_msgbox()
            if response == QMessageBox.StandardButton.No:
                return
        # set the label path of the PlateViewer
        if self._plate_viewer is not None:
            self._plate_viewer.labels_path = path

        # set the model type
        if self._models_combo.currentText() == "custom":
            # get the path to the custom model
            custom_model_path = self._browse_custom_model.value()
            if not custom_model_path:
                show_error_dialog(self, "Please select a custom model path.")
                return
            model = CellposeModel(pretrained_model=custom_model_path)
        else:
            model_type = self._models_combo.currentText()
            model = models.Cellpose(gpu=True, model_type=model_type)

        # set the channel to segment
        channel = [self._channel_combo.currentIndex(), 0]

        # set the diameter
        diameter = self._diameter_spin.value() or None

        self._enable(False)

        self._elapsed_timer.start()

        self._worker = create_worker(
            self._segment,
            path=path,
            model=model,
            channel=channel,
            diameter=diameter,
            positions=positions,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress,
                "finished": self._on_worker_finished,
                "errored": self._on_worker_finished,
            },
        )

    def _overwrite_msgbox(self) -> Any:
        """Show a message box to ask the user if wants to overwrite the labels."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("The Labels directory already contains some files!")
        msg.setInformativeText("Do you want to overwrite them?")
        msg.setWindowTitle("Overwrite Labels")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        return msg.exec()

    def _segment(
        self,
        path: str,
        model: CellposeModel,
        channel: list[int],
        diameter: float,
        positions: list[int],
    ) -> Generator[str, None, None]:
        """Perform the segmentation using Cellpose."""
        if self._data is None:
            return

        # pos = self._data.sequence.sizes.get("p", 0)  # type: ignore
        # progress_bar = tqdm(range(pos))
        for p in tqdm(positions):
            if self._worker is not None and self._worker.abort_requested:
                break
            # get the data
            data, meta = self._data.isel(p=p, metadata=True)
            # get position name from metadata
            pos_name = (
                meta[0].get("Event", {}).get("pos_name", f"pos_{str(p).zfill(4)}")
            )
            yield f"[Well {pos_name} (p{p})]"
            # max projection from half to the end of the stack
            data_half_to_end = data[data.shape[0] // 2 :, :, :]
            # perform cellpose on each time point
            cyto_frame = data_half_to_end.max(axis=0)
            masks, _, _, _ = model.eval(cyto_frame, diameter=diameter, channels=channel)
            self._labels[f"{pos_name}_p{p}"] = masks
            # save to disk
            tifffile.imsave(Path(path) / f"{pos_name}_p{p}.tif", masks)

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()

    def _update_progress(self, state: str) -> None:
        """Update the progress bar with the current state."""
        self._progress_label.setText(state)
        self._progress_bar.setValue(self._progress_bar.value() + 1)

    def _update_progress_label(self, time_str: str) -> None:
        """Update the progress label with elapsed time."""
        self._elapsed_time_label.setText(time_str)

    def _on_worker_finished(self) -> None:
        """Enable the widgets when the segmentation is finished."""
        self._enable(True)
        self._elapsed_timer.stop()
        self._progress_bar.setValue(self._progress_bar.maximum())

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._models_combo.setEnabled(enable)
        self._browse_custom_model.setEnabled(enable)
        self._channel_combo.setEnabled(enable)
        self._output_path.setEnabled(enable)
        self._run_btn.setEnabled(enable)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(event)
