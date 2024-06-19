from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import tifffile
from cellpose import models
from cellpose.models import CellposeModel
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
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
from ._util import show_error_dialog

if TYPE_CHECKING:
    import numpy as np
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
        super().__init__(parent, label, "", tooltip, is_dir=True)

    def _on_browse(self) -> None:
        if path := QFileDialog.getExistingDirectory(
            self, f"Select the {self._label_text}.", self._current_path
        ):
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

        self._data = data

        self._labels: dict[str, np.ndarray] = {}

        self._worker: GeneratorWorker | None = None

        # TODO: maybe add diameter input

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
        if self.data is not None and self.data.sequence:
            chs = self.data.sequence.sizes.get("c")
            if chs is not None:
                self._channel_combo.addItems([str(i) for i in range(chs)])
        channel_layout.addWidget(self._channel_combo_label)
        channel_layout.addWidget(self._channel_combo)

        self._output_path = _BrowseWidget(
            self,
            "Labels Output Path",
            "",
            "Choose the path to save the labels.",
            is_dir=True,
        )

        btn_wdg = QWidget(self)
        btn_wdg_layout = QHBoxLayout(btn_wdg)
        btn_wdg_layout.setContentsMargins(0, 0, 0, 0)
        btn_wdg_layout.setSpacing(5)
        self._segment_btn = QPushButton("Segment")
        self._segment_btn.setSizePolicy(*FIXED)
        self._segment_btn.clicked.connect(self.segment)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.clicked.connect(self._cancel_segmentation)
        btn_wdg_layout.addWidget(self._segment_btn)
        btn_wdg_layout.addWidget(self._cancel_btn)

        # set the minimum width of the labels
        fixed_lbl_width = self._output_path._label.minimumSizeHint().width()
        self._models_combo_label.setMinimumWidth(fixed_lbl_width)
        self._channel_combo_label.setMinimumWidth(fixed_lbl_width)
        self._browse_custom_model._label.setMinimumWidth(fixed_lbl_width)

        progress_wdg = QWidget(self)
        progress_layout = QHBoxLayout(progress_wdg)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(5)
        self._progress_lbl = QLabel()
        self._progress_bar = QProgressBar(self)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._progress_lbl)

        self._settings_groupbox = QGroupBox("Cellpose Segmentation", self)
        _settings_groupbox_layout = QGridLayout(self._settings_groupbox)
        _settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        _settings_groupbox_layout.setSpacing(5)
        _settings_groupbox_layout.addWidget(model_wdg, 0, 0, 1, 2)
        _settings_groupbox_layout.addWidget(self._browse_custom_model, 1, 0, 1, 2)
        _settings_groupbox_layout.addWidget(channel_wdg, 2, 0, 1, 2)
        _settings_groupbox_layout.addWidget(self._output_path, 3, 0, 1, 2)
        _settings_groupbox_layout.addWidget(btn_wdg, 4, 0)
        _settings_groupbox_layout.addWidget(progress_wdg, 4, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self._settings_groupbox)
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

    def segment(self) -> None:
        """Perform the Cellpose segmentation in a separate thread."""
        self._progress_bar.reset()

        if self.data is None:
            return

        # ask the user if wants to overwrite the labels if they already exist
        if self._plate_viewer is not None and (
            self._plate_viewer._labels_path is not None
            and self._plate_viewer._labels_path == self._output_path.value()
            and list(Path(self._plate_viewer._labels_path).iterdir())
        ):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText(
                f"The Labels directory already exist: {self._plate_viewer._labels_path}"
            )
            msg.setInformativeText("Do you want to overwrite the labels?")
            msg.setWindowTitle("Overwrite Labels")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg.setDefaultButton(QMessageBox.StandardButton.No)
            response = msg.exec()
            if response == QMessageBox.StandardButton.No:
                return

        if self.data.sequence is not None:
            self._progress_bar.setRange(0, self.data.sequence.sizes.get("p", 0))

        path = self._output_path.value()
        if not path:
            show_error_dialog(self, "Please select a Labels Output Path.")
            return
        # set the label path of the PlateViewer
        if self._plate_viewer is not None:
            self._plate_viewer._labels_path = path

        # set the model type
        if self._models_combo.currentText() == "custom":
            # get the path to the custom model
            custom_model_path = self._browse_custom_model.value()
            if not custom_model_path:
                show_error_dialog(self, "Please select a custom model path.")
                return
            model_type = CellposeModel(custom_model_path)
        else:
            model_type = self._models_combo.currentText()

        model = models.Cellpose(gpu=True, model_type=model_type)

        # set the channel to segment
        channel = [self._channel_combo.currentIndex(), 0]

        self._enable(False)

        self._worker = create_worker(
            self._segment,
            path=path,
            model=model,
            channel=channel,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress,
                "finished": self._on_finished,
            },
        )

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()

    def _segment(
        self, path: str, model: CellposeModel, channel: list[int]
    ) -> Generator[str, None, None]:
        """Perform the segmentation using Cellpose."""
        if self.data is None:
            return

        pos = self.data.sequence.sizes.get("p", 0)  # type: ignore
        progress_bar = tqdm(range(pos))
        for p in progress_bar:
            # get the data
            data, meta = self.data.isel(p=p, metadata=True)
            # get position name from metadata
            pos_name = meta[0].get("Event", {}).get("pos_name", f"pos_{p}")
            yield f"Segmenting position {p+1} of {pos} (well {pos_name})"
            # max projection
            data_max = data.max(axis=0)
            # perform cellpose on each time point
            cyto_frame = data_max
            masks, _, _, _ = model.eval(cyto_frame, diameter=0, channels=channel)
            self._labels[f"{pos_name}_p{p}"] = masks
            # save to disk
            tifffile.imsave(Path(path) / f"{pos_name}_p{p}.tif", masks)

    def _cancel_segmentation(self) -> None:
        """Cancel the segmentation process."""
        if self._worker is not None:
            self._worker.quit()

    def _update_progress(self, state: str) -> None:
        """Update the progress bar with the current state."""
        self._progress_lbl.setText(state)
        self._progress_bar.setValue(self._progress_bar.value() + 1)

    def _on_finished(self) -> None:
        """Enable the widgets when the segmentation is finished."""
        self._enable(True)
        self._progress_bar.reset()
        self._progress_lbl.setText("Finished!")

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._models_combo.setEnabled(enable)
        self._browse_custom_model.setEnabled(enable)
        self._channel_combo.setEnabled(enable)
        self._output_path.setEnabled(enable)
        self._segment_btn.setEnabled(enable)
