from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile
from cellpose import core, models
from cellpose.models import CellposeModel
from fonticon_mdi6 import MDI6
from qtpy.QtCore import QSize, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
from superqt.fonticon import icon
from superqt.utils import create_worker
from tqdm import tqdm

from ._logger import LOGGER
from ._util import (
    GREEN,
    RED,
    _BrowseWidget,
    _ElapsedTimer,
    parse_lineedit_text,
    show_error_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np
    from qtpy.QtGui import QCloseEvent
    from superqt.utils import GeneratorWorker

    from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

    from ._plate_viewer import PlateViewer


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

CUSTOM_MODEL_PATH = (
    Path(__file__).parent.parent
    / "_cellpose"
    / "cellpose_models"
    / "cp3_img8_epoch7000_py"
)


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
    """Widget to perform Cellpose segmentation on a PlateViewer data."""

    segmentationFinished = Signal()

    def __init__(
        self,
        parent: PlateViewer | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
    ) -> None:
        super().__init__(parent)

        self._plate_viewer: PlateViewer | None = parent
        self._data: TensorstoreZarrReader | OMEZarrReader | None = data
        self._labels_path: str | None = None
        self._labels: dict[str, np.ndarray] = {}
        self._worker: GeneratorWorker | None = None

        # ELAPSED TIMER ---------------------------------------------------------
        self._elapsed_timer = _ElapsedTimer()
        self._elapsed_timer.elapsed_time_updated.connect(self._update_progress_label)

        # MODEL WIDGET ----------------------------------------------------------
        self._model_wdg = QWidget(self)
        model_wdg_layout = QHBoxLayout(self._model_wdg)
        model_wdg_layout.setContentsMargins(0, 0, 0, 0)
        model_wdg_layout.setSpacing(5)
        self._models_combo_label = QLabel("Model Type:")
        self._models_combo_label.setSizePolicy(*FIXED)
        self._models_combo = QComboBox()
        # self._models_combo.addItems(["nuclei", "cyto", "cyto2", "cyto3", "custom"])
        self._models_combo.addItems(["cyto3", "custom"])
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)
        # self._use_gpu_checkbox = QCheckBox("Use GPU")
        # self._use_gpu_checkbox.setToolTip("Run Cellpose on the GPU.")
        # self._use_gpu_checkbox.setChecked(True)
        model_wdg_layout.addWidget(self._models_combo_label)
        model_wdg_layout.addWidget(self._models_combo, 1)
        # model_wdg_layout.addWidget(self._use_gpu_checkbox)

        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        # DIAMETER WIDGETS ------------------------------------------
        self._diameter_wdg = QWidget(self)
        diameter_layout = QHBoxLayout(self._diameter_wdg)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.setSpacing(5)
        self._diameter_label = QLabel("Diameter:")
        self._diameter_label.setSizePolicy(*FIXED)
        self._diameter_spin = QDoubleSpinBox(self)
        self._diameter_spin.setRange(0, 1000)
        self._diameter_spin.setValue(0)
        self._diameter_spin.setToolTip(
            "Set the diameter of the cells. Leave 0 for automatic detection."
        )
        diameter_layout.addWidget(self._diameter_label)
        diameter_layout.addWidget(self._diameter_spin)

        # POSITIONS WIDGET ------------------------------------------------------
        self._pos_wdg = QWidget(self)
        self._pos_wdg.setToolTip(
            "Select the Positions to segment. Leave blank to segment all Positions. "
            "You can input single Positions (e.g. 30, 33) a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65). "
            "NOTE: The Positions are 0-indexed."
        )
        pos_wdg_layout = QHBoxLayout(self._pos_wdg)
        pos_wdg_layout.setContentsMargins(0, 0, 0, 0)
        pos_wdg_layout.setSpacing(5)
        pos_lbl = QLabel("Segment Positions:")
        pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit()
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33. Leave empty for all.")
        pos_wdg_layout.addWidget(pos_lbl)
        pos_wdg_layout.addWidget(self._pos_le)

        # PROGRESS BAR WIDGET ---------------------------------------------------
        progress_wdg = QWidget(self)
        progress_layout = QHBoxLayout(progress_wdg)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(5)

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

        self._progress_label = QLabel("[0/0]")
        self._progress_bar = QProgressBar(self)
        self._elapsed_time_label = QLabel("00:00:00")

        # STYLING ---------------------------------------------------------------
        fixed_lbl_width = pos_lbl.sizeHint().width()
        self._models_combo_label.setMinimumWidth(fixed_lbl_width)
        self._browse_custom_model._label.setMinimumWidth(fixed_lbl_width)
        self._diameter_label.setMinimumWidth(fixed_lbl_width)

        # LAYOUT ----------------------------------------------------------------
        progress_layout.addWidget(self._run_btn)
        progress_layout.addWidget(self._cancel_btn)
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._elapsed_time_label)

        self.groupbox = QGroupBox("Cellpose Segmentation", self)
        settings_groupbox_layout = QVBoxLayout(self.groupbox)
        settings_groupbox_layout.setContentsMargins(10, 10, 10, 10)
        settings_groupbox_layout.setSpacing(5)
        settings_groupbox_layout.addWidget(self._model_wdg)
        settings_groupbox_layout.addWidget(self._browse_custom_model)
        settings_groupbox_layout.addWidget(self._diameter_wdg)
        settings_groupbox_layout.addSpacing(10)
        settings_groupbox_layout.addWidget(self._pos_wdg)
        settings_groupbox_layout.addSpacing(10)
        settings_groupbox_layout.addWidget(progress_wdg)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.groupbox)
        main_layout.addStretch(1)

    @property
    def data(
        self,
    ) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader | None) -> None:
        self._data = data

    @property
    def labels(self) -> dict[str, np.ndarray]:
        return self._labels

    @property
    def labels_path(self) -> str | None:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, labels_path: str | None) -> None:
        self._labels_path = labels_path

    # PUBLIC METHODS ------------------------------------------------------------------

    def run(self) -> None:
        """Perform the Cellpose segmentation in a separate thread."""
        if self._worker is not None and self._worker.is_running:
            return

        self._reset_progress_bar()

        if not self._validate_segmentation_setup():
            return

        positions = self._get_positions()
        if positions is None:
            return

        if not self._handle_existing_labels():
            return

        self._start_segmentation_thread(positions)

    def cancel(self) -> None:
        """Cancel the current run."""
        if self._worker is not None:
            self._worker.quit()
        self._elapsed_timer.stop()
        self._reset_progress_bar()
        self._enable(True)
        LOGGER.info("Cellpose segmentation canceled.")

    # PRIVATE METHODS -----------------------------------------------------------------

    # PREPARE FOR RUN -----------------------------------------------------------------
    def _validate_segmentation_setup(self) -> bool:
        """Check if the necessary data is available before segmentation."""
        if self._data is None:
            return False

        if not self._labels_path:
            LOGGER.error("No Labels Output Path selected.")
            show_error_dialog(
                self,
                "Please select a Labels Output Path.\n"
                "You can do this in `File > Load Data and Set Directories...' "
                "and set the `Segmentation Path'.",
            )
            return False

        if not Path(self._labels_path).is_dir():
            LOGGER.error("Invalid Segmentation Path.")
            show_error_dialog(
                self,
                "The `Segmentation Path` is not a valid directory!\n"
                "Please select a valid directory "
                "in `File > Load Data and Set Directories...`.",
            )
            return False

        sequence = self._data.sequence
        if sequence is None:
            msg = "No useq.MDAsequence found!"
            LOGGER.error(msg)
            show_error_dialog(self, msg)
            return False

        return True

    def _get_positions(self) -> list[int] | None:
        """Retrieve and validate the positions for segmentation."""
        # this should never happen, it has been checked in _validate_segmentation_setup
        if self._data is None or (sequence := self._data.sequence) is None:
            return None

        if not self._pos_le.text():
            return list(range(len(sequence.stage_positions)))

        positions = parse_lineedit_text(self._pos_le.text())
        if not positions or max(positions) >= len(sequence.stage_positions):
            msg = "Invalid or out-of-range Positions provided!"
            LOGGER.error(msg)
            show_error_dialog(self, msg)
            return None

        return positions

    def _handle_existing_labels(self) -> bool:
        """Check if label files exist and ask the user for overwrite confirmation."""
        if not (path := self._labels_path):
            # at this point, the path should always be set, adding for typing
            return False
        if list(Path(path).glob("*.tif")):
            response = self._overwrite_msgbox()
            if response == QMessageBox.StandardButton.No:
                return False
        return True

    # RUN THE SEGMENTATION ------------------------------------------------------------

    def _start_segmentation_thread(self, positions: list[int]) -> None:
        """Prepare segmentation and start it in a separate thread."""
        model = self._initialize_model()
        if model is None:
            return

        self._progress_bar.setRange(0, len(positions))
        self._enable(False)
        self._elapsed_timer.start()

        self._worker = create_worker(
            self._segment,
            path=self._labels_path,
            model=model,
            diameter=self._diameter_spin.value() or None,
            positions=positions,
            _start_thread=True,
            _connect={
                "yielded": self._update_progress_bar,
                "finished": self._on_worker_finished,
                "errored": self._on_worker_finished,
            },
        )

    def _segment(
        self,
        path: str,
        model: CellposeModel,
        diameter: float | None,
        positions: list[int],
    ) -> Generator[str | int, None, None]:
        """Perform the segmentation using Cellpose."""
        LOGGER.info("Starting Cellpose segmentation.")

        if self._data is None:
            return

        for p in tqdm(positions):
            if self._worker is not None and self._worker.abort_requested:
                break
            # get the data
            data, meta = self._data.isel(p=p, metadata=True)

            # get position name from metadata (in old metadata, the key was "Event")
            key = "mda_event" if "mda_event" in meta[0] else "Event"
            pos_name = meta[0].get(key, {}).get("pos_name", f"pos_{str(p).zfill(4)}")
            # yield the current position name to update the progress bar
            yield f"[Well {pos_name} p{p} (tot {len(positions)})]"
            # max projection from half to the end of the stack
            data_half_to_end = data[data.shape[0] // 2 :, :, :]
            # perform cellpose on each time point
            cyto_frame = data_half_to_end.max(axis=0)
            output = model.eval(cyto_frame, diameter=diameter)
            labels = output[0]
            # store the masks in the labels dict
            self._labels[f"{pos_name}_p{p}"] = labels
            # yield the current position to update the progress bar
            if len(positions) == 1:
                yield p
            elif len(positions) > 1:
                if p + 1 > len(positions):
                    yield len(positions)
                else:
                    yield p + 1
            # save to disk
            tifffile.imwrite(Path(path) / f"{pos_name}_p{p}.tif", labels)

    def _on_worker_finished(self) -> None:
        """Enable the widgets when the segmentation is finished."""
        LOGGER.info("Cellpose segmentation finished.")
        self._enable(True)
        self._elapsed_timer.stop()
        self._progress_bar.setValue(self._progress_bar.maximum())
        self.segmentationFinished.emit()

    # WIDGET---------------------------------------------------------------------------

    def _enable(self, enable: bool) -> None:
        """Enable or disable the widgets."""
        self._model_wdg.setEnabled(enable)
        self._browse_custom_model.setEnabled(enable)
        self._diameter_wdg.setEnabled(enable)
        self._pos_wdg.setEnabled(enable)
        self._run_btn.setEnabled(enable)
        if self._plate_viewer is None:
            return
        self._plate_viewer._analysis_wdg.setEnabled(enable)
        self._plate_viewer._traces_extraction_wdg.setEnabled(enable)
        # disable graphs tabs
        self._plate_viewer._tab.setTabEnabled(1, enable)
        self._plate_viewer._tab.setTabEnabled(2, enable)

    def _reset_progress_bar(self) -> None:
        """Reset and initialize progress bar."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_label.setText("[0/0]")
        self._elapsed_time_label.setText("00:00:00")

    def _initialize_model(self) -> CellposeModel | None:
        """Initialize the Cellpose model based on user selection."""
        use_gpu = core.use_gpu()
        LOGGER.info(f"Use GPU: {use_gpu}")

        if self._models_combo.currentText() == "custom":
            custom_model_path = self._browse_custom_model.value()
            if not custom_model_path:
                show_error_dialog(self, "Please select a custom model path.")
                LOGGER.error("No custom model path selected.")
                return None
            return CellposeModel(pretrained_model=custom_model_path, gpu=use_gpu)

        return models.Cellpose(gpu=use_gpu, model_type=self._models_combo.currentText())

    def _overwrite_msgbox(self) -> Any:
        """Show a message box to ask the user if wants to overwrite the labels."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(
            "The Labels directory already contains some files!\n\n"
            "Do you want to overwrite them?"
        )
        msg.setWindowTitle("Overwrite Labels")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        return msg.exec()

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()

    def _update_progress_bar(self, value: str | int) -> None:
        # update only the progress label if the value is a string
        if isinstance(value, str):
            self._progress_label.setText(value)
            return
        # update the progress bar value if the value is an integer
        self._progress_bar.setValue(value)

    def _update_progress_label(self, time_str: str) -> None:
        """Update the progress label with elapsed time."""
        self._elapsed_time_label.setText(time_str)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Override the close event to cancel the worker."""
        if self._worker is not None:
            self._worker.quit()
        super().closeEvent(a0)
