from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING

import tifffile
import torch
from cellpose import models
from cellpose.models import CellposeModel
from pymmcore_plus._logger import logger
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from superqt.utils import create_worker

from micromanager_gui.readers import OMEZarrReader, TensorstoreZarrReader

from ._widgets._mda_widget._save_widget import (
    ALL_EXTENSIONS,
    EXT_TO_WRITER,
    OME_ZARR,
    ZARR_TESNSORSTORE,
)

if TYPE_CHECKING:
    import numpy as np
    import useq
    from pymmcore_plus import CMMCorePlus


CHANNEL = [0, 0]
DIAMETER = 0
CUSTOM = "custom"
CYTO = "cyto3"


class SegmentAndAnalyse:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._enabled: bool = False

        self._is_running: bool = False

        self._segment_and_analise_process: Process | None = None

        self._timepoints: int | None = None

        self._model: CellposeModel

        self._save_path_and_name: tuple[str, str] | None = None

        # Create a multiprocessing Queue
        self._queue: mp.Queue[tuple[str, tuple[str, str], CellposeModel] | None] = (
            mp.Queue()
        )

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def enable(
        self, enable: bool, model_type: str = CYTO, model_path: str = ""
    ) -> None:
        """Enable or disable the segmentation."""
        self._enabled = enable
        self.set_model(model_type, model_path)

    def set_model(self, model_type: str, model_path: str) -> None:
        """Set the cellpose model."""
        use_gpu = torch.cuda.is_available()
        dev = torch.device("cuda" if use_gpu else "cpu")
        if model_type == CUSTOM:
            self._model = CellposeModel(
                pretrained_model=model_path, gpu=use_gpu, device=dev
            )
        else:  # cyto3
            self._model = models.Cellpose(
                gpu=use_gpu, model_type=model_type, device=dev
            )

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        self._is_running = True

        if not self._enabled:
            return

        # get metadata from the sequence
        meta = sequence.metadata.get(PYMMCW_METADATA_KEY, {})
        save_name: str = meta.get("save_name")
        self._save_path_and_name = (meta.get("save_dir"), save_name)

        # continue only if the ZARR_TESNSORSTORE or OME_ZARR
        for ext in ALL_EXTENSIONS:
            if save_name.endswith(ext) and ext in {ZARR_TESNSORSTORE, OME_ZARR}:
                self._enabled = False
                return

        self._timepoints = None
        if sequence.time_plan is not None:
            self._timepoints = sequence.time_plan.num_timepoints() or None

        # create a separate process for segmentation and analysis
        self._segment_and_analise_process = Process(
            target=_segment_and_analyse_process, args=(self._queue,)
        )

        logger.info("SegmentAndAnalyse -> Starting worker...")
        # start the segmentation process
        self._segment_and_analise_process.start()

    def _on_frame_ready(self, image: np.ndarray, event: useq.MDAEvent) -> None:
        if not self._enabled or self._save_path_and_name is None:
            return

        if (t_index := event.index.get("t")) is None or self._timepoints is None:
            return

        # send it to the segmentation and analysis process
        if t_index == self._timepoints - 1:
            # create the labels file name
            p_idx = event.index.get("p", 0)

            pos_name = (
                event.pos_name
                if event.pos_name is not None
                else f"pos_{str(p_idx).zfill(4)}"
            )
            label_name = f"{pos_name}_p{p_idx}"

            # send the max_proj image to the segmentation process
            self._queue.put((label_name, self._save_path_and_name, self._model))

            logger.info(
                f"SegmentAndAnalyse -> Sending info to segment and analyse {pos_name}."
            )

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        self._is_running = False

        if not self._enabled:
            return

        # stop the segmentation process
        self._queue.put(None)
        if self._segment_and_analise_process is not None:
            self._segment_and_analise_process.join()
        self._segment_and_analise_process = None
        logger.info("SegmentAndAnalyse -> Segmentation worker stopped.")


# this must not be part of the SegmentAndAnalyse class
def _segment_and_analyse_process(queue: mp.Queue) -> None:
    """Segmentation worker running in a separate process."""
    while True:
        args = queue.get()
        if args is None:
            break
        _segment_and_analyse(*args)


def _segment_and_analyse(
    label_name: str, save_path_and_name: tuple[str, str], model: CellposeModel
) -> None:
    """Segment the image."""
    logger.info(f"SegmentAndAnalyse -> received: {label_name}")

    save_path, save_name = save_path_and_name

    # build datastore path
    datastore_path = Path(save_path) / f"{save_name}"  # has extension

    # get the writer and remove the extension to create the labels directory
    writer: str = ""
    for ext in ALL_EXTENSIONS:
        if save_name.endswith(ext):
            save_name = save_name[: -len(ext)]
            writer = EXT_TO_WRITER[ext]
            break

    # load the data
    datastore: TensorstoreZarrReader | OMEZarrReader
    if writer == ZARR_TESNSORSTORE:
        datastore = TensorstoreZarrReader(datastore_path)
    elif writer == OME_ZARR:
        datastore = OMEZarrReader(datastore_path)
    else:
        return

    # create the save directory path and make the save directory if it does not exist
    save_dir = Path(save_path) / f"{save_name}_labels"
    save_dir.mkdir(parents=True, exist_ok=True)

    # get the position index from label_name
    p_idx = int(label_name.split("_")[-1][1:])  # A1_0020_p23 -> p23-> 23

    # get data and metadata from the datastore for the position index
    data, meta = datastore.isel(p=p_idx, metadata=True)

    # SEGMENTATION - CELLPOSE --------------------------------------------------------
    # max projection from half to the end of the stack
    data_half_to_end = data[data.shape[0] // 2 :, :, :]
    max_proj = data_half_to_end.max(axis=0)

    # run cellpose
    logger.info(f"SegmentAndAnalyse -> Segmenting image: {label_name}...")
    output = model.eval(max_proj, diameter=DIAMETER, channels=CHANNEL)
    labels = output[0]

    # save to disk
    tifffile.imwrite(save_dir / label_name, labels)
    logger.info(f"SegmentAndAnalyse -> Saving labels: {save_dir}/{label_name}.tif")

    # ANALYSIS -----------------------------------------------------------------------

    create_worker(
        _extract_traces,
        data,
        labels,
        _start_thread=True,
        # _connect={
        #     "yielded": _show_and_log_error,
        #     "finished": _on_worker_finished,
        #     "errored": _on_worker_finished,
        # },
    )


def _extract_traces(data: np.ndarray, labels: np.ndarray) -> None:
    """Extract traces."""
    pass
