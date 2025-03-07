from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
import torch
import useq
from cellpose import models
from cellpose.models import CellposeModel
from pymmcore_plus._logger import logger
from pymmcore_widgets.mda._save_widget import ALL_EXTENSIONS
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


CHANNEL = [0, 0]
DIAMETER = 0
CYTO = "cyto3"


class SegmentNeurons:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._enabled: bool = False

        self._is_running: bool = False

        self._segmentation_process: Process | None = None

        self._timepoints: int | None = None

        self._max_proj: np.ndarray | None = None

        self._model: CellposeModel

        # Create a multiprocessing Queue
        self._queue: mp.Queue[tuple[np.ndarray, dict, CellposeModel] | None] = (
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
        if model_type == "custom":
            self._model = CellposeModel(
                pretrained_model=model_path, gpu=use_gpu, device=dev
            )
        else:
            self._model = models.Cellpose(
                gpu=use_gpu, model_type=model_type, device=dev
            )

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        self._is_running = True

        if not self._enabled:
            return

        self._max_proj = None
        self._timepoints = None
        if sequence.time_plan is not None:
            self._timepoints = sequence.time_plan.num_timepoints() or None

        # create a separate process for segmentation
        self._segmentation_process = Process(
            target=_segmentation_worker, args=(self._queue,)
        )

        logger.info("SegmentNeurons -> Starting segmentation worker...")
        # start the segmentation process
        self._segmentation_process.start()

    def _on_frame_ready(self, image: np.ndarray, event: useq.MDAEvent) -> None:
        if not self._enabled:
            return

        if event.sequence is None:
            self._enabled = False
            logger.warning("SegmentNeurons -> No save directory found.")
            return

        # TODO: maybe move this logic to the other process
        t_index = event.index.get("t")
        if t_index is None or self._timepoints is None:
            return
        start_timepoint = (self._timepoints // 2) - 1
        # create a max projection of the images for segmentation
        if t_index >= start_timepoint and t_index <= self._timepoints - 1:
            self._max_proj = (
                image if self._max_proj is None else np.maximum(self._max_proj, image)
            )
            # when the max_proj is ready, send it to the segmentation process
            if t_index == self._timepoints - 1:
                # send the max_proj image to the segmentation process
                self._queue.put((self._max_proj, event.model_dump(), self._model))
                self._max_proj = None
                pos_idx = event.index.get("p", None)
                pos = f"(pos{pos_idx})" if pos_idx is not None else ""
                logger.info(f"SegmentNeurons -> Sending max_proj to segment {pos}.")

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        self._is_running = False

        if not self._enabled:
            return

        # stop the segmentation process
        self._queue.put(None)
        if self._segmentation_process is not None:
            self._segmentation_process.join()
        self._segmentation_process = None
        logger.info("SegmentNeurons -> Segmentation worker stopped.")


# this must not be part of the SegmentNeurons class
def _segmentation_worker(queue: mp.Queue) -> None:
    """Segmentation worker running in a separate process."""
    while True:
        args = queue.get()
        if args is None:
            break
        _segment_image(*args)


def _segment_image(image: np.ndarray, event: dict, model: CellposeModel) -> None:
    """Segment the image."""
    useq_event = useq.MDAEvent(**event)
    seq = useq_event.sequence
    if seq is None:
        logger.warning("SegmentNeurons -> No sequence found.")
        return

    # get metadata from the sequence
    meta = seq.metadata.get(PYMMCW_METADATA_KEY, {})
    save_dir = meta.get("save_dir")
    save_name = meta.get("save_name")

    if save_dir is None or save_name is None:
        logger.warning("SegmentNeurons -> No save directory found.")
        return

    # remove extension if present
    for ext in ALL_EXTENSIONS:
        if save_name.endswith(ext):
            save_name = save_name[: -len(ext)]
            break

    # create the save directory path
    save_dir = Path(save_dir) / f"{save_name}_labels"

    # make the save directory if it does not exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # create the labels file name
    p_idx = useq_event.index.get("p", 0)
    label_name = f"{useq_event.pos_name}_p{p_idx}.tif"

    # run cellpose
    logger.info(f"SegmentNeurons -> Segmenting image: {label_name}...")
    output = model.eval(image, diameter=DIAMETER, channels=CHANNEL)
    labels = output[0]

    # save to disk
    tifffile.imwrite(save_dir / label_name, labels)
    logger.info(f"SegmentNeurons -> Saving labels: {save_dir}/{label_name}")
