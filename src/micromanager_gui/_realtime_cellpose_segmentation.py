from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import tifffile
import useq
from cellpose import core, models
from pymmcore_plus._logger import logger
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY

from ._widgets._mda_widget._mda_widget import SEGMENTATION
from ._widgets._mda_widget._realtime_cellpose_segmentation_wdg import CUSTOM, CYTO3
from ._widgets._mda_widget._save_widget import ALL_EXTENSIONS

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


class RealTimeCellposeSegmentation:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._enabled: bool = False

        self._is_running: bool = False

        self._segmentation_process: Process | None = None

        # Create a multiprocessing Queue
        self._queue: mp.Queue[tuple[np.ndarray, dict] | None] = mp.Queue()

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        self._is_running = True

        meta = sequence.metadata.get(PYMMCW_METADATA_KEY, {})
        self._enabled = meta.get(SEGMENTATION, False)

        if not self._enabled:
            return

        supported_axes = {("p", "t", "c")}
        if sequence.axis_order not in supported_axes:
            self._enabled = False
            logger.warning(
                "SegmentNeurons -> Currently Supported axis orders are "
                f"{supported_axes} but got {sequence.axis_order}."
            )
            return

        if len(sequence.channels) != 1:
            self._enabled = False
            logger.warning(
                f"SegmentNeurons -> Currently, only one channel is supported, "
                f"but got {len(sequence.channels)}."
            )
            return

        # create a separate process for segmentation
        self._segmentation_process = Process(
            target=_segmentation_worker, args=(self._queue, sequence)
        )

        logger.info("SegmentNeurons -> Starting segmentation worker...")
        # start the segmentation process
        self._segmentation_process.start()

    def _on_frame_ready(self, image: np.ndarray, event: useq.MDAEvent) -> None:
        if not self._enabled:
            return

        self._queue.put((image, event.model_dump()))

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        self._is_running = False

        if not self._enabled:
            return

        # stop the segmentation process
        self._queue.put(None)
        if self._segmentation_process is not None:
            logger.info(
                "SegmentNeurons -> Waiting for segmentation worker to finish..."
            )
            self._segmentation_process.join()
        self._segmentation_process = None
        logger.info("SegmentNeurons -> Segmentation worker stopped.")


# this must not be part of the SegmentNeurons class
def _segmentation_worker(queue: mp.Queue, sequence: useq.MDASequence) -> None:
    """Segmentation worker running in a separate process."""
    _max_proj: np.ndarray | None = None
    # this at this point of the code should never be None and thus never be triggered
    if (t_plan := sequence.time_plan) is None:
        return
    timepoints = t_plan.num_timepoints()
    start_timepoint = timepoints // 2 - 1

    while True:
        args = queue.get()
        if args is None:
            break

        image, event = args
        useq_event = useq.MDAEvent(**event)
        t_index = useq_event.index.get("t")
        if t_index is None:
            break

        # update the max projection
        if t_index >= start_timepoint and t_index <= timepoints - 1:
            _max_proj = image if _max_proj is None else np.maximum(_max_proj, image)

        # segment the max projection once it is ready
        if t_index == timepoints - 1 and _max_proj is not None:
            _segment_image(_max_proj, useq_event, sequence)
            _max_proj = None


def _segment_image(
    image: np.ndarray, event: useq.MDAEvent, sequence: useq.MDASequence
) -> None:
    """Segment the image."""
    # get saving metadata from the sequence
    meta = cast("dict", sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
    save_dir = cast("str", meta.get("save_dir", ""))
    save_name = cast("str", meta.get("save_name", ""))

    if not save_dir or not save_name:
        logger.warning(
            "SegmentNeurons -> No save directory found. Skipping segmentation."
        )
        return

    # remove extension if present
    for ext in ALL_EXTENSIONS:
        if save_name.endswith(ext):
            save_name = save_name[: -len(ext)]
            break

    # create the save directory path
    labels_dir = Path(save_dir) / f"{save_name}_labels"

    # make the save directory if it does not exist
    labels_dir.mkdir(parents=True, exist_ok=True)

    # create the labels file name
    p_idx = event.index.get("p", 0)
    if pos_name := event.pos_name:
        label_name = f"{pos_name}_p{p_idx}.tif"
    else:
        label_name = f"p{p_idx}.tif"

    # set the cellpose model and parameters
    model_info = cast("dict", meta.get(SEGMENTATION, {}))
    model_type: str = model_info.get("model_type", "")
    model_path: str = model_info.get("model_path", "")

    if not model_type:
        logger.warning(
            "SegmentNeurons -> No Cellpose model type found. Skipping segmentation."
        )
        return

    if model_type == CUSTOM and not model_path:
        logger.warning(
            "SegmentNeurons -> No Cellpose model path found. Skipping segmentation."
        )
        return

    use_gpu = core.use_gpu()
    if model_type == CUSTOM and model_path:
        model = models.CellposeModel(pretrained_model=model_path, gpu=use_gpu)
    else:
        model = models.Cellpose(gpu=use_gpu, model_type=model_type or CYTO3)

    # run cellpose
    logger.info(
        f"SegmentNeurons -> Segmenting image: {label_name}... "
        f"(gpu={use_gpu}, model_type={model_type}, model_path={model_path or 'None'})"
    )
    output = model.eval(image)

    # save to disk
    logger.info(f"SegmentNeurons -> Saving labels: {labels_dir}/{label_name}")
    tifffile.imwrite(labels_dir / label_name, output[0])
