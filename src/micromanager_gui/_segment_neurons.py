from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path
from typing import TYPE_CHECKING, cast

import tifffile
import useq
from cellpose import models
from pymmcore_plus._logger import logger
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY

if TYPE_CHECKING:
    import numpy as np
    from pymmcore_plus import CMMCorePlus

# use this to select which frame to pick for segmentation. Maybe we can find a way to
# average multiple frames for better segmentation.
T_INDEX = 0


class SegmentNeurons:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._run_cellpose: bool = False

        self._segmentation_process: Process | None = None

        # Create a multiprocessing Queue
        self._queue: mp.Queue[tuple[np.ndarray, dict] | None] = mp.Queue()

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        if sequence.metadata.get(PYMMCW_METADATA_KEY, {}).get("should_save", False):
            self._run_cellpose = True
        else:
            self._run_cellpose = False
            return

        # create a separate process for segmentation
        self._segmentation_process = Process(
            target=_segmentation_worker, args=(self._queue,)
        )

        logger.info("SegmentNeurons -> Starting segmentation worker...")
        # start the segmentation process
        self._segmentation_process.start()

    def _on_frame_ready(self, image: np.ndarray, event: useq.MDAEvent) -> None:
        if not self._run_cellpose:
            return

        # if t=0, add the image to the queue
        t_index = event.index.get("t")
        if t_index is not None and t_index == T_INDEX:
            # send the image to the segmentation process
            self._queue.put((image, event.model_dump()))
            logger.info(f"SegmentNeurons -> Sending image to segment: {event.index}")

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        if not self._run_cellpose:
            return

        # stop the segmentation process
        self._queue.put(None)
        if self._segmentation_process is not None:
            self._segmentation_process.join()
        self._segmentation_process = None
        logger.info("SegmentNeurons -> Segmentation worker stopped.")

        self._run_cellpose = False


# this must not be part of the SegmentNeurons class
def _segmentation_worker(queue: mp.Queue) -> None:
    """Segmentation worker running in a separate process."""
    while True:
        args = queue.get()
        if args is None:
            break
        image, event = args
        _segment_image(image, event)


def _segment_image(image: np.ndarray, event: dict) -> None:
    """Segment the image."""
    logger.info(f"SegmentNeurons -> Segmenting image: {image.shape}")

    useq_event = useq.MDAEvent(**event)
    seq = useq_event.sequence
    if seq is None:
        return

    # get position index
    p_idx = useq_event.index.get("p", 0)

    # get the metadata from the sequence
    seq_meta = cast(dict, seq.metadata.get(PYMMCW_METADATA_KEY, {}))

    # get the save path
    save_path = seq_meta.get("save_dir", "")

    # create the labels save name
    save_name = seq_meta.get("save_name", "")

    # create a labels folder if it does not exist
    save_path = Path(save_path) / f"{save_name}_labels"
    save_path.mkdir(parents=True, exist_ok=True)

    # create the labels file name
    label_name = f"{save_name}_labels_{useq_event.pos_name}_p{p_idx}.tif"

    # set the cellpose model and parameters
    model_type = "cyto3"  # or CellposeModel(custom_model_path)
    channel = [0, 0]
    diameter = 0
    model = models.Cellpose(gpu=True, model_type=model_type)

    # run cellpose
    masks, _, _, _ = model.eval(image, diameter=diameter, channels=channel)

    # save to disk
    tifffile.imsave(save_path / label_name, masks)
    logger.info(f"SegmentNeurons -> Saving labels: {save_path}/{label_name}")
