from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process
from typing import TYPE_CHECKING

from pymmcore_plus._logger import logger

if TYPE_CHECKING:
    import numpy as np
    import useq
    from pymmcore_plus import CMMCorePlus


class SegmentNeurons:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._is_running: bool = False

        self._segmentation_process: Process | None = None

        # Create a multiprocessing Queue
        self._queue: mp.Queue[np.ndarray | None] = mp.Queue()

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        self._is_running = True

        # create a separate process for segmentation
        self._segmentation_process = Process(
            target=_segmentation_worker, args=(self._queue,)
        )

        logger.info("SegmentNeurons -> Starting segmentation worker...")
        # start the segmentation process
        self._segmentation_process.start()

    def _on_frame_ready(self, image: np.ndarray, event: useq.MDAEvent) -> None:
        # if t=0, add the image to the queue
        t_index = event.index.get("t")
        if t_index is not None and t_index == 0:
            # send the image to the segmentation process
            self._queue.put(image)
            logger.info(f"SegmentNeurons -> Sending image to segment: {event.index}")

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        self._is_running = False

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
        image = queue.get()
        if image is None:
            break
        _segment_image(image)


def _segment_image(image: np.ndarray) -> None:
    """Segment the image."""
    logger.info(f"SegmentNeurons -> Segmenting image: {image.shape}")
