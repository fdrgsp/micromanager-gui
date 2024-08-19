from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process
from typing import TYPE_CHECKING

import numpy as np
from pymmcore_plus._logger import logger

if TYPE_CHECKING:
    import useq
    from pymmcore_plus import CMMCorePlus


IMAGES_MAX_PROJ = 50


class SegmentNeurons:
    """Segment neurons."""

    def __init__(self, mmcore: CMMCorePlus):
        self._mmc = mmcore

        self._enabled: bool = False

        self._is_running: bool = False

        self._segmentation_process: Process | None = None

        self._timepoints: int | None = None

        self._max_proj: np.ndarray | None = None

        # Create a multiprocessing Queue
        self._queue: mp.Queue[np.ndarray | None] = mp.Queue()

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def enable(self, enable: bool) -> None:
        """Enable or disable the segmentation."""
        self._enabled = enable

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

        t_index = event.index.get("t")
        if t_index is None or self._timepoints is None:
            return
        start_timepoint = (self._timepoints // 2) - 1
        end_timepoint = min(start_timepoint + IMAGES_MAX_PROJ, self._timepoints - 1)
        # create a max projection of the images for segmentation
        if t_index >= start_timepoint and t_index <= end_timepoint:
            self._max_proj = (
                image if self._max_proj is None else np.maximum(self._max_proj, image)
            )
            # when the max_proj is ready, send it to the segmentation process
            if t_index == end_timepoint:
                # send the max_proj image to the segmentation process
                self._queue.put(self._max_proj)
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
        image = queue.get()
        if image is None:
            break
        _segment_image(image)


def _segment_image(image: np.ndarray) -> None:
    """Segment the image."""
    logger.info(f"SegmentNeurons -> Segmenting image: {image.shape}")
