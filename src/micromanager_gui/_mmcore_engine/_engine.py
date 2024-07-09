from __future__ import annotations

import time
from itertools import product
from typing import (
    TYPE_CHECKING,
    Iterable,
)

from pymmcore_plus._logger import logger
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine
from rich import print
from useq import HardwareAutofocus, MDAEvent, MDASequence

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda._protocol import PImagePayload

    from micromanager_gui._slackbot._mm_slackbot import MMSlackBot

PYMMCW_METADATA_KEY = "pymmcore_widgets"
STIMULATION = "stimulation"
WARNING_EMOJI = ":warning:"
ALARM_EMOJI = ":rotating_light:"


class Engine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus,
        use_hardware_sequencing: bool = True,
        slackbot: MMSlackBot | None = None,
    ) -> None:
        super().__init__(mmc, use_hardware_sequencing)
        self._slackbot = slackbot

    def exec_event(self, event: MDAEvent) -> Iterable[PImagePayload]:
        """Execute an individual event and return the image data."""
        action = getattr(event, "action", None)
        if isinstance(action, HardwareAutofocus):
            print(f"***Autofocus Event: {event.index}, action: {action}***")
            # skip if no autofocus device is found
            if not self._mmc.getAutoFocusDevice():
                logger.warning("No autofocus device found. Cannot execute autofocus.")
                return ()  #  type: ignore

            try:
                # execute hardware autofocus
                new_correction = self._execute_autofocus(action)
                self._af_succeeded = True
            except RuntimeError as e:
                logger.warning("Hardware autofocus failed. %s", e)
                self._af_succeeded = False
                if self._slackbot is not None:
                    self._slackbot.send_message(
                        {
                            "icon_emoji": WARNING_EMOJI,
                            "text": f"Hardware autofocus failed: {e}!",
                        }
                    )
            else:
                # store correction for this position index
                p_idx = event.index.get("p", None)
                self._z_correction[p_idx] = new_correction + self._z_correction.get(
                    p_idx, 0.0
                )
            return ()  #  type: ignore

        # if the autofocus was engaged at the start of the sequence AND autofocus action
        # did not fail, re-engage it. NOTE: we need to do that AFTER the runner calls
        # `setup_event`, so we can't do it inside the exec_event autofocus action above.
        if self._af_was_engaged and self._af_succeeded:
            self._mmc.enableContinuousFocus(True)

        # open the shutter for x sec before starting the acquisition when using GCaMP6
        if (
            event.index.get("t", None) == 0
            and self._mmc.getCurrentConfig("Channels") == "GCaMP6"
        ):
            self._mmc.setShutterOpen(True)
            time.sleep(1)

        if isinstance(event, SequencedEvent):
            yield from self.exec_sequenced_event(event)
        else:
            yield from self.exec_single_event(event)

    def exec_sequenced_event(self, event: SequencedEvent) -> Iterable[PImagePayload]:
        """Execute a sequenced (triggered) event and return the image data.

        This method is not part of the PMDAEngine protocol (it is called by
        `exec_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.

        **Why Override?** to add slackbot notifications for buffer overflow.
        """
        n_events = len(event.events)

        t0 = event.metadata.get("runner_t0") or time.perf_counter()
        event_t0_ms = (time.perf_counter() - t0) * 1000

        # Start sequence
        # Note that the overload of startSequenceAcquisition that takes a camera
        # label does NOT automatically initialize a circular buffer.  So if this call
        # is changed to accept the camera in the future, that should be kept in mind.
        self._mmc.startSequenceAcquisition(
            n_events,
            0,  # intervalMS  # TODO: add support for this
            True,  # stopOnOverflow
        )
        self.post_sequence_started(event)

        n_channels = self._mmc.getNumberOfCameraChannels()
        count = 0
        iter_events = product(event.events, range(n_channels))
        # block until the sequence is done, popping images in the meantime
        while self._mmc.isSequenceRunning():
            if remaining := self._mmc.getRemainingImageCount():
                yield self._next_seqimg_payload(
                    *next(iter_events), remaining=remaining - 1, event_t0=event_t0_ms
                )
                count += 1
            else:
                time.sleep(0.001)

        if self._mmc.isBufferOverflowed():  # pragma: no cover
            if self._slackbot is not None:
                self._slackbot.send_message(
                    {"icon_emoji": ALARM_EMOJI, "text": "Buffer Overflowed!"}
                )
            raise MemoryError("Buffer overflowed")

        while remaining := self._mmc.getRemainingImageCount():
            yield self._next_seqimg_payload(
                *next(iter_events), remaining=remaining - 1, event_t0=event_t0_ms
            )
            count += 1

        # necessary?
        expected_images = n_events * n_channels
        if count != expected_images:
            logger.warning(
                "Unexpected number of images returned from sequence. "
                "Expected %s, got %s",
                expected_images,
                count,
            )

    def _next_seqimg_payload(
        self,
        event: MDAEvent,
        channel: int = 0,
        *,
        event_t0: float = 0.0,
        remaining: int = 0,
    ) -> PImagePayload:
        """Grab next image from the circular buffer and return it as an ImagePayload."""
        # TEMPORARY SOLUTION TO DTOP ACQUISITION
        if self._mmc.mda._wait_until_event(event):  # SLF001
            self._mmc.mda.cancel()
            self._mmc.stopSequenceAcquisition()
        return super()._next_seqimg_payload(event)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Perform any teardown required after the sequence has been executed."""
        # close the current shutter at the end of the sequence
        if self._mmc.getShutterDevice():
            self._mmc.setShutterOpen(False)
