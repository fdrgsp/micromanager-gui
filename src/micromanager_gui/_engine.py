from __future__ import annotations

import time
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pymmcore_plus._logger import logger
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine
from useq import CustomAction, HardwareAutofocus, MDAEvent, MDASequence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pyfirmata2 import Arduino
    from pyfirmata2.pyfirmata2 import Pin
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda._protocol import PImagePayload
    from pymmcore_plus.metadata import SummaryMetaV1

    from micromanager_gui._slackbot._mm_slackbot import MMSlackBot

WARNING_EMOJI = ":warning:"
ALARM_EMOJI = ":rotating_light:"


class ArduinoEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus,
        use_hardware_sequencing: bool = True,
        arduino_board: Arduino | None = None,
        arduino_led_pin: Pin | None = None,
        slackbot: MMSlackBot | None = None,
    ) -> None:
        super().__init__(mmc, use_hardware_sequencing=use_hardware_sequencing)

        self._mmc = self.mmcore
        self._slackbot = slackbot

        # for LED stimulation
        self._arduino_board = arduino_board
        self._arduino_led_pin = arduino_led_pin

        self._stimulation_action: CustomAction | None = None

    def setArduinoBoard(self, arduino_board: Arduino | None) -> None:
        """Set the Arduino board to use for LED stimulation."""
        self._arduino_board = arduino_board

    def setArduinoLedPin(self, arduino_led_pin: Pin | None) -> None:
        """Set the pin on the Arduino board to use for LED stimulation."""
        self._arduino_led_pin = arduino_led_pin

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup the hardware for the entire sequence."""
        # Arduino LED Setup
        if self._arduino_board is not None and self._arduino_led_pin is not None:
            self._arduino_led_pin = cast("Pin", self._arduino_led_pin)
            self._arduino_led_pin.write(0.0)
        return super().setup_sequence(sequence)

    def exec_event(self, event: MDAEvent) -> Iterable[PImagePayload]:
        """Execute an individual event and return the image data."""
        action = getattr(event, "action", None)

        if isinstance(action, HardwareAutofocus):
            # skip if no autofocus device is found
            if not self._mmc.getAutoFocusDevice():
                logger.warning("No autofocus device found. Cannot execute autofocus.")
                return

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
            return

        if (
            isinstance(action, CustomAction)
            and action.type == "custom"
            and action.name == "arduino_stimulation"
            and action.data
        ):
            # here I am only setting the stimulation action info so that I
            # will execute the stimulation just before starting the sequenced event
            self._stimulation_action = action
            return

        # if the autofocus was engaged at the start of the sequence AND autofocus action
        # did not fail, re-engage it. NOTE: we need to do that AFTER the runner calls
        # `setup_event`, so we can't do it inside the exec_event autofocus action above.
        if self._arduino_board is None and self._af_was_engaged and self._af_succeeded:
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

    def _exec_led_stimulation(self, data: dict) -> None:
        """Execute LED stimulation."""
        led_power = data.get("led_power", 0)
        led_pulse_duration = data.get("led_pulse_duration", 0)
        self._arduino_board = cast("Arduino", self._arduino_board)
        self._arduino_led_pin = cast("Pin", self._arduino_led_pin)
        led_pulse_duration = led_pulse_duration / 1000  # convert to sec
        # switch on the LED
        self._arduino_led_pin.write(led_power / 100)
        # wait for the duration of the pulse
        time.sleep(led_pulse_duration)
        # switch off the LED
        self._arduino_led_pin.write(0)

    def exec_sequenced_event(self, event: SequencedEvent) -> Iterable[PImagePayload]:
        """Execute a sequenced (triggered) event and return the image data.

        This method is not part of the PMDAEngine protocol (it is called by
        `exec_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        # NOTE: only overriding because of the "Buffer Overflowed" slakbot message

        n_events = len(event.events)

        t0 = event.metadata.get("runner_t0") or time.perf_counter()
        event_t0_ms = (time.perf_counter() - t0) * 1000

        if event.slm_image is not None:
            self._exec_event_slm_image(event.slm_image)

        # execute LED stimulation if it was requested
        if self._stimulation_action is not None:
            self._exec_led_stimulation(self._stimulation_action.data)
            self._stimulation_action = None

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
        self, event: MDAEvent, *args: Any, **kwargs: Any
    ) -> PImagePayload:
        """Grab next image from the circular buffer and return it as an ImagePayload."""
        # TEMPORARY SOLUTION to cancel the sequence acquisition
        if self._mmc.mda._wait_until_event(event):  # SLF001
            self._mmc.mda.cancel()
            self._mmc.stopSequenceAcquisition()
        return super()._next_seqimg_payload(event, *args, **kwargs)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Perform any teardown required after the sequence has been executed."""
        # close the current shutter at the end of the sequence
        if self._mmc.getShutterDevice():
            self._mmc.setShutterOpen(False)
        super().teardown_sequence(sequence)
