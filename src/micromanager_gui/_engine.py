from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
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

        # for GCaMP shutter pre-open delay
        self._gcamp_channel: str | None = None
        self._gcamp_delay_ms: float = 0.0

    def setArduinoBoard(self, arduino_board: Arduino | None) -> None:
        """Set the Arduino board to use for LED stimulation."""
        self._arduino_board = arduino_board

    def setArduinoLedPin(self, arduino_led_pin: Pin | None) -> None:
        """Set the pin on the Arduino board to use for LED stimulation."""
        self._arduino_led_pin = arduino_led_pin

    def setGCaMPChannel(self, channel: str | None) -> None:
        """Set the channel name that requires a shutter pre-open delay."""
        self._gcamp_channel = channel

    def setGCaMPDelayMs(self, delay_ms: float) -> None:
        """Set the shutter pre-open delay in milliseconds for the GCaMP channel."""
        self._gcamp_delay_ms = delay_ms

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

        if (
            self._gcamp_channel is not None
            and self._gcamp_delay_ms > 0
            and event.index.get("t") == 0
            and self._mmc.getCurrentConfig("Channels") == self._gcamp_channel
        ):
            self._mmc.setShutterOpen(True)
            time.sleep(self._gcamp_delay_ms / 1000)

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
        # Fire LED stimulation just before the sequence starts
        if self._stimulation_action is not None:
            self._exec_led_stimulation(self._stimulation_action.data)
            self._stimulation_action = None

        try:
            yield from super().exec_sequenced_event(event)
        except MemoryError:  # pragma: no cover
            if self._slackbot is not None:
                self._slackbot.send_message(
                    {"icon_emoji": ALARM_EMOJI, "text": "Buffer Overflowed!"}
                )
            raise

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Perform any teardown required after the sequence has been executed."""
        # close the current shutter at the end of the sequence
        if self._mmc.getShutterDevice():
            self._mmc.setShutterOpen(False)
        super().teardown_sequence(sequence)
