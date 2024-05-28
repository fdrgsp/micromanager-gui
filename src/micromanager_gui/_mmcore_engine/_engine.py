from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    cast,
)

from pyfirmata2 import Arduino
from pyfirmata2.pyfirmata2 import Pin
from pymmcore_plus._logger import logger
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine
from pymmcore_plus.mda._engine import ImagePayload
from rich import print
from useq import AcquireImage, HardwareAutofocus, MDAEvent, MDASequence

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda._protocol import PImagePayload

    from micromanager_gui._slackbot import SlackBot

PYMMCW_METADATA_KEY = "pymmcore_widgets"
STIMULATION = "stimulation"


class ArduinoEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus,
        use_hardware_sequencing: bool = True,
        arduino_board: Arduino | None = None,
        arduino_led_pin: Pin | None = None,
        slackbot: SlackBot | None = None,
    ) -> None:
        super().__init__(mmc, use_hardware_sequencing)
        self._slackbot = slackbot

        # for LED stimulation
        self._arduino_board = arduino_board
        self._arduino_led_pin = arduino_led_pin
        self._exec_stimulation: dict[int, tuple[int, int]] = {}

    def setArduinoBoard(self, arduino_board: Arduino | None) -> None:
        """Set the Arduino board to use for LED stimulation."""
        self._arduino_board = arduino_board

    def setArduinoLedPin(self, arduino_led_pin: Pin | None) -> None:
        """Set the pin on the Arduino board to use for LED stimulation."""
        self._arduino_led_pin = arduino_led_pin

    def setup_sequence(self, sequence: MDASequence) -> Mapping[str, Any]:
        """Setup the hardware for the entire sequence."""
        # Arduino LED Setup
        self._exec_stimulation.clear()
        if self._arduino_board is not None and self._arduino_led_pin is not None:
            self._setup_stimulation_events(sequence)
        return super().setup_sequence(sequence)  # type: ignore

    def _setup_stimulation_events(self, sequence: MDASequence) -> None:
        # switch off the LED if it was on
        self._arduino_led_pin = cast(Pin, self._arduino_led_pin)
        self._arduino_led_pin.write(0.0)
        # get metadata from the sequence and store it in the _exec_stimulation
        meta = cast(dict, sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        stim_meta = cast(dict, meta.get(STIMULATION, {}))
        pulse_on_frame = stim_meta.get("pulse_on_frame", None)
        led_pulse_duration = stim_meta.get("led_pulse_duration", None)
        if pulse_on_frame is not None and led_pulse_duration is not None:
            # create the _exec_stimulation dict with info about when to pulse the
            # LED, for how long and with what power
            # e.g. {frame: led_power, led_pulse_duration}
            pulse_on_frame = cast(dict, pulse_on_frame)
            for k, v in pulse_on_frame.items():
                self._exec_stimulation[k] = (v, led_pulse_duration)

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
                    self._slackbot.send_message(f"⚠️ Hardware autofocus failed: {e}! ⚠️")
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

        # execute stimulation if the event if it is in the sequence metadata
        # if self._arduino_board is not None and self._arduino_led_pin is not None:
        if t_index := event.index.get("t", None):
            if t_index in self._exec_stimulation:
                self._exec_led_stimulation(t_index, event)

        if isinstance(event, SequencedEvent):
            yield from self.exec_sequenced_event(event)
        else:
            yield from self.exec_single_event(event)

    def _exec_led_stimulation(self, t_index: int, event: MDAEvent) -> None:
        """Execute LED stimulation."""
        self._arduino_board = cast(Arduino, self._arduino_board)
        self._arduino_led_pin = cast(Pin, self._arduino_led_pin)
        led_power = self._exec_stimulation[t_index][0]
        led_pulse_duration = self._exec_stimulation[t_index][1] / 1000  # convert to sec

        print(
            f"\n***Stimulation Event: {event.index}, "
            f"LED: {self._arduino_led_pin}, "
            f"LED Pulse Duration: {led_pulse_duration * 1000} ms, "
            f"LED Power: {led_power} %***\n"
        )

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
        # TODO: add support for multiple camera devices
        n_events = len(event.events)

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

        count = 0
        iter_events = iter(event.events)
        # block until the sequence is done, popping images in the meantime
        while self._mmc.isSequenceRunning():
            if self._mmc.getRemainingImageCount():
                yield self._next_img_payload(next(iter_events))
                count += 1
            else:
                time.sleep(0.001)

        if self._mmc.isBufferOverflowed():  # pragma: no cover
            if self._slackbot is not None:
                self._slackbot.send_message("🚨 Buffer Overflowed! 🚨")
            raise MemoryError("Buffer overflowed")

        while self._mmc.getRemainingImageCount():
            yield self._next_img_payload(next(iter_events))
            count += 1

        if count != n_events:
            logger.warning(
                "Unexpected number of images returned from sequence. "
                "Expected %s, got %s",
                n_events,
                count,
            )

    def _next_img_payload(self, event: MDAEvent) -> PImagePayload:
        """Grab next image from the circular buffer and return it as an ImagePayload."""
        img, meta = self._mmc.popNextImageAndMD()
        tags = self.get_frame_metadata(meta)

        # TEMPORARY SOLUTION
        if self._mmc.mda._wait_until_event(event):  # SLF001
            self._mmc.mda.cancel()
            self._mmc.stopSequenceAcquisition()

        return ImagePayload(img, event, tags)

    def event_iterator(self, events: Iterable[MDAEvent]) -> Iterator[MDAEvent]:
        """Event iterator that merges events for hardware sequencing if possible.

        This wraps `for event in events: ...` inside `MDARunner.run()` and combines
        sequenceable events into an instance of `SequencedEvent` if
        `self.use_hardware_sequencing` is `True`.
        """
        if not self.use_hardware_sequencing:
            yield from events
            return

        seq: list[MDAEvent] = []
        for event in events:
            # if the sequence is empty or the current event can be sequenced with the
            # previous event, add it to the sequence
            if not seq or self.can_sequence_events(seq[-1], event, len(seq)):
                seq.append(event)
            else:
                # otherwise, yield a SequencedEvent if the sequence has accumulated
                # more than one event, otherwise yield the single event
                yield seq[0] if len(seq) == 1 else SequencedEvent.create(seq)
                # add this current event and start a new sequence
                seq = [event]
        # yield any remaining events
        if seq:
            yield seq[0] if len(seq) == 1 else SequencedEvent.create(seq)

    def can_sequence_events(
        self,
        e1: MDAEvent,
        e2: MDAEvent,
        cur_length: int = -1,
        *,
        return_reason: bool = False,
    ) -> bool | tuple[bool, str]:
        """Check whether two [`useq.MDAEvent`][] are sequenceable.

        Micro-manager calls hardware triggering "sequencing".  Two events can be
        sequenced if *all* device properties that are changing between the first and
        second event support sequencing.

        If `cur_length` is provided, it is used to determine if the sequence is
        "full" (i.e. the sequence is already at the maximum length) as determined by
        the `...SequenceMaxLength()` method corresponding to the device property.

        See: <https://micro-manager.org/Hardware-based_Synchronization_in_Micro-Manager>

        Parameters
        ----------
        core : CMMCorePlus
            The core instance.
        e1 : MDAEvent
            The first event.
        e2 : MDAEvent
            The second event.
        cur_length : int
            The current length of the sequence.  Used when checking
            `.get<...>SequenceMaxLength` for a given property. If the current length
            is greater than the max length, the events cannot be sequenced. By default
            -1, which means the current length is not checked.
        return_reason : bool
            If True, return a tuple of (bool, str) where the str is a reason for
            failure. Otherwise just return a bool.

        Returns
        -------
        bool | tuple[bool, str]
            If return_reason is True, return a tuple of a boolean indicating whether the
            events can be sequenced and a string describing the reason for failure if
            the events cannot be sequenced.  Otherwise just return a boolean indicating
            whether the events can be sequenced.

        Examples
        --------
        !!! note

            The results here will depend on the current state of the core and devices.

        ```python
        >>> from useq import MDAEvent
        >>> core = CMMCorePlus.instance()
        >>> core.loadSystemConfiguration()
        >>> can_sequence_events(core, MDAEvent(), MDAEvent())
        (True, "")
        >>> can_sequence_events(core, MDAEvent(x_pos=1), MDAEvent(x_pos=2))
        (False, "Stage 'XY' is not sequenceable")
        >>> can_sequence_events(
        ...     core,
        ...     MDAEvent(channel={'config': 'DAPI'}),
        ...     MDAEvent(channel={'config': 'FITC'})
        ... )
        (False, "'Dichroic-Label' is not sequenceable")
        ```
        """

        def _nope(reason: str) -> tuple[bool, str] | bool:
            return (False, reason) if return_reason else False

        # stimulation event. here we want to have the event with the stimulation at the
        # start of the sequenced event so we can run it before the sequence starts.
        # if e2.sequence is not None:
        if e2.sequence is not None:
            e2_meta = cast(dict, e2.sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
            # {"pymmcore_widgets": {"stimulation": {"pulse_on_frame": {0: 0.5, 10: 1}}}
            if (
                e2_meta.get(STIMULATION)
                and e2.index.get("t") is not None
                and e2.index["t"] in e2_meta[STIMULATION].get("pulse_on_frame", {})
            ):
                return _nope("Cannot sequence events before stimulation.")

        # Action
        if not isinstance(e1.action, (AcquireImage, type(None))) or not isinstance(
            e2.action, (AcquireImage, type(None))
        ):
            return _nope("Cannot sequence non-'AcquireImage' events.")

        # channel
        if e1.channel and e1.channel != e2.channel:
            if not e2.channel or e1.channel.group != e2.channel.group:
                e2_channel_group = getattr(e2.channel, "group", None)
                return _nope(
                    "Cannot sequence across config groups: "
                    f"{e1.channel.group=}, {e2_channel_group=}"
                )
            cfg = self._mmc.getConfigData(e1.channel.group, e1.channel.config)
            for dev, prop, _ in cfg:
                # note: we don't need _ here, so can perhaps speed up with native=True
                if not self._mmc.isPropertySequenceable(dev, prop):
                    return _nope(f"'{dev}-{prop}' is not sequenceable")
                max_len = self._mmc.getPropertySequenceMaxLength(dev, prop)
                if cur_length >= max_len:  # pragma: no cover
                    return _nope(f"'{dev}-{prop}' {max_len=} < {cur_length=}")

        # Z
        if e1.z_pos != e2.z_pos:
            focus_dev = self._mmc.getFocusDevice()
            if not self._mmc.isStageSequenceable(focus_dev):
                return _nope(f"Focus device {focus_dev!r} is not sequenceable")
            max_len = self._mmc.getStageSequenceMaxLength(focus_dev)
            if cur_length >= max_len:  # pragma: no cover
                return _nope(f"Focus device {focus_dev!r} {max_len=} < {cur_length=}")

        # XY
        if e1.x_pos != e2.x_pos or e1.y_pos != e2.y_pos:
            stage = self._mmc.getXYStageDevice()
            if not self._mmc.isXYStageSequenceable(stage):
                return _nope(f"XYStage {stage!r} is not sequenceable")
            max_len = self._mmc.getXYStageSequenceMaxLength(stage)
            if cur_length >= max_len:  # pragma: no cover
                return _nope(f"XYStage {stage!r} {max_len=} < {cur_length=}")

        # camera
        cam_dev = self._mmc.getCameraDevice()
        if not self._mmc.isExposureSequenceable(cam_dev):
            if e1.exposure != e2.exposure:
                return _nope(f"Camera {cam_dev!r} is not exposure-sequenceable")
        elif cur_length >= self._mmc.getExposureSequenceMaxLength(
            cam_dev
        ):  # pragma: no cover
            return _nope(f"Camera {cam_dev!r} {max_len=} < {cur_length=}")

        # time
        # TODO: use better axis keys when they are available
        if (
            e1.index.get("t") != e2.index.get("t")
            and e1.min_start_time != e2.min_start_time
        ):
            pause = (e2.min_start_time or 0) - (e1.min_start_time or 0)
            return _nope(f"Must pause at least {pause} s between events.")

        # misc additional properties
        if e1.properties and e2.properties:
            for dev, prop, value1 in e1.properties:
                for dev2, prop2, value2 in e2.properties:
                    if dev == dev2 and prop == prop2 and value1 != value2:
                        if not self._mmc.isPropertySequenceable(dev, prop):
                            return _nope(f"'{dev}-{prop}' is not sequenceable")
                        if cur_length >= self._mmc.getPropertySequenceMaxLength(
                            dev, prop
                        ):
                            return _nope(f"'{dev}-{prop}' {max_len=} < {cur_length=}")

        return (True, "") if return_reason else True

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Perform any teardown required after the sequence has been executed."""
        # close the current shutter at the end of the sequence
        if self._mmc.getShutterDevice():
            self._mmc.setShutterOpen(False)
