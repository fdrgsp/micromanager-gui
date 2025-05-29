from __future__ import annotations

import time
from collections import defaultdict
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import useq
from pyfirmata2 import Arduino
from pyfirmata2.pyfirmata2 import Pin
from pymmcore_plus._logger import logger
from pymmcore_plus.core._constants import Keyword
from pymmcore_plus.core._sequencing import EventCombiner, SequencedEvent
from pymmcore_plus.mda import MDAEngine
from useq import AcquireImage, CustomAction, HardwareAutofocus, MDAEvent, MDASequence

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda._protocol import PImagePayload
    from pymmcore_plus.metadata import SummaryMetaV1

    from micromanager_gui._slackbot._mm_slackbot import MMSlackBot

WARNING_EMOJI = ":warning:"
ALARM_EMOJI = ":rotating_light:"


def iter_custom_sequenced_events(
    core: CMMCorePlus, events: Iterable[MDAEvent]
) -> Iterator[MDAEvent | SequencedEvent]:
    """Iterate over a sequence of MDAEvents, yielding SequencedEvents when possible.

    Parameters
    ----------
    core : CMMCorePlus
        The core object to use for determining sequenceable properties.
    events : Iterable[MDAEvent]
        The events to iterate over.

    Returns
    -------
    Iterator[MDAEvent | SequencedEvent]
        A new iterator that will combine multiple MDAEvents into a single SequencedEvent
        when possible, based on the sequenceable properties of the core object.
        Note that `SequencedEvent` itself is a subclass of `MDAEvent`, but it's up to
        the engine to check `isinstance(event, SequencedEvent)` in order to handle
        SequencedEvents differently.
    """
    combiner = CustomEventCombiner(core)
    for e in events:
        if (flushed := combiner.feed_event(e)) is not None:
            yield flushed

    if (leftover := combiner.flush()) is not None:
        yield leftover


class CustomEventCombiner(EventCombiner):
    """Custom event combiner to handle custom actions.

    This combiner treats the `CustomAction` with the name
    "arduino_stimulation" as a special case, allowing it to be sequenced.
    """

    def __init__(self, core: CMMCorePlus) -> None:
        super().__init__(core)

    def can_extend(self, event: MDAEvent) -> bool:
        """Return True if the new event can be added to the current batch."""
        # cannot add pre-existing SequencedEvents to the sequence
        if not self.event_batch:
            return True

        e0 = self.event_batch[0]

        # cannot sequence on top of SequencedEvents
        if isinstance(e0, SequencedEvent) or isinstance(event, SequencedEvent):
            return False

        # Check if actions are sequenceable
        def is_sequenceable_action(action: useq.Action) -> bool:
            # Allow AcquireImage, None, or arduino_stimulation CustomAction
            if isinstance(action, (AcquireImage, type(None))):
                return True
            return (
                isinstance(action, CustomAction)
                and action.name == "arduino_stimulation"
            )

        if not is_sequenceable_action(e0.action) or not is_sequenceable_action(
            event.action
        ):
            return False

        # Special logic for stimulation events:
        # Every stimulation event should start a NEW sequence
        # This ensures each stimulation gets its own SequencedEvent
        if (
            isinstance(event.action, CustomAction)
            and event.action.name == "arduino_stimulation"
        ):
            # Never extend when encountering a stimulation event
            # This forces a new sequence to start
            return False

        new_chunk_len = len(self.event_batch) + 1

        # NOTE: these should be ordered from "fastest to check / most likely to fail",
        # to "slowest to check / most likely to pass"

        # If it's a new timepoint, and they have a different start time
        # we don't (yet) support sequencing.
        if (
            event.index.get("t") != e0.index.get("t")
            and event.min_start_time != e0.min_start_time
        ):
            return False

        # Exposure
        if event.exposure != e0.exposure:
            if new_chunk_len > self.max_lengths[Keyword.CoreCamera]:
                return False
            self.attribute_changes[Keyword.CoreCamera] = True

        # XY
        if event.x_pos != e0.x_pos or event.y_pos != e0.y_pos:
            if new_chunk_len > self.max_lengths[Keyword.CoreXYStage]:
                return False
            self.attribute_changes[Keyword.CoreXYStage] = True

        # Z
        if event.z_pos != e0.z_pos:
            if new_chunk_len > self.max_lengths[Keyword.CoreFocus]:
                return False
            self.attribute_changes[Keyword.CoreFocus] = True

        # SLM
        if event.slm_image != e0.slm_image:
            if new_chunk_len > self.max_lengths[Keyword.CoreSLM]:
                return False
            self.attribute_changes[Keyword.CoreSLM] = True

        # properties
        event_props = self._event_properties(event)
        all_props = event_props.keys() | self.first_event_props.keys()
        for dev_prop in all_props:
            new_val = event_props.get(dev_prop)
            old_val = self.first_event_props.get(dev_prop)
            if new_val != old_val:
                # if the property has changed, (or is missing in one dict)
                if new_chunk_len > self._get_property_max_length(dev_prop):
                    return False
                self.attribute_changes[dev_prop] = True

        return True

    def _create_sequenced_event(self) -> MDAEvent | SequencedEvent:
        """Convert self.event_batch into a SequencedEvent.

        If the batch contains only a single event, that event is returned directly.

        Overriding because we add to the metadata a "stimulation" info to perform
        the LED stimulation event in the engine.
        """
        if not self.event_batch:
            raise RuntimeError("Cannot flush an empty chunk")

        first_event = self.event_batch[0]

        if (num_events := len(self.event_batch)) == 1:
            return first_event

        exposures: list[float | None] = []
        x_positions: list[float | None] = []
        y_positions: list[float | None] = []
        z_positions: list[float | None] = []
        slm_images: list[Any] = []
        property_sequences: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
        static_props: list[tuple[str, str, Any]] = []

        # Single pass
        for e in self.event_batch:
            exposures.append(e.exposure)
            x_positions.append(e.x_pos)
            y_positions.append(e.y_pos)
            z_positions.append(e.z_pos)
            slm_images.append(e.slm_image)
            for dev_prop, val in self._event_properties(e).items():
                property_sequences[dev_prop].append(val)

        # remove any property sequences that are static
        for key, prop_seq in list(property_sequences.items()):
            if not self.attribute_changes.get(key):
                static_props.append((*key, prop_seq[0]))
                property_sequences.pop(key)
            elif len(prop_seq) != num_events:
                raise RuntimeError(
                    "Property sequence length mismatch. "
                    "Please report this with an example."
                )

        exp_changed = self.attribute_changes.get(Keyword.CoreCamera)
        xy_changed = self.attribute_changes.get(Keyword.CoreXYStage)
        z_changed = self.attribute_changes.get(Keyword.CoreFocus)
        slm_changed = self.attribute_changes.get(Keyword.CoreSLM)

        exp_seq = tuple(exposures) if exp_changed else ()
        x_seq = tuple(x_positions) if xy_changed else ()
        y_seq = tuple(y_positions) if xy_changed else ()
        z_seq = tuple(z_positions) if z_changed else ()
        slm_seq = tuple(slm_images) if slm_changed else ()

        # Extract stimulation information for metadata
        stimulation_info: dict[str, Any] | None = None
        for e in self.event_batch:
            if (
                isinstance(e.action, CustomAction)
                and e.action.name == "arduino_stimulation"
                and e.action.data
            ):
                stimulation_info = e.action.data
                break

        # Create metadata with stimulation info if present
        metadata = {}
        if stimulation_info:
            metadata["stimulation"] = stimulation_info

        return SequencedEvent(
            events=tuple(self.event_batch),
            exposure_sequence=exp_seq,
            x_sequence=x_seq,
            y_sequence=y_seq,
            z_sequence=z_seq,
            slm_sequence=slm_seq,
            property_sequences=property_sequences,
            properties=static_props,
            metadata=metadata,
            # all other "standard" MDAEvent fields are derived from the first event
            # the engine will use these values if the corresponding sequence is empty
            x_pos=first_event.x_pos,
            y_pos=first_event.y_pos,
            z_pos=first_event.z_pos,
            exposure=first_event.exposure,
            channel=first_event.channel,
        )


class ArduinoEngine(MDAEngine):
    def __init__(
        self,
        mmc: CMMCorePlus,
        use_hardware_sequencing: bool = True,
        arduino_board: Arduino | None = None,
        arduino_led_pin: Pin | None = None,
        slackbot: MMSlackBot | None = None,
    ) -> None:
        super().__init__(mmc, use_hardware_sequencing)

        self._slackbot = slackbot

        # for LED stimulation
        self._arduino_board = arduino_board
        self._arduino_led_pin = arduino_led_pin

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
            self._arduino_led_pin = cast(Pin, self._arduino_led_pin)
            self._arduino_led_pin.write(0.0)
        return super().setup_sequence(sequence)

    def event_iterator(self, events: Iterable[MDAEvent]) -> Iterator[MDAEvent]:
        """Event iterator that merges events for hardware sequencing if possible.

        This wraps `for event in events: ...` inside `MDARunner.run()` and combines
        sequenceable events into an instance of `SequencedEvent` if
        `self.use_hardware_sequencing` is `True`.
        """
        if not self.use_hardware_sequencing:
            yield from events
            return

        yield from iter_custom_sequenced_events(self._mmc, events)

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

        # don't try to execute any other action types. Mostly, this is just
        # CustomAction, which is a user-defined action that the engine doesn't know how
        # to handle.  But may include other actions in the future, and this ensures
        # backwards compatibility.
        if not isinstance(action, (AcquireImage, type(None))):
            return

        # if the autofocus was engaged at the start of the sequence AND autofocus action
        # did not fail, re-engage it. NOTE: we need to do that AFTER the runner calls
        # `setup_event`, so we can't do it inside the exec_event autofocus action above.
        if self._af_was_engaged and self._af_succeeded:
            # if self._arduino_board is None and...
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
        self._arduino_board = cast(Arduino, self._arduino_board)
        self._arduino_led_pin = cast(Pin, self._arduino_led_pin)
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
        # NOTE: overriding because of the "Buffer Overflowed" slakbot message and for
        # the stimulation event handling.

        n_events = len(event.events)

        t0 = event.metadata.get("runner_t0") or time.perf_counter()
        event_t0_ms = (time.perf_counter() - t0) * 1000

        if event.slm_image is not None:
            self._exec_event_slm_image(event.slm_image)

        # check for stimulation event in the metadata and execute it if present
        if (stim_meta := event.metadata.get("stimulation")) is not None:
            self._exec_led_stimulation(stim_meta)

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
