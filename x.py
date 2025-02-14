from typing import Iterable

import useq
from pymmcore_plus import CMMCorePlus
from rich import print


class CustomMDASequence:
    """A sequence of events to be executed by the MDAEngine."""

    def __init__(
        self,
        sequence: useq.MDASequence | None = None,
        events: list[useq.MDAEvent] | None = None,
    ):
        self._sequence = sequence
        self.events = events or []

    @property
    def sequence(self) -> useq.MDASequence | None:
        """Return the sequence."""
        return self._sequence

    @sequence.setter
    def sequence(self, sequence: useq.MDASequence | None) -> None:
        """Set the sequence."""
        self._sequence = sequence

    def clear_events(self) -> None:
        """Clear the events."""
        self.events.clear()

    def add_events(self, event: useq.MDAEvent | list[useq.MDAEvent]) -> None:
        """Add an event to the sequence."""
        if isinstance(event, list):
            self.events.extend(event)
        else:
            self.events.append(event)

    def __iter__(self) -> Iterable[useq.MDAEvent]:
        """Iterate over the events in the sequence."""
        return iter(self.events)


mmc = CMMCorePlus()
mmc.loadSystemConfiguration()
mda = CustomMDASequence()

sequence = useq.MDASequence(
    channels=[useq.Channel(config="DAPI", exposure=100.0) for _ in range(3)]
)
events = [
    useq.MDAEvent(channel={"config": "DAPI", "exposure": 100.0}),
    useq.MDAEvent(channel={"config": "DAPI", "exposure": 100.0}),
    useq.MDAEvent(channel={"config": "DAPI", "exposure": 100.0}),
]

mda.sequence = sequence
mda.add_events(events)

print(mda.sequence)

for event in mda:
    print(event)

mmc.run_mda(mda)
