from typing import Iterable

import useq
from pymmcore_plus import CMMCorePlus
from rich import print


class CustomMDASequence:
    """A sequence of events to be executed by the MDAEngine."""

    def __init__(self):
        self._sequence: useq.MDASequence | None = None
        self.events = [
            useq.MDAEvent(channel={"config": "DAPI", "exposure": 100.0}),
            useq.MDAEvent(channel={"config": "DAPI", "exposure": 100.0}),
            useq.MDAEvent(channel={"config": "DAPI", "exposure": 100.0}),
        ]

    @property
    def sequence(self) -> useq.MDASequence | None:
        """Return the sequence."""
        return self._sequence

    @sequence.setter
    def sequence(self, sequence: useq.MDASequence | None) -> None:
        """Set the sequence."""
        self._sequence = sequence

    def __iter__(self) -> Iterable[useq.MDAEvent]:
        """Iterate over the events in the sequence."""
        return iter(self.events)


mmc = CMMCorePlus()
mmc.loadSystemConfiguration()
mda = CustomMDASequence()
print(mda.sequence)

# for event in mda:
#     print(event)

# mmc.run_mda(mda)
