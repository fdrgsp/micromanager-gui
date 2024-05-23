import json
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorstore as ts
import useq


class TensorZarrReader:
    """Read data from a tensorstore zarr file.

    Parameters
    ----------
    path : str | Path
        The path to the tensorstore zarr file.

    Attributes
    ----------
    path : Path
        The path to the tensorstore zarr file.
    store : ts.TensorStore
        The tensorstore.
    metadata : dict
        The metadata from the acquisition. They are stored in the `.zattrs` file and
        should contain two keys: `useq_MDASequence` and `useq_MDASequence`.
    sequence : useq.MDASequence
        The acquired useq.MDASequence. It is loaded from the metadata using the
        `useq.MDASequence` key.

    Usage
    -----
    reader = TensorZarrReader("path/to/file")
    # to get the numpy array for a specific axis, for example, the first time point for
    # the first position and the first z-slice:
    data = reader.isel({"p": 0, "t": 1, "z": 0})
    """

    def __init__(self, path: str | Path):
        self._path = path

        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": self._path},
        }

        self._store = ts.open(spec)

        self._metadata: dict = {}
        if metadata_json := self.store.kvstore.read(".zattrs").result().value:
            self._metadata = json.loads(metadata_json)

        self._axis_max: dict[str, int] = {}

    @property
    def path(self) -> Path:
        """Return the path."""
        return Path(self._path)

    @property
    def store(self) -> ts.TensorStore:
        """Return the tensorstore."""
        return self._store.result()

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def sequence(self) -> useq.MDASequence:
        seq = self._metadata.get("useq_MDASequence")
        return useq.MDASequence(**json.loads(seq)) if seq is not None else None

    def _get_axis_index(self, indexers: Mapping[str, int]) -> tuple[object, ...]:
        """Return a tuple to index the data for the given axis."""
        if self.sequence is None:
            raise ValueError("No 'useq.MDASequence' found in the metadata!")

        axis_order = self.sequence.axis_order

        # if any of the indexers are not in the axis order, raise an error
        if not set(indexers.keys()).issubset(set(axis_order)):
            raise ValueError("Invalid axis in indexers!")

        # get the correct index for the axis
        # e.g. (slice(None), 1, slice(None), slice(None))
        return tuple(
            indexers[axis] if axis in indexers else slice(None) for axis in axis_order
        )

    def isel(self, indexers: Mapping[str, int]) -> np.ndarray:
        """Select data from the array."""
        index = self._get_axis_index(indexers)
        return self.store[index].read().result().squeeze()
