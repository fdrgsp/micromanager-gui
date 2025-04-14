from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymmcore_plus.mda.handlers import OMETiffWriter
from pymmcore_plus.metadata.serialize import to_builtins

if TYPE_CHECKING:
    import numpy as np

EXT = ".ome.tif"
META = "_metadata.json"
SEQ = "_useq_MDASequence.json"


class _OMETiffWriter(OMETiffWriter):
    """MDA handler that writes to a 5D OME-TIFF file.

    Positions will be split into different files.

    Data is memory-mapped to disk using numpy.memmap via tifffile.  Tifffile handles
    the OME-TIFF format.

    Parameters
    ----------
    filename : Path | str
        The filename to write to.  Must end with '.ome.tiff' or '.ome.tif'.
    """

    def __init__(self, filename: Path | str) -> None:
        super().__init__(filename)

        self._folder: Path = Path(self._filename)
        # create a folder to store the OME-TIFF files
        self._folder.mkdir(parents=True, exist_ok=True)

    def new_array(
        self, position_key: str, dtype: np.dtype, sizes: dict[str, int]
    ) -> np.memmap:
        """Create a new tifffile file and memmap for this position.

        In this version, we save ome-tiff files in a dedicated folder, and each position
        will be saved in a separate file within that folder. The position name, if
        opresent, will be used to name the file.
        """
        from tifffile import imwrite, memmap

        dims, shape = zip(*sizes.items())

        metadata: dict[str, Any] = self._sequence_metadata()
        metadata["axes"] = "".join(dims).upper()

        # add the position key to the filename if there are multiple positions
        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            folder_name = self._folder.name.replace(EXT, "")
            pos_name = f"_{self._get_current_pos_name(position_key)}{EXT}"
        else:
            folder_name = self._folder.name
            pos_name = ""

        fname = self._folder / f"{folder_name}{pos_name}"

        # write empty file to disk
        imwrite(
            fname,
            shape=shape,
            dtype=dtype,
            metadata=metadata,
            imagej=not self._is_ome,
            ome=self._is_ome,
        )

        # memory-mapped NumPy array of image data stored in TIFF file.
        mmap = memmap(fname, dtype=dtype)
        # This line is important, as tifffile.memmap appears to lose singleton dims
        mmap.shape = shape

        return mmap  # type: ignore

    def _get_current_pos_name(self, position_key: str) -> str:
        """Get the position name from the position_key if any."""
        if self.current_sequence is None:
            return position_key

        pos_n = int(position_key[1:])
        current_pos = self.current_sequence.stage_positions[pos_n]
        return current_pos.name or position_key

    def finalize_metadata(self) -> None:
        """Write the metadata per position to a json file.

        This method is called when the 'sequenceFinished' signal is received.
        """
        if not self._is_ome:
            return

        # store all the position metadata in a single file. Needed because we overwrite
        # the frame_metadatas keys with the position name
        meta: dict[str, Any] = {}
        for position_key in list(self.frame_metadatas.keys()):
            pos_name = self._get_current_pos_name(position_key)
            meta[pos_name] = self.frame_metadatas[position_key]

        # save metadata
        with open(self._folder / META, "w") as f:
            formatted = json.dumps(to_builtins(meta), indent=2)
            f.write(formatted)

        # save sequence
        with open(self._folder / SEQ, "w") as f:
            if self.current_sequence is not None:
                f.write(
                    self.current_sequence.model_dump_json(exclude_unset=True, indent=4)
                )
