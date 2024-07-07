"""Simple Image sequence writer for MDASequences.

Writes each frame of an MDA to a directory as individual TIFF files by default,
but can write to other formats if `imageio` is installed or a custom writer is
provided.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from pymmcore_plus.mda.handlers import ImageSequenceWriter
from pymmcore_plus.metadata.serialize import to_builtins

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt
    import useq
    from typing_extensions import TypeAlias  # py310

    ImgWriter: TypeAlias = Callable[[str, npt.NDArray], Any]

FRAME_KEY = "frame"


class TiffSequenceWriter(ImageSequenceWriter):
    """Write each frame of an MDA to a directory as individual image files.

    This writer assumes very little about the sequence, and simply writes each frame
    to a file in the specified directory as a tif file. A subfolder is created for each
    position. It is a good option for ragged or sparse sequences, or where the exact
    number of frames is not known in advance.

    The metadata for each frame is stored in a JSON file in the directory (by default,
    named "_frame_metadata.json").  The metadata is stored as a dict, with the key
    being the index string for the frame (see index_template), and the value being
    the metadata dict for that frame.

    The metadata for the entire MDA sequence is stored in a JSON file in the directory
    (by default, named "_useq_MDASequence.json").

    Parameters
    ----------
    directory: Path | str
        The directory to write the files to.
    extension: str
        The file extension to use.  By default, ".tif".
    prefix: str
        A prefix to add to the file names.  By default, no prefix is added.
    imwrite: Callable[[str, npt.NDArray], Any] | None
        A function to write the image data to disk. The function should take a filename
        and image data as positional arguments. If None, a writer will be selected based
        on the extension. For the default extension `.tif`, this will be
        `tifffile.imwrite` (which must be installed).
    overwrite: bool
        Whether to overwrite the directory if it already exists.  If False, a
        FileExistsError will be raised if the directory already exists.
    include_frame_count: bool
        Whether to include a frame count item in the template (`{frame:05}`). This
        will come after the prefix and before the indices. It is a good way to
        ensure unique keys. by default True
    imwrite_kwargs: dict | None
        Extra keyword arguments to pass to the `imwrite` function.
    """

    def __init__(
        self,
        directory: Path | str,
        extension: str = ".tif",
        prefix: str = "",
        *,
        imwrite: ImgWriter | None = None,
        overwrite: bool = False,
        include_frame_count: bool = True,
        imwrite_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            directory,
            extension,
            prefix,
            imwrite=imwrite,
            overwrite=overwrite,
            include_frame_count=include_frame_count,
            imwrite_kwargs=imwrite_kwargs,
        )

    def frameReady(
        # meta should be FrameMetaV1 but we need to change it in pymmcore-plus
        self,
        frame: np.ndarray,
        event: useq.MDAEvent,
        meta: dict,  # FrameMetaV1
    ) -> None:
        """Write a frame to disk."""
        frame_idx = next(self._counter)
        if self._name_template:
            if FRAME_KEY in self._name_template:
                indices = {**self._first_index, **event.index, FRAME_KEY: frame_idx}
            else:
                indices = {**self._first_index, **event.index}
            filename = self._name_template.format(**indices)
        else:
            # if we don't have a sequence, just use the counter
            filename = f"{self._prefix}_fr{frame_idx:05}.tif"

        pos_name = event.pos_name or f"p{event.index.get('p', 0)}"
        _dir = self._directory / pos_name
        if not _dir.exists():
            _dir.mkdir(parents=True, exist_ok=True)

        # WRITE DATA TO DISK
        self._imwrite(str(_dir / filename), frame, **self._imwrite_kwargs)

        # store metadata
        self._frame_metadata[filename] = to_builtins(meta)
        # write metadata to disk every 10 frames
        if frame_idx % 10 == 0:
            self._frame_meta_file.write_text(json.dumps(self._frame_metadata, indent=2))
