import json
from os import PathLike
from typing import (
    ContextManager,
    Literal,
    MutableMapping,
    Protocol,
    Sequence,
    TypedDict,
)

import numpy as np
import zarr
from fsspec import FSMap
from numcodecs.abc import Codec
from pymmcore_plus.mda.handlers import OMEZarrWriter


class ZarrSynchronizer(Protocol):
    def __getitem__(self, key: str) -> ContextManager: ...


class ArrayCreationKwargs(TypedDict, total=False):
    compressor: str | Codec
    fill_value: int | None
    order: Literal["C", "F"]
    synchronizer: ZarrSynchronizer | None
    overwrite: bool
    filters: Sequence[Codec] | None
    cache_attrs: bool
    read_only: bool
    object_codec: Codec | None
    dimension_separator: Literal["/", "."] | None
    write_empty_chunks: bool


class _OMEZarrWriter(OMEZarrWriter):
    """MDA handler that writes to a zarr file following the ome-ngff spec.

    This implements v0.4
    https://ngff.openmicroscopy.org/0.4/index.html

    It also aims to be compatible with the xarray Zarr spec:
    https://docs.xarray.dev/en/latest/internals/zarr-encoding-spec.html

    Note: this does *not* currently calculate any additional pyramid levels.
    But it would be easy to do so after acquisition.
    Chunk size is currently 1 XY plane.

    Zarr directory structure will be:

    ```
    root.zarr/
    ├── .zgroup                 # group metadata
    ├── .zattrs                 # contains ome-multiscales metadata
    │
    ├── p0                      # each position is a separate <=5D array
    │   ├── .zarray
    │   ├── .zattrs
    │   └── t                   # nested directories for each dimension
    │       └── c               # (only collected dimensions will be present)
    │           └── z
    │               └── y
    │                   └── x   # chunks will be each XY plane
    ├── ...
    ├── p<n>
    │   ├── .zarray
    │   ├── .zattrs
    │   └── t...
    ```

    Parameters
    ----------
    store: MutableMapping | str | None
        Zarr store or path to directory in file system to write to.
        Semantics are the same as for `zarr.group`: If a string, it is interpreted as a
        path to a directory. If None, an in-memory store is used.  May also be any
        mutable mapping or instance of `zarr.storage.BaseStore`.
    overwrite : bool
        If True, delete any pre-existing data in `store` at `path` before
        creating the group. If False, raise an error if there is already data
        in `store` at `path`. by default False.
    synchronizer : ZarrSynchronizer | None, optional
        Array synchronizer passed to `zarr.group`.
    zarr_version : {2, 3, None}, optional
        Zarr version passed to `zarr.group`.
    array_kwargs : dict, optional
        Keyword arguments passed to `zarr.group.create` when creating the arrays.
        This may be used to set the zarr `compressor`, `fill_value`, `synchronizer`,
        etc... Default is `{'dimension_separator': '/'}`.
    minify_attrs_metadata : bool, optional
        If True, zattrs metadata will be read from disk, minified, and written
        back to disk at the end of a successful acquisition (to save space). Default is
        False.
    """

    def __init__(
        self,
        store: MutableMapping | str | PathLike | FSMap | None = None,
        *,
        overwrite: bool = False,
        synchronizer: ZarrSynchronizer | None = None,
        zarr_version: None | Literal[2] | Literal[3] = None,
        array_kwargs: ArrayCreationKwargs | None = None,
        minify_attrs_metadata: bool = False,
    ) -> None:
        super().__init__(
            store,
            overwrite=overwrite,
            synchronizer=synchronizer,
            zarr_version=zarr_version,
            array_kwargs=array_kwargs,
            minify_attrs_metadata=minify_attrs_metadata,
        )

    def new_array(self, key: str, dtype: np.dtype, sizes: dict[str, int]) -> zarr.Array:
        """Create a new array in the group, under `key`."""
        dims, shape = zip(*sizes.items())
        # try to get position name from sequence metadata
        name = self._get_current_pos_name(key)
        ary: zarr.Array = self._group.create(
            name,
            shape=shape,
            chunks=(1,) * len(shape[:-2]) + shape[-2:],  # single XY plane chunks
            dtype=dtype,
            **self._array_kwargs,
        )

        # add minimal OME-NGFF metadata
        scales = self._group.attrs.get("multiscales", [])
        scales.append(self._multiscales_item(ary.path, ary.path, dims))
        self._group.attrs["multiscales"] = scales
        ary.attrs["_ARRAY_DIMENSIONS"] = dims
        if self.current_sequence is not None:
            ary.attrs["useq_MDASequence"] = json.loads(
                self.current_sequence.model_dump_json(exclude_unset=True)
            )
        return ary

    def _get_current_pos_name(self, position_key: str) -> str:
        """Get the position name from the position_key if any."""
        if self.current_sequence is None:
            return position_key

        pos_n = int(position_key[1:])
        current_pos = self.current_sequence.stage_positions[pos_n]
        return current_pos.name or position_key
