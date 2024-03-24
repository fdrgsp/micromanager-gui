from pathlib import Path
from typing import cast

import napari
import numpy as np
import useq
import zarr
from napari.layers import Image
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEZarrWriter
from pymmcore_plus.mda.handlers._ome_zarr_writer import POS_PREFIX
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QObject
from zarr.storage import TempStore

EXP = "experiment"


class _NapariViewer(QObject, OMEZarrWriter):
    def __init__(
        self,
        parent: QObject | None = None,
        *,
        viewer: napari.Viewer,
        mmcore: CMMCorePlus | None = None,
        store: Path | TempStore | None = None,
    ) -> None:

        self._store = store or TempStore(suffix=".zarr", prefix="pymmcore_zarr_")

        QObject.__init__(self, parent)
        OMEZarrWriter.__init__(self)

        self._mmc = mmcore or CMMCorePlus.instance()

        self._viewer = viewer

        self._layer_name: str = EXP

        self._is_mda_running: bool = False

        # connections
        ev = self._mmc.mda.events
        ev.sequenceStarted.connect(self.sequenceStarted)
        ev.frameReady.connect(self.frameReady)
        ev.sequenceFinished.connect(self.sequenceFinished)

        self.destroyed.connect(self._disconnect)

    def _disconnect(self) -> None:
        """Disconnect the signals."""
        ev = self._mmc.mda.events
        ev.sequenceStarted.disconnect(self.sequenceStarted)
        ev.frameReady.disconnect(self.frameReady)
        ev.sequenceFinished.disconnect(self.sequenceFinished)

    def sequenceStarted(self, sequence: useq.MDASequence) -> None:
        """On sequence started, simply store the sequence."""
        self._is_mda_running = True

        # create a new group for the sequence within the same store by creating a
        # subfolder named with the sequence uid
        if isinstance(self._store, Path):
            store = self._store / f"{sequence.uid}"
        elif isinstance(self._store, TempStore):
            store = Path(self._store.path) / f"{sequence.uid}"
        else:
            raise ValueError("store must be a `Path` or `TempStore`.")
        self._group = zarr.group(store=store)

        # clear the arrays and sizes
        self.position_arrays.clear()
        self.position_sizes.clear()

        # get the filename from the metadata
        self._layer_name = self._get_filename_from_metadata(sequence)

        super().sequenceStarted(sequence)

    def _get_filename_from_metadata(self, sequence: useq.MDASequence) -> str:
        """Get the filename from the sequence metadata."""
        meta = cast(dict, sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        fname = cast(str, meta.get("save_name", EXP))
        # Remove extension
        fname = fname.rsplit(".", maxsplit=1)[0].replace(".ome", "")
        return fname or EXP

    def frameReady(self, image: np.ndarray, event: useq.MDAEvent, meta: dict) -> None:
        super().frameReady(image, event, meta)

        # get position index and key
        p_index = event.index.get("p", 0)
        key = f"{POS_PREFIX}{p_index}"

        if key not in self.position_arrays:
            return

        # get the current layer
        layer = self._get_layer()

        # add new layer or update it if it exists
        if layer is None:
            self._add_new_layer(key)
        else:
            layer.data = self.position_arrays[key]
            self._update_slider(event, p_index)

    def _get_layer(self) -> Image | None:
        """Get the layer if it has the same `uid` as the current sequence."""
        layer = next(
            (
                layer
                for layer in self._viewer.layers
                if layer.metadata.get("uid") == self.current_sequence.uid
            ),
            None,
        )
        return layer

    def _add_new_layer(self, key: str) -> None:
        """Add a new layer to the viewer."""
        data = self.position_arrays[key]
        layer = self._viewer.add_image(data, name=f"{self._layer_name}_{key}")
        layer.scale = self._get_scale(key)
        self._viewer.dims.axis_labels = data.attrs["_ARRAY_DIMENSIONS"]
        layer.metadata = {
            "uid": self.current_sequence.uid,
            "sequence": self.current_sequence,
            "dims": data.attrs["_ARRAY_DIMENSIONS"],
        }

    def _get_scale(self, key: str) -> list[float]:
        """Get the scale for the layer."""
        if self.current_sequence is None:
            raise ValueError("Not a MDA sequence.")

        # add Z to layer scale
        arr = self.position_arrays[key]
        if (pix_size := self._mmc.getPixelSizeUm()) != 0:
            scale = [1.0] * (arr.ndim - 2) + [pix_size] * 2
            if (index := self.current_sequence.used_axes.find("z")) > -1:
                scale[index] = getattr(self.current_sequence.z_plan, "step", 1)
        else:
            # return to default
            scale = [1.0, 1.0]
        return scale

    def _update_slider(self, event: useq.MDAEvent, p_index: int) -> None:
        """Update the slider to the current position."""
        index = tuple(event.index[k] for k in self.position_sizes[p_index])
        cs = list(self._viewer.dims.current_step)
        for a, v in enumerate(index):
            cs[a] = v
        self._viewer.dims.current_step = cs

    def sequenceFinished(self, seq: useq.MDASequence) -> None:
        """On sequence finished, clear the current sequence."""
        self._is_mda_running = False

        super().sequenceFinished(seq)

        self._reset_viewer_dims()

    def _reset_viewer_dims(self) -> None:
        """Reset the viewer dims to the first image."""
        self._viewer.dims.current_step = [0] * len(self._viewer.dims.current_step)
