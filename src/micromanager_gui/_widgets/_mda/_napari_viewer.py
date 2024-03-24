from pathlib import Path
from typing import TYPE_CHECKING, cast

import napari
import numpy as np
import useq
import zarr
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import OMEZarrWriter
from pymmcore_plus.mda.handlers._ome_zarr_writer import POS_PREFIX
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QObject
from zarr.storage import TempStore

if TYPE_CHECKING:
    from napari.layers import Image


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

        self._count = -1

        self._mmc = mmcore or CMMCorePlus.instance()

        self._viewer = viewer

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

    def sequenceStarted(self, seq: useq.MDASequence) -> None:
        """On sequence started, simply store the sequence."""
        self._is_mda_running = True

        if isinstance(self._store, Path):
            store = self._store / f"{seq.uid}"
        elif isinstance(self._store, TempStore):
            store = Path(self._store.path) / f"{seq.uid}"
        else:
            raise ValueError("store must be a `Path` or `TempStore`.")

        self._group = zarr.group(store=store)

        self.position_arrays.clear()
        self.position_sizes.clear()

        self._count += 1

        super().sequenceStarted(seq)

    def frameReady(self, image: np.ndarray, event: useq.MDAEvent, meta: dict) -> None:
        super().frameReady(image, event, meta)

        # get position index and key
        p_index = event.index.get("p", 0)
        key = f"{POS_PREFIX}{p_index}"

        if key not in self.position_arrays:
            return

        # set the layer name (get it from the metadata if available))
        meta = self.current_sequence.metadata.get(PYMMCW_METADATA_KEY)
        layer_name = meta.get("save_name", f"experiment_{self._count:03d}")

        # if the layer does not exist, create it
        if layer_name not in self._viewer.layers:
            data = self.position_arrays[key]
            layer = self._viewer.add_image(data, name=layer_name, blending="additive")
            layer.scale = self._get_scale(key)
            self._viewer.dims.axis_labels = data.attrs["_ARRAY_DIMENSIONS"]
            layer.metadata["sequence"] = self.current_sequence
        # if the layer exists, update the data
        else:
            layer = cast("Image", self._viewer.layers[layer_name])
            layer.data = self.position_arrays[key]
            index = tuple(event.index[k] for k in self.position_sizes[p_index])
            self._update_slider(index)

    def _update_slider(self, index: tuple[int, ...]) -> None:
        """Update the slider to the current position."""
        cs = list(self._viewer.dims.current_step)
        for a, v in enumerate(index):
            cs[a] = v
        self._viewer.dims.current_step = cs

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

    def sequenceFinished(self, seq: useq.MDASequence) -> None:
        """On sequence finished, clear the current sequence."""
        self._is_mda_running = False

        super().sequenceFinished(seq)

        self._reset_viewer_dims()

    def _reset_viewer_dims(self) -> None:
        """Reset the viewer dims to the first image."""
        self._viewer.dims.current_step = [0] * len(self._viewer.dims.current_step)
