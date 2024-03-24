from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import napari
import napari.layers
import numpy as np
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QObject, Qt, QTimerEvent

from ._mda_viewers._napari_viewer import _NapariViewer

if TYPE_CHECKING:
    import napari.viewer
    import numpy as np
    from napari.layers import Image
    from pymmcore_plus.core.events._protocol import PSignalInstance

DIALOG = Qt.WindowType.Dialog

if TYPE_CHECKING:

    from ._main_window import MicroManagerGUI


class _CoreLinkWithNapari(QObject):
    def __init__(
        self,
        parent: MicroManagerGUI,
        *,
        mmcore: CMMCorePlus | None = None,
        viewer: napari.Viewer,
    ):
        super().__init__(parent)

        self._live_timer_id: int | None = None

        self._mmc = mmcore or CMMCorePlus.instance()

        self.viewer = viewer

        self._mda_viewer = _NapariViewer(self, viewer=self.viewer, mmcore=self._mmc)

        # Add all core connections to this list.  This makes it easy to disconnect
        # from core when this widget is closed.
        self._connections: list[tuple[PSignalInstance, Callable]] = [
            (self._mmc.events.imageSnapped, self._image_snapped),
            (self._mmc.events.imageSnapped, self._stop_live),
            (self._mmc.events.continuousSequenceAcquisitionStarted, self._start_live),
            (self._mmc.events.sequenceAcquisitionStopped, self._stop_live),
            (self._mmc.events.exposureChanged, self._restart_live),
        ]
        for signal, slot in self._connections:
            signal.connect(slot)

    def _disconnect(self) -> None:
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._mda_viewer._disconnect()

    def timerEvent(self, event: QTimerEvent | None) -> None:
        self._update_viewer()

    def _image_snapped(self) -> None:
        # If we are in the middle of an MDA, don't update the preview viewer.
        if not self._mda_viewer._mda_running:
            self._update_viewer(self._mmc.getImage())

    def _start_live(self) -> None:
        interval = int(self._mmc.getExposure())
        self._live_timer_id = self.startTimer(interval, Qt.TimerType.PreciseTimer)

    def _stop_live(self) -> None:
        if self._live_timer_id is not None:
            self.killTimer(self._live_timer_id)
            self._live_timer_id = None

    def _restart_live(self, camera: str, exposure: float) -> None:
        if self._live_timer_id:
            self._mmc.stopSequenceAcquisition()
            self._mmc.startContinuousSequenceAcquisition()

    def _update_viewer(self, data: np.ndarray | None = None) -> None:
        """Update viewer with the latest image from the circular buffer."""
        if data is None:
            try:
                data = self._mmc.getLastImage()
            except (RuntimeError, IndexError):
                # circular buffer empty
                return
        try:
            preview_layer = cast("Image", self.viewer.layers["preview"])
            preview_layer.data = data
        except KeyError:
            preview_layer = self.viewer.add_image(data, name="preview")

        preview_layer.metadata["mode"] = "preview"

        if (pix_size := self._mmc.getPixelSizeUm()) != 0:
            preview_layer.scale = (pix_size, pix_size)
        else:
            # return to default
            preview_layer.scale = [1.0, 1.0]

        if self._live_timer_id is None:
            self.viewer.reset_view()
