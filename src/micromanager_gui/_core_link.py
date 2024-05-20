from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets._stack_viewer_v2 import MDAViewer
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QObject, Qt

DIALOG = Qt.WindowType.Dialog

if TYPE_CHECKING:
    import useq

    from ._main_window import MicroManagerGUI


class MDAViewersLink(QObject):
    def __init__(self, parent: MicroManagerGUI, *, mmcore: CMMCorePlus | None = None):
        super().__init__(parent)

        self._main_window = parent
        self._mmc = mmcore or CMMCorePlus.instance()

        self._current_viewer: MDAViewer | None = None

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        """Show the MDAViewer when the MDA sequence starts."""
        self._is_mda_running = True

        # pause until the viewer is ready
        self._mmc.mda.toggle_pause()
        # setup the viewer
        self._setup_viewer(sequence)
        # resume the sequence
        self._mmc.mda.toggle_pause()

    def _setup_viewer(self, sequence: useq.MDASequence) -> None:
        self._current_viewer = MDAViewer(parent=self._main_window)

        # rename the viewer if there is a save_name' in the metadata or add a digit
        save_meta = cast(dict, sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        save_name = save_meta.get("save_name")
        number_of_tabs = self._main_window._viewer_tab.count() - 1
        save_name = (
            save_name if save_name is not None else f"MDA Viewer {number_of_tabs + 1}"
        )
        self._main_window._viewer_tab.addTab(self._current_viewer, save_name)
        self._main_window._viewer_tab.setCurrentWidget(self._current_viewer)

        # call it manually insted in _connect_viewer because this signal has been
        # emitted already
        self._current_viewer.data.sequenceStarted(sequence)

        # connect the signals
        self._connect_viewer(self._current_viewer)

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        """Hide the MDAViewer when the MDA sequence finishes."""
        self._is_mda_running = False
        if self._current_viewer is None:
            return
        self._disconnect_viewer(self._current_viewer)

    def _connect_viewer(self, viewer: MDAViewer) -> None:
        self._mmc.mda.events.sequenceFinished.connect(viewer.data.sequenceFinished)
        self._mmc.mda.events.frameReady.connect(viewer.data.frameReady)

    def _disconnect_viewer(self, viewer: MDAViewer) -> None:
        """Disconnect the signals."""
        self._mmc.mda.events.sequenceFinished.disconnect(viewer.data.sequenceFinished)
        self._mmc.mda.events.frameReady.disconnect(viewer.data.frameReady)
