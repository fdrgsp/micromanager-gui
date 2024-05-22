from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import TensorStoreHandler
from pymmcore_widgets._stack_viewer_v2 import MDAViewer
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QObject, Qt

from ._widgets._preview import Preview

DIALOG = Qt.WindowType.Dialog
VIEWER_TEMP_DIR = None

if TYPE_CHECKING:
    import useq

    from ._main_window import MicroManagerGUI


class CoreViewersLink(QObject):
    def __init__(self, parent: MicroManagerGUI, *, mmcore: CMMCorePlus | None = None):
        super().__init__(parent)

        self._main_window = parent
        self._mmc = mmcore or CMMCorePlus.instance()

        # keep track of the current mda viewer
        self._current_viewer: MDAViewer | None = None

        self._mda_running: bool = False

        # show the preview tab when the snap or live button is clicked
        self._main_window._snap_live_toolbar._snap.clicked.connect(self._show_preview)
        self._main_window._snap_live_toolbar._live.clicked.connect(self._show_preview)
        self._mmc.events.continuousSequenceAcquisitionStarted.connect(
            self._show_preview
        )
        self._mmc.events.imageSnapped.connect(self._show_preview)

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)
        self._mmc.mda.events.sequencePauseToggled.connect(self._enable_gui)

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        """Show the MDAViewer when the MDA sequence starts."""
        self._mda_running = True

        # disable the menu bar
        self._main_window._menu_bar._enable(False)

        # pause until the viewer is ready
        self._mmc.mda.toggle_pause()
        # setup the viewer
        self._setup_viewer(sequence)
        # resume the sequence
        self._mmc.mda.toggle_pause()

    def _setup_viewer(self, sequence: useq.MDASequence) -> None:
        datastore = TensorStoreHandler.in_tmpdir(prefix="mm_gui_", dir=VIEWER_TEMP_DIR)
        self._current_viewer = MDAViewer(parent=self._main_window, datastore=datastore)

        # rename the viewer if there is a save_name' in the metadata or add a digit
        save_meta = cast(dict, sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        viewer_name = self._get_viewer_name(save_meta.get("save_name"))
        self._main_window._viewer_tab.addTab(self._current_viewer, viewer_name)
        self._main_window._viewer_tab.setCurrentWidget(self._current_viewer)

        # call it manually insted in _connect_viewer because this signal has been
        # emitted already
        self._current_viewer.data.sequenceStarted(sequence)

        # disable the LUT drop down and the mono/composite button (temporary)
        self._enable_gui(False)

        # connect the signals
        self._connect_viewer(self._current_viewer)

    def _get_viewer_name(self, viewer_name: str | None) -> str:
        """Get the viewer name from the metadata.

        If viewer_name is None, get the highest index for the viewer name. Otherwise,
        return the viewer name.
        """
        if viewer_name:
            return viewer_name

        # loop through the tabs and get the highest index for the viewer name
        index = 0
        for v in range(self._main_window._viewer_tab.count()):
            tab_name = self._main_window._viewer_tab.tabText(v)
            if tab_name.startswith("MDA Viewer"):
                idx = tab_name.replace("MDA Viewer ", "")
                if idx.isdigit():
                    index = max(index, int(idx))
        return f"MDA Viewer {index + 1}"

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        """Hide the MDAViewer when the MDA sequence finishes."""
        self._main_window._menu_bar._enable(True)

        self._mda_running = False

        if self._current_viewer is None:
            return

        # enable the LUT drop down and the mono/composite button (temporary)
        self._enable_gui(True)

        self._disconnect_viewer(self._current_viewer)

        self._current_viewer = None

    def _connect_viewer(self, viewer: MDAViewer) -> None:
        self._mmc.mda.events.sequenceFinished.connect(viewer.data.sequenceFinished)
        self._mmc.mda.events.frameReady.connect(viewer.data.frameReady)

    def _disconnect_viewer(self, viewer: MDAViewer) -> None:
        """Disconnect the signals."""
        self._mmc.mda.events.sequenceFinished.disconnect(viewer.data.sequenceFinished)
        self._mmc.mda.events.frameReady.disconnect(viewer.data.frameReady)

    def _enable_gui(self, state: bool) -> None:
        """Pause the viewer when the MDA sequence is paused."""
        self._main_window._menu_bar._enable(state)
        if self._current_viewer is None:
            return

        # self._current_viewer._lut_drop.setEnabled(state)
        self._current_viewer._channel_mode_btn.setEnabled(state)

    def _show_preview(self) -> None:
        """Show the preview tab."""
        if self._mda_running:
            return

        preview_tab = self._main_window._viewer_tab.widget(0)
        if isinstance(preview_tab, Preview):
            return

        _preview = Preview(self._main_window, mmcore=self._mmc)
        self._main_window._viewer_tab.insertTab(0, _preview, "Preview")

        if self._mmc.isSequenceRunning():
            _preview._image_preview._on_streaming_start()
