from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, cast

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets._stack_viewer_v2 import MDAViewer
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QTabBar, QTabWidget

from ._widgets._preview import Preview

if TYPE_CHECKING:
    import numpy as np
    import useq

    from ._main_window import MicroManagerGUI
    from ._slackbot._mm_slackbot import MMSlackBot
    from ._widgets._mda_widget import _MDAWidget

DIALOG = Qt.WindowType.Dialog
VIEWER_TEMP_DIR = None
NO_R_BTN = (0, QTabBar.ButtonPosition.RightSide, None)
NO_L_BTN = (0, QTabBar.ButtonPosition.LeftSide, None)


class CoreViewersLink(QObject):
    def __init__(
        self,
        parent: MicroManagerGUI,
        *,
        mmcore: CMMCorePlus | None = None,
        slackbot: MMSlackBot | None = None,
    ):
        super().__init__(parent)
        self._main_window = parent
        self._mmc = mmcore or CMMCorePlus.instance()

        # Tab widget for the viewers (preview and MDA)
        self._viewer_tab = QTabWidget()
        # Enable the close button on tabs
        self._viewer_tab.setTabsClosable(True)
        self._viewer_tab.tabCloseRequested.connect(self._close_tab)
        self._main_window._central_wdg_layout.addWidget(self._viewer_tab, 0, 0)

        # preview tab
        self._preview = Preview(self._main_window, mmcore=self._mmc)
        self._viewer_tab.addTab(self._preview, "Preview")
        # remove the preview tab close button
        self._viewer_tab.tabBar().setTabButton(*NO_R_BTN)
        self._viewer_tab.tabBar().setTabButton(*NO_L_BTN)

        # keep track of the current mda viewer
        self._current_viewer: MDAViewer | None = None

        self._mda_running: bool = False

        # keep track of the current event
        self._current_event_index: Mapping[str, int] | None = None

        # the _MDAWidget. It should have been set in the _MenuBar at startup
        self._mda = cast("_MDAWidget", self._main_window._menu_bar._mda)

        ev = self._mmc.events
        ev.continuousSequenceAcquisitionStarted.connect(self._set_preview_tab)
        ev.imageSnapped.connect(self._set_preview_tab)

        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)
        self._mmc.mda.events.sequenceCanceled.connect(self._on_sequence_canceled)
        self._mmc.mda.events.sequencePauseToggled.connect(self._enable_gui)
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)

        # handle the slackbot
        self._slackbot = slackbot
        if self._slackbot is None:
            return
        self._slackbot.slackMessage.connect(self._on_slack_bot_signal)

    def _on_slack_bot_signal(self, text: str) -> None:
        """Listen for slack bot signals."""
        if self._slackbot is None:
            return

        text = text.lower()
        if text == "status":
            if not self._mda_running:
                self._slackbot.send_message("⚠️ No MDA Sequence running! ⚠️")
                return
            self._slackbot.send_message(f"📃 Status 📃 -> {self._current_event_index}")
        elif text == "run":
            if self._mda_running:
                self._slackbot.send_message("⚠️ MDA Sequence already running! ⚠️")
                return
            self._mda.run_mda()
        elif text in {"stop", "cancel"}:
            if not self._mda_running:
                self._slackbot.send_message("⚠️ No MDA Sequence running! ⚠️")
                return
            self._mmc.mda.cancel()

    def _on_frame_ready(
        self, img: np.ndarray, event: useq.MDAEvent, metadata: dict
    ) -> None:
        """Called when a frame is ready."""
        self._current_event_index = event.index

    def _close_tab(self, index: int) -> None:
        """Close the tab at the given index."""
        if index == 0:
            return
        widget = self._viewer_tab.widget(index)
        self._viewer_tab.removeTab(index)
        widget.deleteLater()

        # Delete the current viewer
        del self._current_viewer
        self._current_viewer = None

    def _on_sequence_canceled(self, sequence: useq.MDASequence) -> None:
        """Called when the MDA sequence is cancelled."""
        # slack bot message
        if self._slackbot is not None:
            file_name = sequence.metadata.get(PYMMCW_METADATA_KEY, {}).get("save_name")
            self._slackbot.send_message(
                f"❌ MDA Sequence Cancelled! (file: {file_name}) ❌"
            )

    def _on_sequence_started(self, sequence: useq.MDASequence) -> None:
        """Called when the MDA sequence is started."""
        self._mda_running = True
        self._current_event_index = None

        # disable the menu bar
        self._main_window._menu_bar._enable(False)

        # pause until the viewer is ready
        self._mmc.mda.toggle_pause()
        # setup the viewer
        self._setup_viewer(sequence)
        # resume the sequence
        self._mmc.mda.toggle_pause()

        # slack bot message
        if self._slackbot is not None:
            file_name = sequence.metadata.get(PYMMCW_METADATA_KEY, {}).get("save_name")
            self._slackbot.send_message(
                f"🚀 MDA Sequence Started! (file: {file_name}) 🚀"
            )

    def _setup_viewer(self, sequence: useq.MDASequence) -> None:
        """Setup the MDAViewer."""
        # get the MDAWidget writer
        datastore = self._mda.writer if self._mda is not None else None
        self._current_viewer = MDAViewer(parent=self._main_window, datastore=datastore)

        # rename the viewer if there is a save_name' in the metadata or add a digit
        meta = cast(dict, sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        viewer_name = self._get_viewer_name(meta.get("save_name"))
        self._viewer_tab.addTab(self._current_viewer, viewer_name)
        self._viewer_tab.setCurrentWidget(self._current_viewer)

        # call it manually instead in _connect_viewer because this signal has been
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
        for v in range(self._viewer_tab.count()):
            tab_name = self._viewer_tab.tabText(v)
            if tab_name.startswith("MDA Viewer"):
                idx = tab_name.replace("MDA Viewer ", "")
                if idx.isdigit():
                    index = max(index, int(idx))
        return f"MDA Viewer {index + 1}"

    def _on_sequence_finished(self, sequence: useq.MDASequence) -> None:
        """Called when the MDA sequence is finished."""
        self._main_window._menu_bar._enable(True)

        self._mda_running = False
        self._current_event_index = None

        # reset the mda writer to None
        self._mda.writer = None

        # slack bot message
        if self._slackbot is not None:
            file_name = sequence.metadata.get(PYMMCW_METADATA_KEY, {}).get("save_name")
            self._slackbot.send_message(
                f"🏁 MDA Sequence Finished! (file: {file_name}) 🏁"
            )

        if self._current_viewer is None:
            return

        # enable the LUT drop down and the mono/composite button (temporary)
        self._enable_gui(True)

        # call it before we disconnect the signals or it will not be called
        self._current_viewer.data.sequenceFinished(sequence)

        self._disconnect_viewer(self._current_viewer)

        self._current_viewer = None

    def _connect_viewer(self, viewer: MDAViewer) -> None:
        self._mmc.mda.events.sequenceFinished.connect(viewer.data.sequenceFinished)
        self._mmc.mda.events.frameReady.connect(viewer.data.frameReady)

    def _disconnect_viewer(self, viewer: MDAViewer) -> None:
        """Disconnect the signals."""
        self._mmc.mda.events.frameReady.disconnect(viewer.data.frameReady)
        self._mmc.mda.events.sequenceFinished.disconnect(viewer.data.sequenceFinished)

    def _enable_gui(self, state: bool) -> None:
        """Pause the viewer when the MDA sequence is paused."""
        self._main_window._menu_bar._enable(state)
        if self._current_viewer is None:
            return

        # self._current_viewer._lut_drop.setEnabled(state)
        self._current_viewer._channel_mode_btn.setEnabled(state)

    def _set_preview_tab(self) -> None:
        """Set the preview tab."""
        if self._mda_running:
            return
        self._viewer_tab.setCurrentWidget(self._preview)
