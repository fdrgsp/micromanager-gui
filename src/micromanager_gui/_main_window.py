from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pathlib import Path

from ndv import NDViewer
from pymmcore_plus import CMMCorePlus
from qtpy.QtWidgets import (
    QGridLayout,
    QMainWindow,
    QWidget,
)

from micromanager_gui.readers import TensorstoreZarrReader

from ._core_link import CoreViewersLink
from ._menubar._menubar import _MenuBar
from ._mmcore_engine._engine import Engine
from ._slackbot._mm_slackbot import MMSlackBot
from ._toolbar._shutters_toolbar import _ShuttersToolbar
from ._toolbar._snap_live import _SnapLive

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent


class MicroManagerGUI(QMainWindow):
    """Micro-Manager minimal GUI."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mmcore: CMMCorePlus | None = None,
        config: str | None = None,
        slackbot: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

        # slack bot to handle slack messages
        self._slackbot = MMSlackBot() if slackbot else None
        # if the slackbot is not loaded correctly, set it to None. user will get a
        # warning message from the SlackBot class

        self.setAcceptDrops(True)

        self.setWindowTitle("Micro-Manager")

        # get global CMMCorePlus instance
        self._mmc = mmcore or CMMCorePlus.instance()
        # set the engine
        self._mmc.mda.set_engine(Engine(self._mmc, slackbot=self._slackbot))

        # central widget
        central_wdg = QWidget(self)
        self._central_wdg_layout = QGridLayout(central_wdg)
        self.setCentralWidget(central_wdg)

        # add the menu bar (and the logic to create/show widgets)
        self._menu_bar = _MenuBar(parent=self, mmcore=self._mmc)
        self.setMenuBar(self._menu_bar)

        # add toolbar
        self._snap_live_toolbar = _SnapLive(parent=self, mmcore=self._mmc)
        self.addToolBar(self._snap_live_toolbar)
        self._shutters_toolbar = _ShuttersToolbar(parent=self, mmcore=self._mmc)
        self.addToolBar(self._shutters_toolbar)

        # link the MDA viewers
        self._core_link = CoreViewersLink(
            self, mmcore=self._mmc, slackbot=self._slackbot
        )

        # extend size to fill the screen
        self.showMaximized()

        if config is not None:
            try:
                self._mmc.unloadAllDevices()
                self._mmc.loadSystemConfiguration(config)
            except FileNotFoundError:
                # don't crash if the user passed an invalid config
                warn(f"Config file {config} not found. Nothing loaded.", stacklevel=2)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Override the closeEvent method to stop the SlackBotProcess."""
        if self._slackbot is not None:
            self._slackbot._slack_process.stop()
        # delete any remaining widgets
        from qtpy.QtWidgets import QApplication

        if qapp := QApplication.instance():
            if remaining := qapp.topLevelWidgets():
                for w in remaining:
                    w.deleteLater()
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Open a tensorstore from a directory dropped in the window."""
        for idx, url in enumerate(event.mimeData().urls()):
            path = Path(url.toLocalFile())

            sw = self._open_datastore(idx, path)

            if sw is not None:
                self._core_link._viewer_tab.addTab(sw, f"datastore_{idx}")
                self._core_link._viewer_tab.setCurrentWidget(sw)

        super().dropEvent(event)

    def _open_datastore(self, idx: int, path: Path) -> NDViewer | None:
        if path.name.endswith(".tensorstore.zarr"):
            try:
                reader = TensorstoreZarrReader(path)
                return NDViewer(reader.store, parent=self)
            except Exception as e:
                warn(f"Error opening tensorstore-zarr: {e}!", stacklevel=2)
                return None
        # TODO: implement with OMEZarrReader
        # elif path.name.endswith(".ome.zarr"):
        #     try:
        #         reader = OMEZarrReader(path)
        #         return NDViewer(reader.store, parent=self)
        #     except Exception as e:
        #             warn(f"Error opening OME-zarr: {e}!", stacklevel=2)
        #             return None
        else:
            warn(f"Not yet supported format: {path.name}!", stacklevel=2)
            return None
