import sys
from pathlib import Path
from warnings import warn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import GroupPresetTableWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QWidget,
)

from ._core_link import CoreViewersLink
from ._menubar._main_menubar import _MenuBar
from ._mmcore_engine._engine import ArduinoEngine
from ._toolbar._shutters_toolbar import _ShuttersToolbar
from ._toolbar._snap_live import _SnapLive
from ._widgets._mda_widget import _MDAWidget

FLAGS = Qt.WindowType.Dialog


class MicroManagerGUI(QMainWindow):
    """Micro-Manager minimal GUI."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mmcore: CMMCorePlus | None = None,
        config: str | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Micro-Manager")

        # extend size to fill the screen
        self.showMaximized()

        # get global CMMCorePlus instance
        self._mmc = CMMCorePlus.instance()
        # set the engine
        self._mmc.mda.set_engine(ArduinoEngine(self._mmc))

        # central widget
        central_wdg = QWidget(self)
        self._central_wdg_layout = QGridLayout(central_wdg)
        self.setCentralWidget(central_wdg)

        # Tab widget for the viewers (preview and MDA)
        self._viewer_tab = QTabWidget()
        # Enable the close button on tabs
        self._viewer_tab.setTabsClosable(True)
        self._viewer_tab.tabCloseRequested.connect(self._close_tab)
        self._central_wdg_layout.addWidget(self._viewer_tab, 0, 0)

        # Tab widget for the widgets
        self.widget_tab = QTabWidget()
        self._central_wdg_layout.addWidget(self.widget_tab, 0, 1)
        # main widgets
        self._group_preset = GroupPresetTableWidget()
        self.widget_tab.addTab(self._group_preset, "Groups and Presets")
        self._mdaScrollArea = QScrollArea()
        self._mda = _MDAWidget()
        self._mdaScrollArea.setWidget(self._mda)
        self._mdaScrollArea.setWidgetResizable(True)
        self.widget_tab.addTab(self._mdaScrollArea, "MDA Acquisition")
        # set fixed scroll area size
        self.widget_tab.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        width = self._mda.sizeHint().width()
        # set setFixedWidth to (width + 2%width)
        self._mdaScrollArea.setFixedWidth(width + width // 50)

        # add the menu bar
        self._menu_bar = _MenuBar(parent=self, mmcore=self._mmc)
        self.setMenuBar(self._menu_bar)

        # add toolbar
        self._shutters_toolbar = _ShuttersToolbar(parent=self, mmcore=self._mmc)
        self.addToolBar(self._shutters_toolbar)
        self._snap_live_toolbar = _SnapLive(parent=self, mmcore=self._mmc)
        self.addToolBar(self._snap_live_toolbar)

        # link the MDA viewers
        self._core_link = CoreViewersLink(self, mmcore=self._mmc)

        if config is not None:
            try:
                self._mmc.unloadAllDevices()
                self._mmc.loadSystemConfiguration(config)
            except FileNotFoundError:
                # don't crash if the user passed an invalid config
                warn(f"Config file {config} not found. Nothing loaded.", stacklevel=2)

    def _close_tab(self, index: int) -> None:
        """Close the tab at the given index."""
        widget = self._viewer_tab.widget(index)
        self._viewer_tab.removeTab(index)
        widget.deleteLater()
