import sys
from pathlib import Path
from warnings import warn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import (
    GroupPresetTableWidget,
    MDAWidget,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QMainWindow,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QWidget,
)

from ._core_link import MDAViewersLink
from ._menubar._main_menubar import _MenuBar
from ._toolbar._shutters_toolbar import _ShuttersToolbar
from ._widgets._preview import Preview

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

        self._mmc = mmcore or CMMCorePlus.instance()

        # central widget
        central_wdg = QWidget(self)
        self._central_wdg_layout = QGridLayout(central_wdg)
        self.setCentralWidget(central_wdg)

        # Tab widget for the viewers
        self._viewer_tab = QTabWidget()
        # Enable the close button on tabs
        self._viewer_tab.setTabsClosable(True)
        self._viewer_tab.tabCloseRequested.connect(self._close_tab)
        self._central_wdg_layout.addWidget(self._viewer_tab, 0, 0)
        # Preview tab
        self._preview = Preview(self, mmcore=self._mmc)
        self._viewer_tab.addTab(self._preview, "Preview")
        # remove the close button from the preview tab
        self._viewer_tab.tabBar().setTabButton(0, QTabBar.ButtonPosition.LeftSide, None)

        # Tab widget for the widgets
        self.widget_tab = QTabWidget()
        self._central_wdg_layout.addWidget(self.widget_tab, 0, 1)
        self.widget_tab.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )

        # main widgets
        self._group_preset = GroupPresetTableWidget()
        self.widget_tab.addTab(self._group_preset, "Groups and Presets")
        self._mda = MDAWidget()
        self.widget_tab.addTab(self._mda, "MDA Acquisition")

        # link the MDA viewers
        self._mda_link = MDAViewersLink(self, mmcore=self._mmc)

        # add the menu bar
        self.setMenuBar(_MenuBar(parent=self, mmcore=self._mmc))

        # add toolbar
        self._shutters_toolbar = _ShuttersToolbar(parent=self, mmcore=self._mmc)
        self.addToolBar(self._shutters_toolbar)

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
