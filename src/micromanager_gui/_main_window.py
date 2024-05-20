import sys
from pathlib import Path
from warnings import warn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import (
    ConfigWizard,
    GroupPresetTableWidget,
    MDAWidget,
    PixelConfigurationWidget,
    PropertyBrowser,
)
from pymmcore_widgets.hcwizard.intro_page import SRC_CONFIG
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAction,
    QFileDialog,
    QGridLayout,
    QMainWindow,
    QMenuBar,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QWidget,
)

from ._core_link import MDAViewersLink
from ._widgets._preview import Preview
from ._widgets._shutters_toolbar import _ShuttersToolbar
from ._widgets._stage_control import _StagesControlWidget

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
        # other widgets
        self._wizard = ConfigWizard(parent=self, core=self._mmc)
        self._wizard.setWindowFlags(FLAGS)
        self._prop_browser = PropertyBrowser(parent=self, mmcore=self._mmc)
        self._prop_browser.setWindowFlags(FLAGS)
        self._stage_wdg = _StagesControlWidget(parent=self, mmcore=self._mmc)
        self._stage_wdg.setWindowFlags(FLAGS)
        self._px_cfg = PixelConfigurationWidget(parent=self, mmcore=self._mmc)
        self._px_cfg.setWindowFlags(FLAGS)

        self._mda_link = MDAViewersLink(self, mmcore=self._mmc)

        # add the menu bar
        self._add_menu()

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

    def _add_menu(self) -> None:
        """Add the menu bar to the main window."""
        menubar = QMenuBar(self)

        # configurations_menu
        configurations_menu = menubar.addMenu("System Configurations")
        # hardware cfg wizard
        self.act_cfg_wizard = QAction("Hardware Configuration Wizard", self)
        self.act_cfg_wizard.triggered.connect(self._show_config_wizard)
        configurations_menu.addAction(self.act_cfg_wizard)
        # save cfg
        self.act_save_configuration = QAction("Save Configuration", self)
        self.act_save_configuration.triggered.connect(self._save_cfg)
        configurations_menu.addAction(self.act_save_configuration)
        # load cfg
        self.act_load_configuration = QAction("Load Configuration", self)
        self.act_load_configuration.triggered.connect(self._load_cfg)
        configurations_menu.addAction(self.act_load_configuration)

        # widgets_menu
        widgets_menu = menubar.addMenu("Widgets")
        # property browser
        self.act_property_browser = QAction("Property Browser", self)
        self.act_property_browser.triggered.connect(self._show_property_browser)
        widgets_menu.addAction(self.act_property_browser)
        # stage control
        self.act_stage_control = QAction("Stage Control", self)
        self.act_stage_control.triggered.connect(self._show_stage_control)
        widgets_menu.addAction(self.act_stage_control)
        # pixel configuration
        self.act_pixel_configuration = QAction("Pixel Configuration", self)
        self.act_pixel_configuration.triggered.connect(self._show_px_cfg)
        widgets_menu.addAction(self.act_pixel_configuration)

    def _close_tab(self, index: int) -> None:
        """Close the tab at the given index."""
        widget = self._viewer_tab.widget(index)
        self._viewer_tab.removeTab(index)
        widget.deleteLater()

    def _save_cfg(self) -> None:
        (filename, _) = QFileDialog.getSaveFileName(
            self, "Save Micro-Manager Configuration."
        )
        if filename:
            self._mmc.saveSystemConfiguration(
                filename if str(filename).endswith(".cfg") else f"{filename}.cfg"
            )

    def _load_cfg(self) -> None:
        """Open file dialog to select a config file."""
        (filename, _) = QFileDialog.getOpenFileName(
            self, "Select a Micro-Manager configuration file", "", "cfg(*.cfg)"
        )
        if filename:
            self._mmc.unloadAllDevices()
            self._mmc.loadSystemConfiguration(filename)

    def _show_config_wizard(self) -> None:
        """Show the Micro-Manager Hardware Configuration Wizard."""
        if self._wizard.isVisible():
            self._wizard.raise_()
        else:
            current_cfg = self._mmc.systemConfigurationFile() or ""
            self._wizard.setField(SRC_CONFIG, current_cfg)
            self._wizard.show()

    def _show_property_browser(self) -> None:
        """Show the Micro-Manager Property Browser."""
        if self._prop_browser.isVisible():
            self._prop_browser.raise_()
        else:
            self._prop_browser.show()

    def _show_stage_control(self) -> None:
        """Show the Micro-Manager Stage Control."""
        if self._stage_wdg.isVisible():
            self._stage_wdg.raise_()
        else:
            self._stage_wdg.show()

    def _show_px_cfg(self) -> None:
        """Show the Micro-Manager Pixel Configuration."""
        if self._px_cfg.isVisible():
            self._px_cfg.raise_()
        else:
            self._px_cfg.show()
