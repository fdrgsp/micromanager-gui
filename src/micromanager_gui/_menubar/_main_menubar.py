from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import (
    ConfigWizard,
    PixelConfigurationWidget,
    PropertyBrowser,
)
from pymmcore_widgets.hcwizard.intro_page import SRC_CONFIG
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QAction, QFileDialog, QMenuBar, QWidget

from micromanager_gui._widgets._stage_control import _StagesControlWidget

FLAGS = Qt.WindowType.Dialog


class _MenuBar(QMenuBar):
    """Menu Bar for the Micro-Manager GUI."""

    def __init__(
        self, parent: QWidget | None = None, *, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent)
        self._mmc = mmcore or CMMCorePlus.instance()

        # widgets
        self._wizard: ConfigWizard | None = None
        self._prop_browser = PropertyBrowser(parent=self, mmcore=self._mmc)
        self._prop_browser.setWindowFlags(FLAGS)
        self._stage_wdg = _StagesControlWidget(parent=self, mmcore=self._mmc)
        self._stage_wdg.setWindowFlags(FLAGS)
        self._px_cfg = PixelConfigurationWidget(parent=self, mmcore=self._mmc)
        self._px_cfg.setWindowFlags(FLAGS)

        # configurations_menu
        self._configurations_menu = self.addMenu("System Configurations")
        # hardware cfg wizard
        self._act_cfg_wizard = QAction("Hardware Configuration Wizard", self)
        self._act_cfg_wizard.triggered.connect(self._show_config_wizard)
        self._configurations_menu.addAction(self._act_cfg_wizard)
        # save cfg
        self._act_save_configuration = QAction("Save Configuration", self)
        self._act_save_configuration.triggered.connect(self._save_cfg)
        self._configurations_menu.addAction(self._act_save_configuration)
        # load cfg
        self._act_load_configuration = QAction("Load Configuration", self)
        self._act_load_configuration.triggered.connect(self._load_cfg)
        self._configurations_menu.addAction(self._act_load_configuration)

        # widgets_menu
        self._widgets_menu = self.addMenu("Widgets")
        # property browser
        self._act_property_browser = QAction("Property Browser", self)
        self._act_property_browser.triggered.connect(self._show_property_browser)
        self._widgets_menu.addAction(self._act_property_browser)
        # stage control
        self._act_stage_control = QAction("Stage Control", self)
        self._act_stage_control.triggered.connect(self._show_stage_control)
        self._widgets_menu.addAction(self._act_stage_control)
        # pixel configuration
        self._act_pixel_configuration = QAction("Pixel Configuration", self)
        self._act_pixel_configuration.triggered.connect(self._show_px_cfg)
        self._widgets_menu.addAction(self._act_pixel_configuration)

    def _enable(self, enable: bool) -> None:
        """Enable or disable the actions."""
        self._configurations_menu.setEnabled(enable)
        self._widgets_menu.setEnabled(enable)

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
        if self._wizard is None:
            self._wizard = ConfigWizard(parent=self, core=self._mmc)
            self._wizard.setWindowFlags(FLAGS)
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
