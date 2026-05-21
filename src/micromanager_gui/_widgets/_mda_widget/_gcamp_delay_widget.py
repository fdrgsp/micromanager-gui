from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


class GCaMPDelaySettings(TypedDict):
    """Channel name and shutter pre-open delay in milliseconds."""

    channel: str
    delay_ms: float


class GCaMPDelayWidget(QGroupBox):
    """Widget to enable a shutter pre-open delay for a specific channel."""

    def __init__(
        self,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._enable = QCheckBox("Enable GCaMP Delay")
        self._enable.setSizePolicy(*FIXED)
        self._enable.toggled.connect(self._on_toggled)

        self._settings_btn = QPushButton("GCaMP Settings...")
        self._settings_btn.setSizePolicy(*FIXED)
        self._settings_btn.setEnabled(False)
        self._settings_btn.clicked.connect(self._show_settings)

        self._dialog = GCaMPDelayDialog(mmcore=mmcore, parent=self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        layout.addWidget(self._enable)
        layout.addWidget(self._settings_btn)
        layout.addStretch()

    def value(self) -> GCaMPDelaySettings | None:
        """Return the current settings, or None if disabled."""
        if not self._enable.isChecked():
            return None
        return self._dialog.value()

    def setValue(self, value: GCaMPDelaySettings) -> None:
        """Set the widget state from a GCaMPDelaySettings dict."""
        self._enable.setChecked(True)
        self._dialog.setValue(value)

    def _show_settings(self) -> None:
        if self._dialog.isVisible():
            self._dialog.raise_()
        else:
            self._dialog.show()

    def _on_toggled(self, checked: bool) -> None:
        self._settings_btn.setEnabled(checked)
        if not checked:
            self._dialog.hide()


class GCaMPDelayDialog(QDialog):
    """Dialog to configure the GCaMP shutter pre-open delay."""

    def __init__(
        self,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GCaMP Delay Settings")

        self._mmc = mmcore

        channel_lbl = QLabel("Channel:")
        channel_lbl.setSizePolicy(*FIXED)

        self._channel_combo = QComboBox()
        self._channel_combo.setEditable(True)
        self._channel_combo.setSizePolicy(*FIXED)

        delay_lbl = QLabel("Delay:")
        delay_lbl.setSizePolicy(*FIXED)

        self._delay_spin = QDoubleSpinBox()
        self._delay_spin.setRange(0.0, 60_000.0)
        self._delay_spin.setValue(0.0)
        self._delay_spin.setDecimals(1)
        self._delay_spin.setSuffix(" ms")
        self._delay_spin.setSizePolicy(*FIXED)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        row_layout.addWidget(channel_lbl)
        row_layout.addWidget(self._channel_combo)
        row_layout.addWidget(delay_lbl)
        row_layout.addWidget(self._delay_spin)
        row_layout.addStretch()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(row)

        if self._mmc is not None:
            self._mmc.events.systemConfigurationLoaded.connect(
                self._on_sys_cfg_loaded
            )
            self.destroyed.connect(self._disconnect)
            self._populate_channels()

    def _on_sys_cfg_loaded(self) -> None:
        current = self._channel_combo.currentText()
        self._populate_channels()
        if (idx := self._channel_combo.findText(current)) >= 0:
            self._channel_combo.setCurrentIndex(idx)

    def _populate_channels(self) -> None:
        if self._mmc is None:
            return
        self._channel_combo.clear()
        try:
            groups = self._mmc.getOrGuessChannelGroup()
            group = groups[0] if groups else None
            if group:
                self._channel_combo.addItems(
                    list(self._mmc.getAvailableConfigs(group))
                )
        except Exception:
            pass

    def value(self) -> GCaMPDelaySettings:
        """Return the current channel and delay."""
        return {
            "channel": self._channel_combo.currentText(),
            "delay_ms": self._delay_spin.value(),
        }

    def setValue(self, value: GCaMPDelaySettings) -> None:
        """Set the dialog state from a GCaMPDelaySettings dict."""
        if (idx := self._channel_combo.findText(value["channel"])) >= 0:
            self._channel_combo.setCurrentIndex(idx)
        else:
            self._channel_combo.setCurrentText(value["channel"])
        self._delay_spin.setValue(value["delay_ms"])

    def _disconnect(self) -> None:
        if self._mmc is not None:
            self._mmc.events.systemConfigurationLoaded.disconnect(
                self._on_sys_cfg_loaded
            )
