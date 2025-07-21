from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ._arduino_led_dialog import ArduinoLedControl, StimulationValues

if TYPE_CHECKING:
    from pyfirmata2 import Arduino, Pin

FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class ArduinoLedWidget(QGroupBox):
    """Widget to enable LED stimulation with Arduino."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._arduino_led_control = ArduinoLedControl(self)

        self._enable_led = QCheckBox("Enable Arduino LED stimulation")
        self._enable_led.setSizePolicy(FIXED)
        self._enable_led.toggled.connect(self._on_enable_toggled)

        self._settings_btn = QPushButton("Arduino Settings...")
        self._settings_btn.setSizePolicy(FIXED)
        self._settings_btn.setEnabled(False)
        self._settings_btn.clicked.connect(self._show_settings)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        layout.addWidget(self._enable_led)
        layout.addWidget(self._settings_btn)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        """Return True if the checkbox is checked."""
        return cast("bool", self._enable_led.isChecked())

    def board(self) -> Arduino | None:
        """Return the current Arduino board object."""
        return self._arduino_led_control.board()

    def ledPin(self) -> Pin | None:
        """Return the current LED Pin object."""
        return self._arduino_led_control.ledPin()

    def is_max_power_exceeded(self) -> bool:
        """Return True if the maximum power of the LED has been exceeded."""
        return self._arduino_led_control.is_max_power_exceeded()

    def value(self) -> StimulationValues | dict:
        """Return the current value of the widget."""
        if self._enable_led.isChecked() and self._arduino_led_control._arduino_board:
            return self._arduino_led_control.value()
        return {}

    def setValue(self, value: StimulationValues | dict) -> None:
        """Set the current value of the widget."""
        if value:
            self._enable_led.setChecked(True)
            self._arduino_led_control.setValue(value)
        else:
            self._enable_led.setChecked(False)

    def _show_settings(self) -> None:
        """Show the settings dialog."""
        if self._arduino_led_control.isVisible():
            self._arduino_led_control.raise_()
        else:
            self._arduino_led_control.show()

    def _on_enable_toggled(self, checked: bool) -> None:
        """Enable/disable the analysis settings button based on the checkbox state."""
        self._settings_btn.setEnabled(checked)
        if not checked:
            self._arduino_led_control.hide()
