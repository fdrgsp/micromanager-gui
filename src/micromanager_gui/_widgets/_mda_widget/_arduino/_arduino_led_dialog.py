from __future__ import annotations

from typing import TypedDict, cast

from fonticon_mdi6 import MDI6
from pyfirmata2 import Arduino
from pyfirmata2.pyfirmata2 import Pin
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QPainter, QPaintEvent, QPen
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon


class StimulationValues(TypedDict):
    arduino_port: str
    arduino_led_pin: str
    initial_delay: int
    interval: int
    num_pulses: int
    led_start_power: int
    led_power_increment: int
    led_pulse_duration: float
    pulse_on_frame: dict[int, int]


FIXED = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
PIN = "d:3:p"
TIMEPOINTS_TXT = "(set in the MDA Time Tab)"
GREEN = "#00FF00"
RED = "#C33"
MAX_LBL_WIDTH = 200


class ArduinoLedControl(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Arduino LED Control")

        self._arduino_board: Arduino | None = None
        self._led_pin: Pin | None = None
        self._led_on_frames: list[int] = []
        self._led_power_used: list[int] = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # GROUP - to detect the arduino board
        detect_board = QGroupBox("Arduino")
        port_lbl = QLabel("Arduino Port:")
        port_lbl.setSizePolicy(FIXED)
        self._board_port = QLineEdit()
        self._board_port.setPlaceholderText("e.g COM3")
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._connect_btn.setToolTip("Click to detect the board. If no ")
        self._connect_btn.clicked.connect(self._detect_arduino_board)
        pin_lbl = QLabel("Arduino LED Pin:")
        pin_lbl.setSizePolicy(FIXED)
        self._led_pin_info = QLineEdit()
        self._led_pin_info.setText(PIN)  # default to 'd:3:p'
        # layout
        detect_gp_layout = QGridLayout(detect_board)
        detect_gp_layout.setContentsMargins(10, 15, 10, 10)
        detect_gp_layout.setSpacing(5)
        detect_gp_layout.addWidget(port_lbl, 0, 0)
        detect_gp_layout.addWidget(self._board_port, 0, 1)
        detect_gp_layout.addWidget(self._connect_btn, 0, 2)
        detect_gp_layout.addWidget(pin_lbl, 1, 0)
        detect_gp_layout.addWidget(self._led_pin_info, 1, 1, 1, 2)

        # GROUP - to set on which frames to turn on the LED
        frame_group = QGroupBox("Stimulation Frames")
        # initial delay
        initial_delay_lbl = QLabel("Initial Delay:")
        initial_delay_lbl.setSizePolicy(FIXED)
        self._initial_delay_spin = QSpinBox()
        self._initial_delay_spin.setRange(0, 100000)
        self._initial_delay_spin.setSuffix(" frames")
        self._initial_delay_spin.valueChanged.connect(self._on_frame_values_changed)
        # interval
        interval_lbl = QLabel("Interval:")
        initial_delay_lbl.setSizePolicy(FIXED)
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(0, 100000)
        self._interval_spin.setSuffix(" frames")
        self._interval_spin.valueChanged.connect(self._on_frame_values_changed)
        # number of pulses
        num_pulses_lbl = QLabel("Number of Pulses:")
        num_pulses_lbl.setSizePolicy(FIXED)
        self._num_pulses_spin = QSpinBox()
        self._num_pulses_spin.setRange(0, 100000)
        self._num_pulses_spin.valueChanged.connect(self._on_frame_values_changed)
        self._num_pulses_spin.valueChanged.connect(self._on_led_values_changed)
        # separator
        separator = _SeparatorWidget()
        # total number of frames
        time_points_lbl = QLabel("TimePoints:")
        self._timepoints = QLineEdit()
        self._timepoints.setReadOnly(True)
        self._timepoints.setStyleSheet("border: 0")
        # layout
        frame_gp_layout = QGridLayout(frame_group)
        frame_gp_layout.setContentsMargins(10, 15, 10, 10)
        frame_gp_layout.setSpacing(5)
        frame_gp_layout.addWidget(initial_delay_lbl, 0, 0)
        frame_gp_layout.addWidget(self._initial_delay_spin, 0, 1)
        frame_gp_layout.addWidget(interval_lbl, 1, 0)
        frame_gp_layout.addWidget(self._interval_spin, 1, 1)
        frame_gp_layout.addWidget(num_pulses_lbl, 2, 0)
        frame_gp_layout.addWidget(self._num_pulses_spin, 2, 1)
        frame_gp_layout.addWidget(separator, 3, 0, 1, 2)
        frame_gp_layout.addWidget(time_points_lbl, 4, 0)
        frame_gp_layout.addWidget(self._timepoints, 4, 1)

        # GROUP - to set the led power
        led_group = QGroupBox("LED")
        # start power
        led_start_pwr_lbl = QLabel("Start Power")
        led_start_pwr_lbl.setSizePolicy(FIXED)
        self._led_start_power = QSpinBox()
        self._led_start_power.setRange(0, 100)
        self._led_start_power.setSuffix(" %")
        self._led_start_power.valueChanged.connect(self._on_led_values_changed)
        # power increment
        led_power_increment_lbl = QLabel("Power Increment:")
        led_power_increment_lbl.setSizePolicy(FIXED)
        self._led_power_increment = QSpinBox()
        self._led_power_increment.setRange(0, 100)
        self._led_power_increment.setSuffix(" %")
        self._led_power_increment.valueChanged.connect(self._on_led_values_changed)
        # pulse duration
        pulse_duration_lbl = QLabel("Pulse Duration:")
        pulse_duration_lbl.setSizePolicy(FIXED)
        self._pulse_duration_spin = QSpinBox()
        self._pulse_duration_spin.setRange(0, 100000)
        self._pulse_duration_spin.setSuffix(" ms")
        # separator
        separator = _SeparatorWidget()
        # led info
        _led_powers_lbl = QLabel("Power(s):")
        self._led_pwrs = QLineEdit()
        self._led_pwrs.setReadOnly(True)
        self._led_pwrs.setStyleSheet("border: 0")
        self._led_pwrs_icon_lbl = QLabel()
        self._led_pwrs_icon_lbl.setSizePolicy(FIXED)
        self._led_pwrs_icon_lbl.setPixmap(
            icon(MDI6.alert_outline, color=RED).pixmap(QSize(25, 25))
        )
        self._led_pwrs_icon_lbl.hide()

        # set linedit height to the same as the icon label
        fixed_height = self._led_pwrs_icon_lbl.minimumSizeHint().height()
        self._led_pwrs.setFixedHeight(fixed_height)
        self._timepoints.setFixedHeight(fixed_height)

        # layout
        led_gp_layout = QGridLayout(led_group)
        led_gp_layout.setContentsMargins(10, 15, 10, 10)
        led_gp_layout.setSpacing(5)
        led_gp_layout.addWidget(led_start_pwr_lbl, 0, 0)
        led_gp_layout.addWidget(self._led_start_power, 0, 1, 1, 2)
        led_gp_layout.addWidget(led_power_increment_lbl, 1, 0)
        led_gp_layout.addWidget(self._led_power_increment, 1, 1, 1, 2)
        led_gp_layout.addWidget(pulse_duration_lbl, 2, 0)
        led_gp_layout.addWidget(self._pulse_duration_spin, 2, 1, 1, 2)
        led_gp_layout.addWidget(separator, 3, 0, 1, 3)
        led_gp_layout.addWidget(_led_powers_lbl, 4, 0)
        led_gp_layout.addWidget(self._led_pwrs, 4, 1)
        led_gp_layout.addWidget(self._led_pwrs_icon_lbl, 4, 2)

        # connection indicator labels
        connection_info = QWidget()
        connection_info_layout = QHBoxLayout(connection_info)
        connection_info_layout.setSpacing(0)
        connection_info_layout.setContentsMargins(0, 0, 0, 0)
        self._arduino_connected_icon = QLabel()
        self._arduino_connected_icon.setSizePolicy(FIXED)
        self._arduino_connected_text = QLabel()
        self._arduino_connected_text.setSizePolicy(FIXED)
        connection_info_layout.addWidget(self._arduino_connected_icon)
        connection_info_layout.addWidget(self._arduino_connected_text)

        # layout
        btns_layout = QHBoxLayout()
        btns_layout.setSpacing(10)
        btns_layout.addWidget(connection_info)
        btns_layout.addStretch()

        # add all the widgets to the main layout
        main_layout.addWidget(detect_board)
        main_layout.addWidget(frame_group)
        main_layout.addWidget(led_group)
        main_layout.addLayout(btns_layout)

        # set the fixed size for the labels
        for lbl in [
            port_lbl,
            led_start_pwr_lbl,
            led_power_increment_lbl,
            pulse_duration_lbl,
            initial_delay_lbl,
            interval_lbl,
            num_pulses_lbl,
        ]:
            lbl.setFixedSize(num_pulses_lbl.minimumSizeHint())

        self._enable(False)

    # ________________________Public Methods________________________

    def board(self) -> Arduino | None:
        """Return the connected Arduino board object."""
        return self._arduino_board

    def ledPin(self) -> Pin | None:
        """Return the connected LED Pin object."""
        return self._led_pin

    def is_max_power_exceeded(self) -> bool:
        """Check whether the max power is exceeded."""
        return not self._led_power_used

    def value(self) -> StimulationValues:
        """Return the values set in the dialog."""
        return {
            "arduino_port": self._board_port.text(),
            "arduino_led_pin": self._led_pin_info.text(),
            "initial_delay": self._initial_delay_spin.value(),
            "interval": self._interval_spin.value(),
            "num_pulses": self._num_pulses_spin.value(),
            "led_start_power": self._led_start_power.value(),
            "led_power_increment": self._led_power_increment.value(),
            "led_pulse_duration": self._pulse_duration_spin.value(),
            "pulse_on_frame": self._get_pulse_on_frame(),
        }

    def setValue(self, values: StimulationValues | dict) -> None:
        """Set the values in the dialog.

        Note that "pulse_on_frame" is not necessary to be set in the values dictionary
        as it is calculated from the other values. If provided, it will be ignored.
        """
        self._board_port.setText(values.get("arduino_port", ""))
        self._led_pin_info.setText(values.get("arduino_led_pin", ""))
        self._initial_delay_spin.setValue(values.get("initial_delay", 0))
        self._interval_spin.setValue(values.get("interval", 0))
        self._num_pulses_spin.setValue(values.get("num_pulses", 0))
        self._led_start_power.setValue(values.get("led_start_power", 0))
        self._led_power_increment.setValue(values.get("led_power_increment", 0))
        self._pulse_duration_spin.setValue(values.get("led_pulse_duration", 0))
        self._on_frame_values_changed()
        self._on_led_values_changed()

    # ________________________Private Methods________________________

    def _enable(self, state: bool) -> None:
        lbl_icon = MDI6.check_bold if state else MDI6.close_octagon_outline
        lbl_icon_size = QSize(20, 20) if state else QSize(30, 30)
        lbl_icon_color = GREEN if state else RED
        text = "Arduino Connected!" if state else "Arduino not Connected!"
        pixmap = icon(lbl_icon, color=lbl_icon_color).pixmap(lbl_icon_size)
        self._arduino_connected_icon.setPixmap(pixmap)
        self._arduino_connected_text.setText(text)

    def _get_pulse_on_frame(self) -> dict[int, int]:
        """Return the pulse_on_frame dictionary.

        The dictionary contains the frame number as key and the respective led power to
        be used.
        """
        # make sure that the led is not exceeding the max 100% power
        if not self._led_power_used:
            return {"error": "Max power exceeded!!!"}  # type: ignore
        return dict(zip(self._led_on_frames, self._led_power_used))

    def _detect_arduino_board(self) -> None:
        """Detect the Arduino board and update the GUI."""
        # if the port is empty, try to autodetect the board
        if not self._board_port.text():
            try:
                self._arduino_board = Arduino(Arduino.AUTODETECT)
                self._update_arduino_board_info()
            except Exception:
                return self._show_critical_messagebox(
                    "Unable to Autodetect the Arduino Board. \nPlease insert the port "
                    "manually in the 'Arduino Port' field."
                )
        # if the port is specified, try to detect the board on the specified port
        else:
            try:
                self._arduino_board = Arduino(self._board_port.text())
                self._update_arduino_board_info()
            except Exception:
                return self._show_critical_messagebox(
                    "Unable to detect the Arduino Board on the specified port."
                )

    def _update_arduino_board_info(self) -> None:
        """Update the GUI with the detected Arduino board info."""
        if self._arduino_board is None:
            self._reset()
            return

        # make sure the led pin is available
        try:
            self._led_pin = cast(
                Pin, self._arduino_board.get_pin(self._led_pin_info.text())
            )
        except Exception:
            return self._show_critical_messagebox(
                "Unable to detect the LED Pin on the specified Arduino Board."
            )

        self._led_pin.write(0.0)
        self._board_port.setText(self._arduino_board.sp.port or "")
        self._enable(True)

    def _reset(self) -> None:
        """Reset to the initial state."""
        self._arduino_board = None
        self._led_pin = None
        self._board_port.clear()
        self._enable(False)

    def _show_critical_messagebox(self, message: str) -> None:
        """Show a critical message box with the given message."""
        self._reset()
        self._enable(False)
        QMessageBox.critical(
            self, "Arduino Board Error", message, buttons=QMessageBox.Ok
        )
        return

    def _on_frame_values_changed(self) -> None:
        """Update the frame info and set the led_on_frames."""
        self._led_on_frames.clear()

        if not self._num_pulses_spin.value():
            return

        # get total timepoints that should be set in the MDA time tab
        timepoints = self._initial_delay_spin.value() + (
            (self._interval_spin.value() or 1) * self._num_pulses_spin.value()
        )
        self._timepoints.setText(f"{timepoints} {TIMEPOINTS_TXT}")

        # get the frames in which the led will be turned on
        fr = self._initial_delay_spin.value()
        for _ in range(self._num_pulses_spin.value()):
            self._led_on_frames.append(fr)
            # Add 1 to account for the duration of the pulse
            fr += self._interval_spin.value() + 1

        pulse_on_timepoint = [str(f) for f in self._led_on_frames]
        self._timepoints.setToolTip(f"Pulse On Timepoint: {pulse_on_timepoint}")

    def _on_led_values_changed(self) -> None:
        """Update the led power info."""
        self._led_power_used.clear()

        # get the led powers to be used
        pwr = self._led_start_power.value()
        for _ in range(self._num_pulses_spin.value()):
            self._led_power_used.append(pwr)
            pwr = pwr + self._led_power_increment.value()

        self._led_pwrs.setText(str(self._led_power_used))
        self._led_pwrs_icon_lbl.hide()
        self._led_pwrs.setToolTip(str(self._led_power_used))

        # check if the max power is not exceeded
        power_max = self._led_start_power.value() + (
            self._led_power_increment.value() * (self._num_pulses_spin.value() - 1)
        )

        if power_max > 100:
            self._led_power_used.clear()
            self._led_pwrs_icon_lbl.show()
            self._led_pwrs.setText("Max power exceeded!!!")


class _SeparatorWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.SolidLine))
        painter.drawLine(self.rect().topLeft(), self.rect().topRight())
