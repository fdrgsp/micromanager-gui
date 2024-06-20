from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QElapsedTimer, QObject, QTimer, Signal
from qtpy.QtWidgets import QMessageBox, QWidget


@dataclass
class Peaks:
    """NamedTuple to store peak data."""

    peak: int | None = None
    amplitude: float | None = None
    raise_time: float | None = None
    decay_time: float | None = None
    # ... add whatever other data we need


@dataclass
class ROIData:
    """NamedTuple to store ROI data."""

    trace: list[float] | None = None
    #   dff: list[float] | None = None
    peaks: list[Peaks] | None = None
    mean_frequency: float | None = None
    mean_amplitude: float | None = None
    # ... add whatever other data we need


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


class ElapsedTimer(QObject):
    """A timer to keep track of the elapsed time."""

    elapsed_time_updated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_timer = QElapsedTimer()
        self._time_timer = QTimer()
        self._time_timer.timeout.connect(self._update_elapsed_time)

    def start(self) -> None:
        self._elapsed_timer.start()
        self._time_timer.start(1000)

    def stop(self) -> None:
        self._elapsed_timer.invalidate()
        self._time_timer.stop()

    def _update_elapsed_time(self) -> None:
        elapsed_ms = self._elapsed_timer.elapsed()
        elapsed_time_str = self._format_elapsed_time(elapsed_ms)
        self.elapsed_time_updated.emit(elapsed_time_str)

    @staticmethod
    def _format_elapsed_time(milliseconds: int) -> str:
        seconds = milliseconds // 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
