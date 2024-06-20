from __future__ import annotations

from typing import NamedTuple, cast

from qtpy.QtCore import QElapsedTimer, QObject, QTimer, Signal
from qtpy.QtWidgets import QMessageBox, QWidget


class Peaks(NamedTuple):
    """NamedTuple to store peak data."""

    peak: int | None = None
    amplitude: float | None = None
    raise_time: float | None = None
    decay_time: float | None = None
    # ... add whatever other data we need


class ROIData(NamedTuple):
    """NamedTuple to store ROI data."""

    trace: list[float] | None = None
    #   dff: list[float] | None = None
    peaks: list[Peaks] | None = None
    mean_frequency: float | None = None
    mean_amplitude: float | None = None
    # ... add whatever other data we need


def load_analysis_data(analysis_json_file_path: str) -> dict[str, dict[str, ROIData]]:
    """Load the analysis data from the given JSON file."""
    import json

    with open(analysis_json_file_path) as f:
        data = cast(dict, json.load(f))
        for key in data.keys():
            for i in range(1, len(data[key]) + 1):
                # if there is the 'peaks' key, convert the list[dicts] to list[Peaks]
                if "peaks" in data[key][str(i)]:
                    data[key][str(i)]["peaks"] = [
                        Peaks(**peak) for peak in data[key][str(i)]["peaks"]
                    ]
                # convert the dict to ROIData
                data[key][str(i)] = ROIData(**data[key][str(i)])
    return data


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
