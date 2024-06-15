from __future__ import annotations

from typing import NamedTuple, cast

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
            for i in range(len(data[key])):
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
