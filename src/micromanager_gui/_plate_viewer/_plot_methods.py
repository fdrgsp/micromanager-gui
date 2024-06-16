from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from ._graph_widget import _GraphWidget
    from ._util import ROIData


def plot_raw_traces(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces."""
    ax = widget.figure.add_subplot(111)
    ax.set_title(f"{widget._fov} - raw traces")
    ax.get_yaxis().set_visible(False)
    offset = 10
    count = 0
    for i, key in enumerate(data):
        if rois is not None and i not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        ax.plot(np.array(roi_data.trace) + count * offset, label=f"ROI {i}")
        count += 1
    widget.canvas.draw()


def plot_delta_f_over_f(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    # TODO: dff should be calculated in the analysis and stored in the ROIData class
    # here we will only need to plot roi_data.dff. Also use a better methodfor dff
    """Plot the delta f over f traces."""
    ax = widget.figure.add_subplot(111)
    ax.set_title(f"{widget._fov} - DeltaF/F0")
    ax.get_yaxis().set_visible(False)
    offset = 10
    count = 0
    for i, key in enumerate(data):
        if rois is not None and i not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        traces = roi_data.trace
        if not traces:
            continue
        median = np.median(traces)
        dff = (np.array(traces) - median) / median
        ax.plot(dff + count * offset, label=f"ROI {i}")
        count += 1
    widget.canvas.draw()


def plot_traces_with_peaks(widget: _GraphWidget, data: dict) -> None:
    """Plot the traces with the detected peaks."""
    ...


def plot_raster_plot(widget: _GraphWidget, data: dict) -> None:
    """Plot the raster plot for the given FOV."""
    ...


def plot_mean_amplitude(widget: _GraphWidget, data: dict) -> None:
    """Plot the mean amplitude for the given FOV."""
    ...


def plot_mean_frequency(widget: _GraphWidget, data: dict) -> None:
    """Plot the mean frequency for the given FOV."""
    ...
