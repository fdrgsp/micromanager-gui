from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from ._graph_widget import _GraphWidget
    from ._util import ROIData


from typing import Callable


def plot_traces(
    widget: _GraphWidget,
    data: dict,
    rois: list[int] | None,
    title: str,
    plot_func: Callable,
) -> None:
    """Helper function to plot traces with hover functionality."""
    ax = widget.figure.add_subplot(111)
    ax.set_title(title)
    ax.get_yaxis().set_visible(False)
    count = 0
    for i, key in enumerate(data):
        if rois is not None and i not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        plot_func(ax, roi_data.trace, count, i)
        count += 1
    # Adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_raw_traces(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces."""

    def plot_func(ax: plt.Axes, trace: np.ndarray, count: int, i: int) -> None:
        offset = 10
        ax.plot(np.array(trace) + count * offset, label=f"ROI {i}")

    plot_traces(widget, data, rois, f"{widget._fov} - raw traces", plot_func)


def plot_normalized_raw_traces(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces normalized to the range [0, 1]."""

    def plot_func(ax: plt.Axes, trace: np.ndarray, count: int, i: int) -> None:
        normalized_trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
        ax.plot(normalized_trace + count, label=f"ROI {i}")

    plot_traces(
        widget, data, rois, f"{widget._fov} - normalized raw traces [0, 1]", plot_func
    )


def plot_delta_f_over_f(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    # TODO: dff should be calculated in the analysis and stored in the ROIData class
    # here we will only need to plot roi_data.dff. Also use a better methodfor dff
    """Plot the delta f over f traces."""
    ...


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
