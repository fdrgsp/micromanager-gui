from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ._graph_widget import _GraphWidget
    from ._util import ROIData


def plot_raw_traces(widget: _GraphWidget, data: dict) -> None:
    """Plot the raw traces."""
    # maybe use  ax = widget.figure.add_axes([0.1, 0.1, 0.8, 0.8]) instead of subplot
    ax = widget.figure.add_subplot(111)
    # # set title
    ax.set_title(f"{widget._fov} - raw traces")
    for key in data:
        roi_data = cast("ROIData", data[key])
        ax.plot(roi_data.trace, label=f"trace {key}")
    # Draw the plot
    widget.canvas.draw()


def plot_delta_f_over_f(widget: _GraphWidget, data: dict) -> None:
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
