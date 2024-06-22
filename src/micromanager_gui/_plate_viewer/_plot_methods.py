from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
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
    # ax.get_yaxis().set_visible(False)
    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.raw_trace
        if trace is None:
            continue
        ax.plot(trace, label=f"ROI {key}")
        count += 1

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_raw_traces_photobleach_corrected(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces with photobleach correction."""
    ax = widget.figure.add_subplot(111)
    title = "Raw Traces with Photobleach Correction"
    ax.set_title(title)
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.bleach_corrected_traces
        if trace is None:
            continue
        ax.plot(trace, label=f"ROI {key}")

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_raw_traces_with_peaks(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces with detected peaks."""
    ax = widget.figure.add_subplot(111)
    title = "Raw Traces with Peaks"
    ax.set_title(title)
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.raw_trace
        peaks = roi_data.peaks
        if trace is None or peaks is None:
            continue
        pks = [pk.peak for pk in peaks if pk.peak is not None]
        ax.plot(trace, label=f"ROI {key}")
        ax.plot(pks, np.array(trace)[pks], "x", label=f"Peaks ROI {key}")

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_raw_traces_photobleach_corrected_with_peaks(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces with photobleach correction and detected peaks."""
    ax = widget.figure.add_subplot(111)
    title = "Raw Traces with Photobleach Correction and Peaks"
    ax.set_title(title)
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.bleach_corrected_traces
        peaks = roi_data.peaks
        if trace is None or peaks is None:
            continue
        pks = [pk.peak for pk in peaks if pk.peak is not None]
        ax.plot(trace, label=f"ROI {key}")
        ax.plot(pks, np.array(trace)[pks], "x", label=f"Peaks ROI {key}")

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_normalized_traces(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the raw traces normalized to the range [0, 1]."""
    ax = widget.figure.add_subplot(111)
    title = "Normalized Traces [0, 1]"
    ax.set_title(title)
    # ax.get_yaxis().set_visible(False)
    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.raw_trace
        if trace is None:
            continue
        tr = np.array(trace)
        normalized_trace = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
        ax.plot(normalized_trace + count, label=f"ROI {key}")
        count += 1

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_normalized_traces_photobleach_corrected(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot raw traces with photobleach correction normalized to the range [0, 1]."""
    ax = widget.figure.add_subplot(111)
    title = "Normalized Traces with Photobleach Correction [0, 1]"
    ax.set_title(title)
    # ax.get_yaxis().set_visible(False)
    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.bleach_corrected_traces
        if trace is None:
            continue
        tr = np.array(trace)
        normalized_trace = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
        ax.plot(normalized_trace + count, label=f"ROI {key}")
        count += 1

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_normalized_traces_with_peaks(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the normalized traces with detected peaks."""
    ax = widget.figure.add_subplot(111)
    title = "Normalized Traces with Peaks [0, 1]"
    ax.set_title(title)
    # ax.get_yaxis().set_visible(False)
    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.raw_trace
        peaks = roi_data.peaks
        if trace is None or peaks is None:
            continue
        pks = [pk.peak for pk in peaks if pk.peak is not None]
        count += 1
        tr = np.array(trace)
        normalized_trace = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
        ax.plot(normalized_trace + count, label=f"ROI {key}")
        ax.plot(
            pks,
            np.array(normalized_trace)[pks] + count,
            "x",
            label=f"Peaks ROI {key}",
        )

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_normalized_traces_photobleach_corrected_with_peaks(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the normalized traces with photobleach correction and detected peaks."""
    ax = widget.figure.add_subplot(111)
    title = "Normalized Traces with Photobleach Correction [0, 1]"
    ax.set_title(title)
    # ax.get_yaxis().set_visible(False)
    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue
        roi_data = cast("ROIData", data[key])
        trace = roi_data.bleach_corrected_traces
        peaks = roi_data.peaks
        if trace is None or peaks is None:
            continue
        pks = [pk.peak for pk in peaks if pk.peak is not None]
        tr = np.array(trace)
        normalized_trace = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
        ax.plot(normalized_trace + count, label=f"ROI {key}")
        ax.plot(
            pks,
            np.array(normalized_trace)[pks] + count,
            "x",
            label=f"Peaks ROI {key}",
        )
        count += 1

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_traces_used_for_bleach_correction(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the traces used for photobleach correction."""
    ax = widget.figure.add_subplot(111)
    title = "Traces used for Photobleach Correction"
    ax.set_title(title)
    # ax.get_yaxis().set_visible(False)
    traces_roi_id = data[next(iter(data.keys()))].traces_for_bleach_correction
    if traces_roi_id is None:
        return
    for roi_key in traces_roi_id:
        if rois is not None and int(roi_key) not in rois:
            continue
        roi_data = cast("ROIData", data[roi_key])
        trace = roi_data.raw_trace
        if trace is None:
            continue
        ax.plot(trace, label=f"ROI {roi_key}")

    # adding hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()


def plot_photobleaching_fitted_curve(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    """Plot the photobleaching fitted curve used for photobleach correction."""
    ax = widget.figure.add_subplot(111)
    title = "Photobleaching Fitted Curve"
    ax.set_title(title)
    # ax.get_yaxis().set_visible(False)
    curve = data[next(iter(data.keys()))].photobleaching_fitted_curve
    if curve is None:
        return
    ax.plot(curve, label="Fitted Curve")

    widget.canvas.draw()


def plot_delta_f_over_f(
    widget: _GraphWidget, data: dict, rois: list[int] | None = None
) -> None:
    # TODO: dff should be calculated in the analysis and stored in the ROIData class
    # here we will only need to plot roi_data.dff. Also use a better methodfor dff
    """Plot the delta f over f traces."""
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
