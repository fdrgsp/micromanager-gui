from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


COUNT_INCREMENT = 1
P1 = 5
P2 = 100


def _plot_traces_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    dff: bool = False,
    dec: bool = False,
    normalize: bool = False,
    with_peaks: bool = False,
    active_only: bool = False,
) -> None:
    """Plot traces data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # compute nth and nth percentiles globally
    p1 = p2 = 0.0
    if normalize:
        all_values = []
        for roi_key, roi_data in data.items():
            if rois is not None and int(roi_key) not in rois:
                continue
            trace = _get_trace(roi_data, dff, dec)
            if trace:
                all_values.extend(trace)
        if all_values:
            percentiles = np.percentile(all_values, [P1, P2])
            p1, p2 = float(percentiles[0]), float(percentiles[1])
        else:
            p1, p2 = 0.0, 1.0

    count = 0
    rois_rec_time: list[float] = []
    for roi_key, roi_data in data.items():
        trace = _get_trace(roi_data, dff, dec)

        if (rois is not None and int(roi_key) not in rois) or not trace:
            continue

        if (ttime := roi_data.total_recording_time_in_sec) is not None:
            rois_rec_time.append(ttime)

        # plot only active neurons if asked to plot peaks or active only
        if (with_peaks or active_only) and not roi_data.active:
            continue

        _plot_trace(ax, roi_key, trace, normalize, with_peaks, roi_data, count, p1, p2)
        count += COUNT_INCREMENT

    _set_graph_title_and_labels(ax, dff, dec, normalize, with_peaks)

    _update_time_axis(ax, rois_rec_time, trace)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_trace(roi_data: ROIData, dff: bool, dec: bool) -> list[float] | None:
    """Get the appropriate trace based on the flags."""
    data = roi_data.dff if dff else roi_data.dec_dff if dec else roi_data.raw_trace
    return data or None


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float] | np.ndarray,
    normalize: bool,
    with_peaks: bool,
    roi_data: ROIData,
    count: int,
    p1: float,
    p2: float,
) -> None:
    """Plot trace data with optional percentile-based normalization and peaks."""
    offset = count * 1.1  # vertical offset

    if normalize:
        trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    if with_peaks and roi_data.peaks_dec_dff:
        peaks_indices = [int(p) for p in roi_data.peaks_dec_dff]
        ax.plot(peaks_indices, np.array(trace)[peaks_indices], "x")


def _normalize_trace_percentile(
    trace: list[float] | np.ndarray, p1: float, p2: float
) -> np.ndarray:
    """Normalize a trace using p1th-p2th percentile, clipped to [0, 1]."""
    tr = np.array(trace) if isinstance(trace, list) else trace
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels(
    ax: Axes,
    dff: bool,
    dec: bool,
    normalize: bool,
    with_peaks: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    if dff:
        title = "Normalized Traces (ΔF/F)" if normalize else "Traces (ΔF/F)"
        y_lbl = "ROIs" if normalize else "ΔF/F"
    elif dec:
        title = (
            "Normalized Traces (Deconvolved ΔF/F)"
            if normalize
            else "Traces (Deconvolved ΔF/F)"
        )
        y_lbl = "ROIs" if normalize else "Deconvolved ΔF/F"
    else:
        title = "Normalized Traces" if normalize else "Raw Traces"
        y_lbl = "ROIs" if normalize else "Fluorescence Intensity"
    if with_peaks:
        title += " with Peaks"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    # get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # get total number of frames from the trace
    total_frames = len(trace) if trace is not None else 1
    # compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        if (lbl := sel.artist.get_label()) and "ROI" in lbl:
            roi = cast(str, sel.artist.get_label().split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)
