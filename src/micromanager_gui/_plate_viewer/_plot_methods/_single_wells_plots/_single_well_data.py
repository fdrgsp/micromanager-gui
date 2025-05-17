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


def _plot_single_well_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    dff: bool = False,
    dec: bool = False,
    normalize: bool = False,
    with_peaks: bool = False,
    amp: bool = False,
    freq: bool = False,
    iei: bool = False,
) -> None:
    """Plot various types of traces."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # compute global min/max if normalization is requested
    p5 = p100 = 0.0
    if normalize:
        all_values = []
        for roi_key, roi_data in data.items():
            if rois is not None and int(roi_key) not in rois:
                continue
            trace = _get_trace(roi_data, dff, dec)
            if trace is not None:
                all_values.extend(trace)
        p5, p100 = (
            np.percentile(all_values, [5, 100]) if all_values else (0.0, 1.0)
        )


    rois_rec_time: list[float] = []
    count = 0

    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        trace = _get_trace(roi_data, dff, dec)
        if trace is None:
            continue

        if (ttime := roi_data.total_recording_time_in_sec) is not None:
            rois_rec_time.append(ttime)

        if amp or freq or iei:
            _plot_metrics(ax, roi_key, roi_data, amp, freq, iei)
        else:
            # plot only active neurons if asked to plot peaks
            if with_peaks and not roi_data.active:
                continue
            _plot_trace(
                ax,
                roi_key,
                trace,
                normalize,
                with_peaks,
                roi_data,
                count,
                p5,
                p100,
            )
            count += COUNT_INCREMENT

    title = [
        "Normalized Traces [global min, global max]" if normalize else "",
        "Peaks" if with_peaks else "",
    ]

    ax.set_title(" - ".join(filter(None, title)))
    _set_axis_labels(ax, amp, freq, iei, dff, dec)
    if not (amp or freq or iei):
        _update_time_axis(ax, rois_rec_time, trace)
    widget.figure.tight_layout()

    _add_hover_functionality(ax, widget)
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes, roi_key: str, roi_data: ROIData, amp: bool, freq: bool, iei: bool
) -> None:
    """Plot amplitude, frequency, or inter-event intervals."""
    if amp and freq:
        if (
            roi_data.peaks_amplitudes_dec_dff is None
            or roi_data.dec_dff_frequency is None
        ):
            return
        ax.plot(
            roi_data.peaks_amplitudes_dec_dff,
            [roi_data.dec_dff_frequency] * len(roi_data.peaks_amplitudes_dec_dff),
            "o",
            label=f"ROI {roi_key}",
        )
    elif amp:
        if roi_data.peaks_amplitudes_dec_dff is None:
            return
        ax.plot(
            [int(roi_key)] * len(roi_data.peaks_amplitudes_dec_dff),
            roi_data.peaks_amplitudes_dec_dff,
            "o",
            label=f"ROI {roi_key}",
        )
    elif freq:
        if roi_data.dec_dff_frequency is None:
            return
        ax.plot(int(roi_key), roi_data.dec_dff_frequency, "o", label=f"ROI {roi_key}")
    elif iei and roi_data.iei:
        ax.plot(
            [int(roi_key)] * len(roi_data.iei),
            roi_data.iei,
            "o",
            label=f"ROI {roi_key}",
        )


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float] | np.ndarray,
    normalize: bool,
    with_peaks: bool,
    roi_data: ROIData,
    count: int,
    p5: float,
    p100: float,
) -> None:
    """Plot trace data with optional percentile-based normalization and peaks."""
    offset = count * 1.1  # vertical offset

    if normalize:
        trace = _normalize_trace_percentile(trace, p5, p100) + offset
        ax.plot(trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    if with_peaks and roi_data.peaks_dec_dff:
        peaks_indices = [int(p) for p in roi_data.peaks_dec_dff]
        ax.plot(peaks_indices, np.array(trace)[peaks_indices], "x")


def _set_axis_labels(
    ax: Axes, amp: bool, freq: bool, iei: bool, dff: bool, dec: bool
) -> None:
    """Set axis labels based on the plotted data."""
    if amp and freq:
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Frequency (Hz)")
    elif amp:
        ax.set_xlabel("ROIs")
        ax.set_ylabel("Amplitude")
    elif freq:
        ax.set_xlabel("ROIs")
        ax.set_ylabel("Frequency (Hz)")
    elif iei:
        ax.set_xlabel("ROIs")
        ax.set_ylabel("Inter-event intervals (sec)")
    else:
        ax.set_ylabel(
            "dF/F" if dff else "Deconvolved dF/F" if dec else "Fluorescence Intensity"
        )


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        if sel.artist.get_label():
            roi = cast(str, sel.artist.get_label().split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)


def _get_trace(roi_data: ROIData, dff: bool, dec: bool) -> list[float] | None:
    """Get the appropriate trace based on the flags."""
    if dff and dec:
        return None
    if dff:
        return roi_data.dff
    if dec:
        return roi_data.dec_dff
    return roi_data.raw_trace


def _normalize_trace_percentile(
    trace: list[float] | np.ndarray, p5: float, p100: float
) -> np.ndarray:
    """Normalize a trace using 5th-100th percentile, clipped to [0, 1]."""
    tr = np.array(trace) if isinstance(trace, list) else trace
    denom = p100 - p5
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p5) / denom
    return np.clip(normalized, 0, 1)


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
