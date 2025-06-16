from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors
import numpy as np

from micromanager_gui._plate_viewer._util import _get_spikes_over_threshold

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_inferred_spikes(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    raw: bool = False,
    normalize: bool = False,
    active_only: bool = False,
    dec_dff: bool = False,
    thresholds: bool = False,
) -> None:
    """Plot inferred spikes data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # show peaks thresholds only if only 1 roi is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

    # compute percentiles for normalization if needed
    p1 = p2 = 0.0
    if normalize:
        all_values = []
        for roi_key, roi_data in data.items():
            if rois is not None:
                try:
                    roi_id = int(roi_key)
                    if roi_id not in rois:
                        continue
                except ValueError:
                    # Skip non-numeric ROI keys when rois filter is specified
                    continue
            if the_spikes := _get_spikes_over_threshold(roi_data):
                all_values.extend(the_spikes)

        if all_values:
            percentiles = np.percentile(all_values, [5, 100])
            p1, p2 = float(percentiles[0]), float(percentiles[1])
        else:
            p1, p2 = 0.0, 1.0

    count = 0
    rois_rec_time: list[float] = []
    last_trace: list[float] | None = None

    for roi_key, roi_data in data.items():
        if rois is not None:
            try:
                roi_id = int(roi_key)
                if roi_id not in rois:
                    continue
            except ValueError:
                # Skip non-numeric ROI keys when rois filter is specified
                continue

        if not roi_data.inferred_spikes:
            continue

        if (ttime := roi_data.total_recording_time_sec) is not None:
            rois_rec_time.append(ttime)

        # plot only active neurons if asked to plot active only
        if active_only and not roi_data.active:
            continue
        _plot_trace(
            ax,
            roi_key,
            _get_spikes_over_threshold(roi_data, raw),
            normalize,
            count,
            p1,
            p2,
            thresholds,
            roi_data.inferred_spikes_threshold,
        )
        if dec_dff and roi_data.dec_dff:
            _plot_trace(ax, roi_key, roi_data.dec_dff, normalize, count, p1, p2)
        last_trace = roi_data.inferred_spikes
        count += 1

    _set_graph_title_and_labels(ax, normalize, raw)

    _update_time_axis(ax, rois_rec_time, last_trace)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float] | None,
    normalize: bool,
    count: int,
    p1: float,
    p2: float,
    thresholds: bool = False,
    spikes_threshold: float | None = None,
) -> None:
    """Plot inferred spikes trace with optional percentile-based normalization."""
    if trace is None or not trace:
        return
    if normalize:
        offset = count * 1.1  # vertical offset
        spike_trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(spike_trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    # Add horizontal line for spike detection threshold
    if thresholds and spikes_threshold is not None and spikes_threshold > 0.0:
        ax.axhline(
            y=spikes_threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"Spike threshold (ROI {roi_key})",
        )


def _normalize_trace_percentile(trace: list[float], p1: float, p2: float) -> np.ndarray:
    """Normalize a trace using p1th-p2th percentile, clipped to [0, 1]."""
    tr = np.array(trace)
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels(ax: Axes, normalize: bool, raw: bool) -> None:
    """Set axis labels based on the plotted data."""
    title = ("Normalized Inferred Spikes" if normalize else "Inferred Spikes") + (
        " (Raw)" if raw else " (Thresholded Spike Data)"
    )
    y_lbl = "ROIs" if normalize else "Inferred Spikes (magnitude)"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update the time axis based on recording time."""
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
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for other elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi = label.split(" ")[1] if len(label.split(" ")) > 1 else ""
            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
