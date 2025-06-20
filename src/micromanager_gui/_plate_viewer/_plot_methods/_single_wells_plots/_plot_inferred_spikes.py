from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors
import numpy as np
from scipy.ndimage import gaussian_filter1d

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import (
    _get_spikes_over_threshold,
)

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
            label=f"Spike threshold (ROI {roi_key} - {spikes_threshold:.4f})",
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

        # Show hover for anything with ROI in the label (traces and thresholds)
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            # Extract ROI number for selection (works for both traces and thresholds)
            roi_parts = label.split("ROI ")
            if len(roi_parts) > 1:
                roi_num = roi_parts[1].split()[0] if roi_parts[1].split() else ""
                if roi_num.isdigit():
                    widget.roiSelected.emit(roi_num)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


def _plot_inferred_spikes_normalized_with_bursts(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot normalized inferred spikes with superimposed burst periods.

    This combines the normalized spike traces visualization with burst detection
    to show when network bursts occur overlaid on the individual ROI traces.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    data : dict[str, ROIData]
        Dictionary of ROI data containing spike information
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    """
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    burst_params = _get_burst_parameters(data, rois)
    if burst_params is None:
        LOGGER.warning("Burst parameters not found in ROI data.")
        return
    burst_threshold, min_burst_duration, smoothing_sigma = burst_params

    # Get all traces and compute normalization parameters
    all_values = []
    valid_rois = []
    roi_traces = {}

    for roi_key, roi_data in data.items():
        if rois is not None:
            try:
                roi_id = int(roi_key)
                if roi_id not in rois:
                    continue
            except ValueError:
                continue

        if not roi_data.inferred_spikes:
            continue

        if not roi_data.active:
            continue

        if trace := _get_spikes_over_threshold(roi_data):
            all_values.extend(trace)
            valid_rois.append(roi_key)
            roi_traces[roi_key] = trace

    if not all_values:
        LOGGER.warning(
            "No valid spike data found for the specified ROIs. "
            "Ensure that the ROIs have inferred spikes data."
        )
        return

    # Compute normalization percentiles
    percentiles = np.percentile(all_values, [5, 100])
    p1, p2 = float(percentiles[0]), float(percentiles[1])

    # Plot normalized traces
    count = 0
    rois_rec_time = []
    last_trace = None

    for roi_key in valid_rois:
        roi_data = data[roi_key]
        trace = roi_traces[roi_key]

        if (ttime := roi_data.total_recording_time_sec) is not None:
            rois_rec_time.append(ttime)

        offset = count * 1.1  # vertical offset
        normalized_trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(normalized_trace, label=f"ROI {roi_key}", alpha=0.7, color="black")

        last_trace = trace
        count += 1

    # Detect and overlay bursts
    if len(valid_rois) > 1:  # Only detect bursts if we have multiple ROIs
        bursts = _detect_bursts_from_traces(
            roi_traces, burst_threshold / 100, min_burst_duration, smoothing_sigma
        )
        _overlay_burst_periods(ax, bursts, count)

    # Set labels and formatting
    ax.set_title(
        "Normalized Inferred Spikes with Network Bursts\n" "(Thresholded Spike Data)"
    )
    ax.set_ylabel("ROIs")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Update time axis
    _update_time_axis(ax, rois_rec_time, last_trace)

    # Add hover functionality
    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_burst_parameters(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
) -> tuple[float, int, float] | None:
    """Get burst detection parameters from ROIData."""
    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]
    # use only the first roi since the burst parameters are the same for all ROIs
    roi_key = str(rois[0]) if rois else None
    if roi_key is None or roi_key not in roi_data_dict:
        LOGGER.warning("No valid ROIs found for burst parameter extraction.")
        return None
    roi_data = roi_data_dict[roi_key]
    burst_threshold = roi_data.spikes_burst_threshold
    burst_min_duration = roi_data.spikes_burst_min_duration
    burst_gaussian_sigma = roi_data.spikes_burst_gaussian_sigma
    # if any is NOne, return None
    if (
        burst_threshold is None
        or burst_min_duration is None
        or burst_gaussian_sigma is None
    ):
        LOGGER.warning("Burst parameters not set in ROI data.")
        return None
    return (
        burst_threshold,
        burst_min_duration,
        burst_gaussian_sigma,
    )


def _detect_bursts_from_traces(
    roi_traces: dict[str, list[float]],
    burst_threshold: float,
    min_duration: int,
    smoothing_sigma: float,
) -> list[tuple[int, int]]:
    """Detect bursts from multiple ROI traces.

    Parameters
    ----------
    roi_traces : dict[str, list[float]]
        Dictionary mapping ROI keys to their spike traces
    burst_threshold : float
        Threshold for burst detection (0-1)
    min_duration : int
        Minimum burst duration in samples
    smoothing_sigma : float
        Sigma for Gaussian smoothing

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) indices for detected bursts
    """
    if not roi_traces:
        return []

    # Get the length of traces (assuming all are same length)
    trace_length = len(next(iter(roi_traces.values())))

    # Create binary spike matrix
    spike_matrix = np.zeros((len(roi_traces), trace_length))

    for i, (_, trace) in enumerate(roi_traces.items()):
        # Convert trace to binary spikes (above some threshold)
        trace_array = np.array(trace)
        if len(trace_array) > 0:
            # Use 75th percentile as spike threshold
            threshold = np.percentile(trace_array, 75)
            spike_matrix[i, :] = trace_array > threshold

    # Calculate population activity (fraction of ROIs spiking at each time point)
    population_activity = np.mean(spike_matrix, axis=0)

    # Smooth the population activity
    if smoothing_sigma > 0:
        population_activity = gaussian_filter1d(
            population_activity, sigma=smoothing_sigma
        )

    # Detect bursts
    return _detect_population_bursts(population_activity, burst_threshold, min_duration)


def _detect_population_bursts(
    population_activity: np.ndarray,
    burst_threshold: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Detect population bursts in the smoothed activity."""
    # Find regions above threshold
    above_threshold = population_activity > burst_threshold

    # Find start and end points of bursts
    bursts = []
    in_burst = False
    burst_start = 0

    for i, is_active in enumerate(above_threshold):
        if is_active and not in_burst:
            # Start of new burst
            burst_start = i
            in_burst = True
        elif not is_active and in_burst:
            # End of burst
            burst_duration = i - burst_start
            if burst_duration >= min_duration:
                bursts.append((burst_start, i))
            in_burst = False

    # Handle case where burst extends to the end
    if in_burst and (len(population_activity) - burst_start) >= min_duration:
        bursts.append((burst_start, len(population_activity)))

    return bursts


def _overlay_burst_periods(
    ax: Axes, bursts: list[tuple[int, int]], num_rois: int
) -> None:
    """Overlay burst periods as shaded regions on the plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    bursts : list[tuple[int, int]]
        List of (start, end) indices for bursts
    num_rois : int
        Number of ROIs plotted (for determining y-axis span)
    """
    if not bursts:
        return

    for i, (start, end) in enumerate(bursts):
        label = "Network Burst" if i == 0 else ""
        ax.axvspan(start, end, alpha=0.2, color="green", label=label)
