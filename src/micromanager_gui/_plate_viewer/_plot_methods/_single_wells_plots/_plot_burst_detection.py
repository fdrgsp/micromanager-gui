"""Burst detection and network state analysis for spike data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_burst_detection_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.1,
    burst_threshold: float = 0.3,
    min_burst_duration: int = 3,
    smoothing_sigma: float = 2.0,
) -> None:
    """Plot burst detection and network state analysis.

    Parameters
    ----------
        widget: The widget to plot on
        data: Dictionary of ROIData objects containing spike information
        rois: List of ROI indices to include in the analysis, None for all
        spike_threshold: Threshold for spike detection (default 0.1)
        burst_threshold: Threshold for burst detection (default 0.3)
        min_burst_duration: Minimum duration of a burst in samples (default 3)
        smoothing_sigma: Sigma for Gaussian smoothing of activity (default 2.0)
    """
    widget.figure.clear()

    # Get spike trains and calculate population activity
    spike_trains, roi_names, time_axis = _get_population_spike_data(
        data, rois, spike_threshold
    )

    if spike_trains is None or len(spike_trains) < 2:
        ax = widget.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Insufficient spike data for burst analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Calculate population activity
    population_activity = np.mean(spike_trains, axis=0)

    # Smooth population activity
    smoothed_activity = gaussian_filter1d(population_activity, sigma=smoothing_sigma)

    # Detect bursts
    bursts = _detect_population_bursts(
        smoothed_activity, burst_threshold, min_burst_duration
    )

    # Create subplot layout
    fig = widget.figure
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # Plot 1: Raster plot with bursts
    ax1 = fig.add_subplot(gs[0])
    _plot_raster_with_bursts(ax1, spike_trains, roi_names, time_axis, bursts)

    # Plot 2: Population activity
    ax2 = fig.add_subplot(gs[1])
    _plot_population_activity(
        ax2, population_activity, smoothed_activity, time_axis, bursts, burst_threshold
    )

    # Plot 3: Burst statistics
    ax3 = fig.add_subplot(gs[2])
    _plot_burst_statistics(ax3, bursts, time_axis)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_population_spike_data(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.5,
) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Extract population spike data from ROI data.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all
        spike_threshold: Threshold for spike detection

    Returns
    -------
        Tuple of (spike_trains_array, roi_names, time_axis)
    """
    spike_trains: list[np.ndarray] = []
    roi_names: list[str] = []

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None, [], np.array([])

    max_length = 0
    for roi, roi_data in roi_data_dict.items():
        if int(roi) not in rois or not roi_data.active:
            continue

        if (spike_probs := roi_data.inferred_spikes) is not None:
            # Convert spike probabilities to binary spike train
            spike_train = (np.array(spike_probs) >= spike_threshold).astype(float)
            if np.sum(spike_train) > 0:  # Only include ROIs with at least one spike
                spike_trains.append(spike_train)
                roi_names.append(roi)
                max_length = max(max_length, len(spike_train))

    if len(spike_trains) < 2:
        return None, [], np.array([])

    # Pad all spike trains to same length
    padded_trains = []
    for train in spike_trains:
        if len(train) < max_length:
            padded = np.zeros(max_length)
            padded[: len(train)] = train
            padded_trains.append(padded)
        else:
            padded_trains.append(train[:max_length])

    spike_trains_array = np.array(padded_trains)

    # Create time axis (assuming 10 Hz sampling rate)
    time_axis = np.arange(max_length) / 10.0  # Convert to seconds

    return spike_trains_array, roi_names, time_axis


def _detect_population_bursts(
    population_activity: np.ndarray,
    burst_threshold: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Detect population bursts in the smoothed activity.

    Args:
        population_activity: Population activity signal
        burst_threshold: Threshold for burst detection
        min_duration: Minimum burst duration in samples

    Returns
    -------
        List of (start, end) indices for detected bursts
    """
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

    # Handle case where burst extends to end of recording
    if in_burst:
        burst_duration = len(above_threshold) - burst_start
        if burst_duration >= min_duration:
            bursts.append((burst_start, len(above_threshold)))

    return bursts


def _plot_raster_with_bursts(
    ax: plt.Axes,
    spike_trains: np.ndarray,
    roi_names: list[str],
    time_axis: np.ndarray,
    bursts: list[tuple[int, int]],
) -> None:
    """Plot raster plot with burst periods highlighted."""
    n_rois, n_samples = spike_trains.shape

    # Plot spikes for each ROI
    for i, _roi_name in enumerate(roi_names):
        spike_times = time_axis[spike_trains[i] > 0]
        if len(spike_times) > 0:
            ax.scatter(spike_times, [i] * len(spike_times), s=2, c="black", marker="|")

    # Highlight burst periods
    for burst_start, burst_end in bursts:
        t_start = time_axis[burst_start]
        t_end = (
            time_axis[burst_end - 1] if burst_end < len(time_axis) else time_axis[-1]
        )
        ax.axvspan(t_start, t_end, alpha=0.3, color="red", label="Network Burst")

    ax.set_ylabel("ROI")
    ax.set_title("Spike Raster Plot with Network Bursts")
    ax.set_yticks(range(len(roi_names)))
    ax.set_yticklabels(roi_names)
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(-0.5, len(roi_names) - 0.5)

    # Add legend (only once)
    if bursts:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")


def _plot_population_activity(
    ax: plt.Axes,
    raw_activity: np.ndarray,
    smoothed_activity: np.ndarray,
    time_axis: np.ndarray,
    bursts: list[tuple[int, int]],
    threshold: float,
) -> None:
    """Plot population activity with burst detection threshold."""
    ax.plot(
        time_axis, raw_activity, "lightgray", alpha=0.7, label="Raw Population Activity"
    )
    ax.plot(
        time_axis,
        smoothed_activity,
        "blue",
        linewidth=2,
        label="Smoothed Population Activity",
    )
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label=f"Burst Threshold ({threshold:.2f})",
    )

    # Highlight burst periods
    for burst_start, burst_end in bursts:
        t_start = time_axis[burst_start]
        t_end = (
            time_axis[burst_end - 1] if burst_end < len(time_axis) else time_axis[-1]
        )
        ax.axvspan(t_start, t_end, alpha=0.3, color="red")

    ax.set_ylabel("Population Activity")
    ax.set_title("Population Activity and Burst Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_burst_statistics(
    ax: plt.Axes,
    bursts: list[tuple[int, int]],
    time_axis: np.ndarray,
) -> None:
    """Plot burst statistics."""
    if not bursts:
        ax.text(
            0.5,
            0.5,
            "No bursts detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Burst Statistics")
        return

    # Calculate burst statistics
    burst_durations = []
    burst_intervals = []

    for i, (start, end) in enumerate(bursts):
        duration = (time_axis[end - 1] - time_axis[start]) if end > start else 0
        burst_durations.append(duration)

        if i > 0:
            prev_end = bursts[i - 1][1]
            interval = time_axis[start] - time_axis[prev_end - 1]
            burst_intervals.append(interval)

    # Create bar plot of statistics
    stats_labels = [
        "Count",
        "Avg Duration (s)",
        "Avg Interval (s)",
        "Rate (bursts/min)",
    ]

    count = len(bursts)
    avg_duration = np.mean(burst_durations) if burst_durations else 0
    avg_interval = np.mean(burst_intervals) if burst_intervals else 0

    # Calculate burst rate (bursts per minute)
    total_time = time_axis[-1] - time_axis[0]  # in seconds
    burst_rate = (count / total_time) * 60 if total_time > 0 else 0

    stats_values = [count, avg_duration, avg_interval, burst_rate]

    bars = ax.bar(
        stats_labels, stats_values, color=["skyblue", "lightgreen", "orange", "pink"]
    )

    # Add value labels on bars
    for bar, value in zip(bars, stats_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_title("Burst Statistics")
    ax.set_ylabel("Value")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _calculate_network_states(
    spike_trains: np.ndarray,
    bursts: list[tuple[int, int]],
    time_axis: np.ndarray,
) -> dict[str, float]:
    """Calculate network state metrics.

    Args:
        spike_trains: Array of spike trains (n_rois, n_samples)
        bursts: List of burst periods
        time_axis: Time axis in seconds

    Returns
    -------
        Dictionary of network state metrics
    """
    total_time = time_axis[-1] - time_axis[0]

    # Calculate burst-related metrics
    burst_count = len(bursts)

    if burst_count > 0:
        burst_durations = [
            time_axis[end - 1] - time_axis[start]
            for start, end in bursts
            if end > start
        ]
        avg_burst_duration = np.mean(burst_durations)
        total_burst_time = np.sum(burst_durations)
        burst_fraction = total_burst_time / total_time

        if burst_count > 1:
            intervals = [
                time_axis[bursts[i][0]] - time_axis[bursts[i - 1][1] - 1]
                for i in range(1, burst_count)
            ]
            avg_burst_interval = np.mean(intervals)
        else:
            avg_burst_interval = 0
    else:
        avg_burst_duration = 0
        burst_fraction = 0
        avg_burst_interval = 0

    # Calculate participation metrics
    total_spikes = np.sum(spike_trains)
    avg_firing_rate = total_spikes / (spike_trains.shape[0] * total_time)

    return {
        "burst_count": burst_count,
        "burst_rate": burst_count / (total_time / 60),  # bursts per minute
        "avg_burst_duration": avg_burst_duration,
        "avg_burst_interval": avg_burst_interval,
        "burst_fraction": burst_fraction,
        "avg_firing_rate": avg_firing_rate,
        "total_spikes": int(total_spikes),
    }
