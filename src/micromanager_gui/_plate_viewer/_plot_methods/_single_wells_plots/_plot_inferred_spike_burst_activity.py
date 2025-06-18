from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d

from micromanager_gui._plate_viewer._util import (
    BURST_GAUSSIAN_SIGMA,
    BURST_MIN_DURATION,
    BURST_THRESHOLD,
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_MIN_BURST_DURATION,
    SETTINGS_PATH,
    _get_spikes_over_threshold,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_inferred_spike_burst_activity(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot burst detection and network state analysis for inferred spikes.

    This function analyzes population-level spike activity to detect synchronized
    burst events and display comprehensive burst statistics.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    data : dict[str, ROIData]
        Dictionary of ROI data containing spike information
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    burst_threshold : float
        Threshold for detecting network bursts in population activity (default 0.3).
        This is applied to the smoothed population activity (mean spike activity
        across all ROIs), where 0.0 = no ROIs spiking and 1.0 = all ROIs spiking
        simultaneously. A value of 0.3 detects bursts when ≥30% of ROIs spike together.
    min_burst_duration : int
        Minimum duration for a burst in samples (default 3)
    smoothing_sigma : float
        Sigma for Gaussian smoothing of population activity (default 2.0)
    """
    widget.figure.clear()

    # get parameters form the analysis path settings.json file
    burst_threshold: float = DEFAULT_BURST_THRESHOLD
    min_burst_duration: int = DEFAULT_MIN_BURST_DURATION
    smoothing_sigma: float = DEFAULT_BURST_GAUSS_SIGMA
    if analysis_path := widget._plate_viewer.analysis_path:
        settings_json_file = Path(analysis_path) / SETTINGS_PATH
        if settings_json_file.exists():
            with open(settings_json_file) as f:
                settings = cast(dict, json.load(f))
                burst_threshold = float(
                    settings.get(BURST_THRESHOLD, DEFAULT_BURST_THRESHOLD)
                )
                min_burst_duration = int(
                    settings.get(BURST_MIN_DURATION, DEFAULT_MIN_BURST_DURATION)
                )
                smoothing_sigma = float(
                    settings.get(BURST_GAUSSIAN_SIGMA, DEFAULT_BURST_GAUSS_SIGMA)
                )
    # or from the widget's analysis widget
    else:
        values = cast(
            tuple[float, int, float],
            tuple(widget._plate_viewer._analysis_wdg._burst_wdg.value().values()),
        )
        burst_threshold, min_burst_duration, smoothing_sigma = values

    # Get spike trains and calculate population activity
    spike_trains, roi_names, time_axis = _get_population_spike_data(data, rois)

    if spike_trains is None or len(spike_trains) < 2:
        ax = widget.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Insufficient spike data for burst analysis\n"
            "Need at least 2 active ROIs with spikes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Burst Activity Analysis")
        widget.canvas.draw()
        return

    # Calculate population activity
    population_activity = np.mean(spike_trains, axis=0)

    # Smooth population activity for burst detection
    smoothed_activity = gaussian_filter1d(population_activity, sigma=smoothing_sigma)

    # Detect bursts
    bursts = _detect_population_bursts(
        smoothed_activity, burst_threshold / 100, min_burst_duration
    )

    # Create single plot layout
    fig = widget.figure
    ax = fig.add_subplot(111)

    # Plot population activity with burst detection
    _plot_population_activity(
        ax,
        population_activity,
        smoothed_activity,
        time_axis,
        bursts,
        burst_threshold / 100,
    )

    # Add statistics legend below the plot
    _add_burst_statistics_legend(ax, bursts, time_axis)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_population_spike_data(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Extract population spike data from ROI data.

    Parameters
    ----------
    roi_data_dict : dict[str, ROIData]
        Dictionary of ROI data
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs

    Returns
    -------
    tuple[np.ndarray | None, list[str], np.ndarray]
        Tuple of (spike_trains_array, roi_names, time_axis)
    """
    spike_trains: list[np.ndarray] = []
    roi_names: list[str] = []

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None, [], np.array([])

    max_length = 0
    rois_rec_time: list[float] = []

    for roi in rois:
        roi_key = str(roi)
        if roi_key not in roi_data_dict:
            continue

        roi_data = roi_data_dict[roi_key]
        if not roi_data.active:
            continue

        # Get thresholded spike data
        spike_probs = _get_spikes_over_threshold(roi_data)
        if spike_probs is None:
            continue

        # Convert spike probabilities to binary spike train
        # _get_spikes_over_threshold already returns thresholded data
        spike_train = (np.array(spike_probs) > 0.0).astype(float)
        if np.sum(spike_train) > 0:  # Only include ROIs with at least one spike
            spike_trains.append(spike_train)
            roi_names.append(roi_key)
            max_length = max(max_length, len(spike_train))

            # Store recording time for time axis calculation
            if roi_data.total_recording_time_sec is not None:
                rois_rec_time.append(roi_data.total_recording_time_sec)

    if len(spike_trains) < 2:
        return None, [], np.array([])

    # Pad all spike trains to same length
    padded_trains: list[np.ndarray] = []
    for train in spike_trains:
        if len(train) < max_length:
            padded = np.zeros(max_length, dtype=np.float64)
            padded[: len(train)] = train
            padded_trains.append(padded)
        else:
            truncated = np.array(train[:max_length], dtype=np.float64)
            padded_trains.append(truncated)

    spike_trains_array = np.array(padded_trains)

    # Create time axis using recording time if available
    if rois_rec_time:
        avg_rec_time = np.mean(rois_rec_time)
        time_axis = np.linspace(0, avg_rec_time, max_length)
    else:
        # Fallback to frame-based time axis (assuming 10 Hz sampling rate)
        time_axis = np.arange(max_length) / 10.0

    return spike_trains_array, roi_names, time_axis


def _detect_population_bursts(
    population_activity: np.ndarray,
    burst_threshold: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Detect population bursts in the smoothed activity.

    Parameters
    ----------
    population_activity : np.ndarray
        Population activity signal
    burst_threshold : float
        Threshold for burst detection
    min_duration : int
        Minimum burst duration in samples

    Returns
    -------
    list[tuple[int, int]]
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


def _plot_population_activity(
    ax: Axes,
    raw_activity: np.ndarray,
    smoothed_activity: np.ndarray,
    time_axis: np.ndarray,
    bursts: list[tuple[int, int]],
    threshold: float,
) -> None:
    """Plot population activity with burst detection threshold.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    raw_activity : np.ndarray
        Raw population activity
    smoothed_activity : np.ndarray
        Smoothed population activity
    time_axis : np.ndarray
        Time axis in seconds
    bursts : list[tuple[int, int]]
        List of burst periods
    threshold : float
        Burst detection threshold
    """
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
    ax.set_xlabel("Time (s)")
    ax.set_title("Population Activity and Burst Detection (Thresholded Spike Data)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _add_burst_statistics_legend(
    ax: Axes,
    bursts: list[tuple[int, int]],
    time_axis: np.ndarray,
) -> None:
    """Add a legend below the plot showing burst statistics.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add legend to
    bursts : list[tuple[int, int]]
        List of burst periods
    time_axis : np.ndarray
        Time axis in seconds
    """
    if not bursts:
        # Add a simple legend indicating no bursts
        ax.text(
            0.5,
            0.95,
            "Burst Statistics: No bursts detected",
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray", "alpha": 0.8},
        )
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

    # Calculate statistics
    count = len(bursts)
    avg_duration = np.mean(burst_durations) if burst_durations else 0
    avg_interval = np.mean(burst_intervals) if burst_intervals else 0

    # Calculate burst rate (bursts per minute)
    total_time = time_axis[-1] - time_axis[0]  # in seconds
    burst_rate = (count / total_time) * 60 if total_time > 0 else 0

    # Create statistics text
    stats_text = (
        f"Count: {count}, "
        f"Avg Duration: {avg_duration:.2f}s, "
        f"Avg Interval: {avg_interval:.2f}s, "
        f"Rate: {burst_rate:.2f} bursts/min"
    )

    # Add text box below the plot, under the x-axis label
    ax.text(
        0.5,
        -0.22,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.8},
        wrap=True,
    )


def _calculate_network_burst_metrics(
    spike_trains: np.ndarray,
    bursts: list[tuple[int, int]],
    time_axis: np.ndarray,
) -> dict[str, float]:
    """Calculate comprehensive network burst metrics.

    Parameters
    ----------
    spike_trains : np.ndarray
        Array of spike trains (n_rois, n_samples)
    bursts : list[tuple[int, int]]
        List of burst periods
    time_axis : np.ndarray
        Time axis in seconds

    Returns
    -------
    dict[str, float]
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
        "burst_rate": float(burst_count / (total_time / 60)),  # bursts per minute
        "avg_burst_duration": float(avg_burst_duration),
        "avg_burst_interval": float(avg_burst_interval),
        "burst_fraction": float(burst_fraction),
        "avg_firing_rate": float(avg_firing_rate),
        "total_spikes": float(total_spikes),
    }
