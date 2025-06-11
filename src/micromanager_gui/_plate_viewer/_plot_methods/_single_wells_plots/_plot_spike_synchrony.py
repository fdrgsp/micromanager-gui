"""Spike-based synchrony analysis for network analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_spike_synchrony_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.1,
    time_window: float = 0.1,
) -> None:
    """Plot spike-based synchrony analysis.

    Args:
        widget: The graph widget to plot on
        data: Dictionary of ROI data
        rois: List of ROI indices to analyze, None for all active ROIs
        spike_threshold: Threshold for considering a spike event (0.0-1.0)
        time_window: Time window in seconds for synchrony detection
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    spike_trains = _get_spike_trains_from_rois(data, rois, spike_threshold)
    if spike_trains is None or len(spike_trains) < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient spike data for synchrony analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    synchrony_matrix = _calculate_spike_synchrony_matrix(spike_trains, time_window)

    if synchrony_matrix is None:
        ax.text(
            0.5,
            0.5,
            "Unable to calculate spike synchrony",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Calculate global synchrony metric
    upper_tri_indices = np.triu_indices_from(synchrony_matrix, k=1)
    global_synchrony = np.median(synchrony_matrix[upper_tri_indices])

    title = (
        f"Spike-based Synchrony (threshold={spike_threshold:.1f}, "
        f"window={time_window:.1f}s)\nGlobal Synchrony: {global_synchrony:.3f}"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Spike Synchrony Index")

    ax.set_title(title)
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_box_aspect(1)

    active_rois = list(spike_trains.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)
    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_spike_trains_from_rois(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.5,
) -> dict[str, np.ndarray] | None:
    """Extract spike trains from ROI data.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all
        spike_threshold: Threshold for spike detection

    Returns
    -------
        Dictionary mapping ROI names to binary spike arrays
    """
    spike_trains: dict[str, np.ndarray] = {}

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None

    for roi, roi_data in roi_data_dict.items():
        if int(roi) not in rois or not roi_data.active:
            continue

        if (spike_probs := roi_data.inferred_spikes) is not None:
            # Convert spike probabilities to binary spike train
            spike_train = np.array(spike_probs) >= spike_threshold
            if np.sum(spike_train) > 0:  # Only include ROIs with at least one spike
                spike_trains[roi] = spike_train

    return spike_trains if len(spike_trains) >= 2 else None


def _calculate_spike_synchrony_matrix(
    spike_trains: dict[str, np.ndarray],
    time_window: float,
    sampling_rate: float = 10.0,  # Hz, typical frame rate
) -> np.ndarray | None:
    """Calculate pairwise spike synchrony matrix.

    Args:
        spike_trains: Dictionary of binary spike trains
        time_window: Time window for synchrony detection (seconds)
        sampling_rate: Sampling rate in Hz

    Returns
    -------
        Square matrix of synchrony values
    """
    roi_names = list(spike_trains.keys())
    n_rois = len(roi_names)

    if n_rois < 2:
        return None

    synchrony_matrix = np.zeros((n_rois, n_rois))

    # Convert time window to sample window
    sample_window = int(time_window * sampling_rate)

    for i, roi_i in enumerate(roi_names):
        for j, roi_j in enumerate(roi_names):
            if i == j:
                synchrony_matrix[i, j] = 1.0
            else:
                sync_value = _calculate_pairwise_spike_synchrony(
                    spike_trains[roi_i], spike_trains[roi_j], sample_window
                )
                synchrony_matrix[i, j] = sync_value

    return synchrony_matrix


def _calculate_pairwise_spike_synchrony(
    spikes1: np.ndarray,
    spikes2: np.ndarray,
    window: int,
) -> float:
    """Calculate synchrony between two spike trains.

    Uses coincidence detection within a time window.

    Args:
        spikes1: Binary spike train 1
        spikes2: Binary spike train 2
        window: Sample window for coincidence detection

    Returns
    -------
        Synchrony value between 0 and 1
    """
    if len(spikes1) != len(spikes2):
        min_len = min(len(spikes1), len(spikes2))
        spikes1 = spikes1[:min_len]
        spikes2 = spikes2[:min_len]

    spike_times1 = np.where(spikes1)[0]
    spike_times2 = np.where(spikes2)[0]

    if len(spike_times1) == 0 or len(spike_times2) == 0:
        return 0.0

    # Count coincident spikes
    coincidences = 0
    for t1 in spike_times1:
        # Check if any spike in train 2 occurs within the window
        time_diffs = np.abs(spike_times2 - t1)
        if np.any(time_diffs <= window):
            coincidences += 1

    # Normalize by the total number of spikes
    total_spikes = len(spike_times1) + len(spike_times2)
    if total_spikes == 0:
        return 0.0

    # Return normalized synchrony metric
    return (2 * coincidences) / total_spikes


def _add_hover_functionality(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[str],
    synchrony_matrix: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            sel.annotation.set(
                text=f"ROI {roi_x} ↔ ROI {roi_y}\nSpike Synchrony: {synchrony_matrix[y, x]:.3f}",
                fontsize=8,
                color="black",
            )
            if roi_x.isdigit() and roi_y.isdigit():
                widget.roiSelected.emit([roi_x, roi_y])
