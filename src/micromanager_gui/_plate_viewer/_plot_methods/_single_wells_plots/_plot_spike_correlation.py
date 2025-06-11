"""Spike train cross-correlation analysis for network analysis."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.signal import correlate
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_spike_correlation_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.1,
) -> None:
    """Plot spike train cross-correlation analysis.

    Args:
        widget: The graph widget to plot on
        data: Dictionary of ROI data
        rois: List of ROI indices to analyze, None for all active ROIs
        spike_threshold: Threshold for considering a spike event (0.0-1.0)
    """
    widget.figure.clear()

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        data, rois, spike_threshold
    )

    if correlation_matrix is None or rois_idxs is None:
        ax = widget.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Insufficient spike data for correlation analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Create subplot layout
    gs = widget.figure.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])

    # Main correlation matrix plot
    ax_main = widget.figure.add_subplot(gs[1, 0])

    # Hierarchical clustering
    distance_matrix = 1 - np.abs(correlation_matrix)
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method="average")

    # Get the order of ROIs after clustering
    clustered_order = leaves_list(linkage_matrix)

    # Reorder correlation matrix
    reordered_matrix = correlation_matrix[np.ix_(clustered_order, clustered_order)]
    reordered_rois = [rois_idxs[i] for i in clustered_order]

    # Plot reordered correlation matrix
    img = ax_main.imshow(reordered_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    # Add colorbar
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap=cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1)),
        ax=ax_main,
    )
    cbar.set_label("Spike Correlation")

    # Set labels and title
    ax_main.set_title(
        f"Spike Train Cross-Correlation (threshold={spike_threshold:.1f})"
    )
    ax_main.set_xlabel("ROI")
    ax_main.set_ylabel("ROI")

    # Set tick labels to ROI numbers
    tick_positions = range(len(reordered_rois))
    ax_main.set_xticks(tick_positions)
    ax_main.set_yticks(tick_positions)
    ax_main.set_xticklabels([str(roi) for roi in reordered_rois], rotation=45)
    ax_main.set_yticklabels([str(roi) for roi in reordered_rois])

    # Plot dendrogram
    ax_dendro = widget.figure.add_subplot(gs[1, 1])
    dendrogram(
        linkage_matrix,
        orientation="right",
        ax=ax_dendro,
        labels=[str(roi) for roi in rois_idxs],
        leaf_rotation=0,
    )
    ax_dendro.set_title("Clustering", fontsize=10)
    ax_dendro.set_xlabel("Distance")

    # Add hover functionality
    _add_correlation_hover_functionality(img, widget, reordered_rois, reordered_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _calculate_spike_cross_correlation(
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.5,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate spike train cross-correlation matrix.

    Args:
        data: Dictionary of ROI data
        rois: List of ROI indices to analyze
        spike_threshold: Threshold for spike detection

    Returns
    -------
        Tuple of correlation matrix and ROI indices
    """
    spike_trains: list[np.ndarray] = []
    rois_idxs: list[int] = []

    # Get spike trains for active ROIs
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue
        if not roi_data.active or not roi_data.inferred_spikes:
            continue

        spike_probs = np.array(roi_data.inferred_spikes)
        spike_train = (spike_probs >= spike_threshold).astype(float)

        # Only include ROIs with at least one spike
        if np.sum(spike_train) > 0:
            rois_idxs.append(int(roi_key))
            spike_trains.append(spike_train)

    if len(rois_idxs) <= 1:
        return None, None

    # Ensure all spike trains have the same length
    min_length = min(len(train) for train in spike_trains)
    spike_trains = [train[:min_length] for train in spike_trains]

    spike_trains_array = np.array(spike_trains)  # shape (n_rois, n_frames)

    n_rois = len(rois_idxs)
    correlation_matrix = np.zeros((n_rois, n_rois))

    for i, j in itertools.product(range(n_rois), range(n_rois)):
        if i == j:
            correlation_matrix[i, j] = 1.0
        else:
            # Calculate cross-correlation at zero lag
            spike_train_i = spike_trains_array[i]
            spike_train_j = spike_trains_array[j]

            # Normalize spike trains (z-score if there's variance)
            if np.std(spike_train_i) > 0 and np.std(spike_train_j) > 0:
                spike_train_i_norm = (spike_train_i - np.mean(spike_train_i)) / np.std(
                    spike_train_i
                )
                spike_train_j_norm = (spike_train_j - np.mean(spike_train_j)) / np.std(
                    spike_train_j
                )

                # Calculate Pearson correlation coefficient
                correlation = np.mean(spike_train_i_norm * spike_train_j_norm)
            else:
                # If no variance, check for exact matching
                correlation = (
                    1.0 if np.array_equal(spike_train_i, spike_train_j) else 0.0
                )

            correlation_matrix[i, j] = correlation

    return correlation_matrix, rois_idxs


def _plot_spike_correlation_with_lag(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    roi_pair: tuple[int, int],
    spike_threshold: float = 0.5,
    max_lag: int = 50,
) -> None:
    """Plot cross-correlation function with lag for a specific ROI pair.

    Args:
        widget: The graph widget to plot on
        data: Dictionary of ROI data
        roi_pair: Tuple of two ROI indices to analyze
        spike_threshold: Threshold for spike detection
        max_lag: Maximum lag to compute (in samples)
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    roi1_key, roi2_key = str(roi_pair[0]), str(roi_pair[1])

    if (
        roi1_key not in data
        or roi2_key not in data
        or not data[roi1_key].inferred_spikes
        or not data[roi2_key].inferred_spikes
    ):
        ax.text(
            0.5,
            0.5,
            f"No spike data for ROIs {roi_pair[0]} and {roi_pair[1]}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Get spike trains
    spikes1 = np.array(data[roi1_key].inferred_spikes) >= spike_threshold
    spikes2 = np.array(data[roi2_key].inferred_spikes) >= spike_threshold

    # Ensure same length
    min_len = min(len(spikes1), len(spikes2))
    spikes1 = spikes1[:min_len].astype(float)
    spikes2 = spikes2[:min_len].astype(float)

    if np.sum(spikes1) == 0 or np.sum(spikes2) == 0:
        ax.text(
            0.5,
            0.5,
            "No spikes detected in one or both ROIs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Calculate cross-correlation with different lags
    correlation = correlate(spikes1, spikes2, mode="full")

    # Get lag axis
    lags = np.arange(-len(spikes2) + 1, len(spikes1))

    # Limit to specified max_lag
    center = len(correlation) // 2
    start_idx = max(0, center - max_lag)
    end_idx = min(len(correlation), center + max_lag + 1)

    limited_correlation = correlation[start_idx:end_idx]
    limited_lags = lags[start_idx:end_idx]

    # Normalize correlation
    max_possible = np.sqrt(np.sum(spikes1) * np.sum(spikes2))
    if max_possible > 0:
        limited_correlation = limited_correlation / max_possible

    # Plot
    ax.plot(limited_lags, limited_correlation, "b-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="Zero lag")

    # Find and mark peak
    peak_idx = np.argmax(np.abs(limited_correlation))
    peak_lag = limited_lags[peak_idx]
    peak_value = limited_correlation[peak_idx]

    ax.plot(
        peak_lag,
        peak_value,
        "ro",
        markersize=8,
        label=f"Peak: lag={peak_lag}, corr={peak_value:.3f}",
    )

    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Normalized Cross-Correlation")
    ax.set_title(
        f"Spike Train Cross-Correlation: ROI {roi_pair[0]} vs ROI {roi_pair[1]}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_correlation_hover_functionality(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    correlation_matrix: np.ndarray,
) -> None:
    """Add hover functionality for correlation matrix."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            corr_value = correlation_matrix[y, x]

            sel.annotation.set(
                text=f"ROI {roi_x} ↔ ROI {roi_y}\nSpike Correlation: {corr_value:.3f}",
                fontsize=8,
                color="black",
            )
            widget.roiSelected.emit([str(roi_x), str(roi_y)])
