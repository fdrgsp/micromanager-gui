from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.signal import correlate
from scipy.spatial.distance import squareform
from scipy.stats import zscore

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import _get_spikes_over_threshold

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _calculate_spike_cross_correlation(
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for spike trains from active ROIs.

    This function extracts thresholded spike data from ROIs and computes pairwise
    cross-correlations using the same approach as calcium trace correlation but
    applied to binary spike trains.

    Parameters
    ----------
    data : dict[str, ROIData]
        Dictionary of ROI data containing spike information
    rois : list[int] | None
        List of specific ROI indices to analyze, None for all active ROIs

    Returns
    -------
    tuple[np.ndarray | None, list[int] | None]
        Correlation matrix and corresponding ROI indices, or (None, None) if
        insufficient data
    """
    spike_trains: list[np.ndarray] = []
    rois_idxs: list[int] = []

    # Extract spike trains for the active ROIs
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue
        if not roi_data.active:
            continue

        # Get thresholded spike data using existing utility function
        spike_probs = _get_spikes_over_threshold(roi_data)
        if spike_probs is None or len(spike_probs) == 0:
            continue

        # Convert spike probabilities to binary spike train
        spike_train = np.array(spike_probs) > 0.0

        # Only include ROIs that have at least one spike
        if np.sum(spike_train) > 0:
            rois_idxs.append(int(roi_key))
            spike_trains.append(spike_train.astype(float))

    if len(rois_idxs) <= 1:
        LOGGER.warning(
            "Insufficient spike data for correlation analysis. "
            "Need at least 2 ROIs with spikes."
        )
        return None, None

    # Convert to array for processing
    spike_trains_array = np.array(spike_trains)  # shape (n_rois, n_frames)

    # Z-score normalization (mean centering and std normalization)
    # This is important for spike trains to handle different firing rates
    spike_trains_zscore = zscore(spike_trains_array, axis=1, nan_policy="omit")

    # Handle cases where std is 0 (constant spike trains)
    spike_trains_zscore = np.nan_to_num(
        spike_trains_zscore, nan=0.0, posinf=0.0, neginf=0.0
    )

    n_rois = len(rois_idxs)
    correlation_matrix = np.zeros((n_rois, n_rois))

    # Calculate pairwise cross-correlations
    for i, j in itertools.product(range(n_rois), range(n_rois)):
        x = spike_trains_zscore[i]
        y = spike_trains_zscore[j]

        # Compute cross-correlation using FFT method for efficiency
        corr = correlate(x, y, mode="full", method="fft")

        # Normalize by the norms of the signals
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x > 0 and norm_y > 0:
            corr /= norm_x * norm_y
            correlation_matrix[i, j] = np.max(np.abs(corr))  # Take max absolute value
        else:
            correlation_matrix[i, j] = 0.0  # No correlation for constant signals

    return correlation_matrix, rois_idxs


def _plot_spike_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot pairwise cross-correlation matrix for spike trains.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    data : dict[str, ROIData]
        Dictionary of ROI data
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(data, rois)

    if correlation_matrix is None or rois_idxs is None:
        LOGGER.warning(
            "Insufficient spike data for cross-correlation analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        widget.canvas.draw()
        return

    ax.set_title("Pairwise Spike Cross-Correlation Matrix\n(Thresholded Spike Data)")
    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_box_aspect(1)

    # Create colorbar
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    # Display the correlation matrix
    img = ax.imshow(correlation_matrix, cmap="viridis", vmin=0, vmax=1)

    # Add hover functionality
    _add_hover_functionality_spike_corr(img, widget, rois_idxs, correlation_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_spike_corr(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors for spike correlation matrix.

    Parameters
    ----------
    image : AxesImage
        The imshow image object
    widget : _SingleWellGraphWidget
        Widget containing the plot
    rois : list[int]
        List of ROI indices
    values : np.ndarray
        Correlation matrix values
    """
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            sel.annotation.set(
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Spike Correlation: {values[y, x]:0.3f}"
                ),
                fontsize=8,
                color="black",
            )
            widget.roiSelected.emit([str(roi_x), str(roi_y)])


def _plot_spike_hierarchical_clustering_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    use_dendrogram: bool = False,
) -> None:
    """Plot hierarchical clustering analysis for spike correlation data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    data : dict[str, ROIData]
        Dictionary of ROI data
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    use_dendrogram : bool
        If True, plot dendrogram; if False, plot clustered heatmap
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(data, rois)

    if correlation_matrix is None or rois_idxs is None:
        LOGGER.warning(
            "Insufficient spike data for hierarchical clustering analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        widget.canvas.draw()
        return

    if use_dendrogram:
        _plot_spike_hierarchical_clustering_dendrogram(
            ax, correlation_matrix, rois_idxs
        )
    else:
        _plot_spike_hierarchical_clustering_map(
            widget, ax, correlation_matrix, rois_idxs
        )

    ax.set_xlabel("ROI")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_spike_hierarchical_clustering_dendrogram(
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering dendrogram for spike correlation data.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    correlation_matrix : np.ndarray
        Correlation matrix
    rois_idxs : list[int]
        List of ROI indices
    """
    ax.set_title(
        ""
        "Spike Cross-Correlation (Hierarchical Clustering Dendrogram)\n"
        "(Thresholded Spike Data)"
    )
    ax.set_ylabel("Distance")

    # Round to avoid numerical precision issues
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Convert correlation to distance (1 - correlation)
    dist_condensed = squareform(1 - np.abs(correlation_matrix))

    # Perform hierarchical clustering
    Z = linkage(dist_condensed, method="complete")

    # Create labels
    labels = [str(i) for i in rois_idxs]

    # Plot dendrogram
    dendrogram(Z, ax=ax, labels=labels, leaf_rotation=90, leaf_font_size=12)


def _plot_spike_hierarchical_clustering_map(
    widget: _SingleWellGraphWidget,
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering map for spike correlation data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget containing the plot
    ax : Axes
        Matplotlib axes to plot on
    correlation_matrix : np.ndarray
        Correlation matrix
    rois_idxs : list[int]
        List of ROI indices
    """
    # Round to avoid numerical precision issues
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Convert correlation to distance and perform clustering
    dist_condensed = squareform(1 - np.abs(correlation_matrix))
    order = leaves_list(linkage(dist_condensed, method="complete"))

    # Reorder matrix according to clustering
    reordered_matrix = correlation_matrix[order][:, order]

    ax.set_title(
        "Spike Cross-Correlation (Hierarchical Clustering Map)\n"
        "(Thresholded Spike Data)"
    )
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_box_aspect(1)

    # Display the reordered correlation matrix
    image = ax.imshow(reordered_matrix, cmap="viridis", vmin=0, vmax=1)

    # Add colorbar
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    # Add hover functionality
    _add_hover_functionality_spike_clustering(
        image, widget, rois_idxs, order, reordered_matrix
    )


def _add_hover_functionality_spike_clustering(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    order: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality for spike clustering heatmap.

    Parameters
    ----------
    image : AxesImage
        The imshow image object
    widget : _SingleWellGraphWidget
        Widget containing the plot
    rois : list[int]
        List of ROI indices
    order : list[int]
        Clustering order indices
    values : np.ndarray
        Reordered correlation matrix values
    """
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[order[x]], rois[order[y]]
            sel.annotation.set(
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Spike Correlation: {values[y, x]:0.3f}"
                ),
                fontsize=8,
                color="black",
            )
            widget.roiSelected.emit([str(roi_x), str(roi_y)])
