"""Functional clustering analysis based on spiking patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_spike_clustering_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.1,
    n_clusters: int | None = None,
    clustering_method: str = "ward",
) -> None:
    """Plot functional clustering analysis based on spiking patterns.

    Args:
        widget: The graph widget to plot on
        data: Dictionary of ROI data
        rois: List of ROI indices to analyze, None for all active ROIs
        spike_threshold: Threshold for spike event (0.0-1.0)
        n_clusters: Number of clusters to create, None for automatic selection
        clustering_method: Method for clustering ('ward', 'average', 'complete')
    """
    widget.figure.clear()

    # Extract spike features for clustering
    features, roi_names = _extract_spike_features(data, rois, spike_threshold)

    if features is None or len(features) < 3:
        ax = widget.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Insufficient spike data for clustering analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Perform hierarchical clustering
    cluster_labels, linkage_matrix, distance_matrix = _perform_hierarchical_clustering(
        features, n_clusters, clustering_method
    )

    # Create subplot layout
    fig = widget.figure
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3
    )

    # Plot 1: Simple 2D projection with clusters
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_simple_projection_clusters(ax1, features, cluster_labels, roi_names)

    # Plot 2: Dendrogram
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_dendrogram(ax2, linkage_matrix, roi_names)

    # Plot 3: Feature heatmap with clustering
    ax3 = fig.add_subplot(gs[1, :])
    _plot_feature_heatmap(ax3, features, roi_names, cluster_labels)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _extract_spike_features(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.5,
) -> tuple[np.ndarray | None, list[str]]:
    """Extract features from spike trains for clustering.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all
        spike_threshold: Threshold for spike detection

    Returns
    -------
        Tuple of (feature_matrix, roi_names)

    """
    spike_trains: list[np.ndarray] = []
    roi_names: list[str] = []

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 3:
        return None, []

    max_length = 0
    for roi, roi_data in roi_data_dict.items():
        if int(roi) not in rois or not roi_data.active:
            continue

        if (spike_probs := roi_data.inferred_spikes) is not None:
            spike_train = (np.array(spike_probs) >= spike_threshold).astype(float)
            if np.sum(spike_train) > 0:  # Only include ROIs with at least one spike
                spike_trains.append(spike_train)
                roi_names.append(roi)
                max_length = max(max_length, len(spike_train))

    if len(spike_trains) < 3:
        return None, []

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

    # Extract multiple features from spike trains
    features = []

    for spike_train in spike_trains_array:
        spike_times = np.where(spike_train > 0)[0]

        # Feature 1: Firing rate
        firing_rate = len(spike_times) / len(spike_train)

        # Feature 2: Inter-spike interval statistics
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
        else:
            mean_isi = 0
            std_isi = 0
            cv_isi = 0

        # Feature 3: Burstiness (local variance in spike timing)
        if len(spike_times) > 2:
            # Calculate local variance in spike intervals
            window_size = min(5, len(spike_times) - 1)
            local_vars = []
            for i in range(len(spike_times) - window_size):
                window_isis = np.diff(spike_times[i : i + window_size + 1])
                if len(window_isis) > 1:
                    local_vars.append(np.var(window_isis))
            burstiness = np.mean(local_vars) if local_vars else 0
        else:
            burstiness = 0

        # Feature 4: Temporal distribution
        if len(spike_times) > 0:
            first_spike_pos = spike_times[0] / len(spike_train)
            last_spike_pos = spike_times[-1] / len(spike_train)
            spike_spread = last_spike_pos - first_spike_pos
        else:
            first_spike_pos = 0
            last_spike_pos = 0
            spike_spread = 0

        # Feature 5: Activity periods
        if len(spike_times) > 0:
            # Group spikes that are close together (within 10 samples)
            activity_periods = 1
            for i in range(1, len(spike_times)):
                if spike_times[i] - spike_times[i - 1] > 10:
                    activity_periods += 1
        else:
            activity_periods = 0

        # Combine features
        roi_features = [
            firing_rate,
            mean_isi,
            cv_isi,
            burstiness,
            first_spike_pos,
            last_spike_pos,
            spike_spread,
            activity_periods / len(spike_train),  # Normalize
        ]

        features.append(roi_features)

    features_array = np.array(features)

    # Normalize features (z-score)
    for i in range(features_array.shape[1]):
        feature_col = features_array[:, i]
        if np.std(feature_col) > 0:
            features_array[:, i] = (feature_col - np.mean(feature_col)) / np.std(
                feature_col
            )

    return features_array, roi_names


def _perform_hierarchical_clustering(
    features: np.ndarray,
    n_clusters: int | None = None,
    method: str = "ward",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform hierarchical clustering on spike features.

    Args:
        features: Feature matrix (n_rois, n_features)
        n_clusters: Number of clusters, None for automatic selection
        method: Clustering method

    Returns
    -------
        Tuple of (cluster_labels, linkage_matrix, distance_matrix)

    """
    # Calculate distance matrix
    distance_matrix = pdist(features, metric="euclidean")

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=method)

    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        n_clusters = _find_optimal_clusters_simple(
            features, linkage_matrix, max_k=min(10, len(features) - 1)
        )

    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    return cluster_labels, linkage_matrix, squareform(distance_matrix)


def _find_optimal_clusters_simple(
    features: np.ndarray,
    linkage_matrix: np.ndarray,
    max_k: int,
) -> int:
    """Find optimal number of clusters using within-cluster variance.

    Args:
        features: Feature matrix
        linkage_matrix: Linkage matrix from hierarchical clustering
        max_k: Maximum number of clusters to test

    Returns
    -------
        Optimal number of clusters

    """
    if max_k < 2:
        return 2

    variances = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        cluster_labels = fcluster(linkage_matrix, k, criterion="maxclust")
        # Calculate within-cluster variance
        total_variance = 0
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            if len(cluster_features) > 1:
                total_variance += np.sum(np.var(cluster_features, axis=0))
        variances.append(total_variance)

    # Use elbow method - find point where variance reduction slows
    if len(variances) > 1:
        diffs = np.diff(variances)
        if len(diffs) > 1:
            second_diffs = np.diff(diffs)
            elbow_point = np.argmax(second_diffs) + 2
            return int(min(elbow_point, max_k))

    return 3  # Default fallback


def _plot_simple_projection_clusters(
    ax,
    features: np.ndarray,
    cluster_labels: np.ndarray,
    roi_names: list[str],
) -> None:
    """Plot simple 2D projection of clusters using first two principal components."""
    # Simple PCA implementation using SVD
    features_centered = features - np.mean(features, axis=0)
    U, s, Vt = np.linalg.svd(features_centered, full_matrices=False)
    features_pca = U[:, :2] * s[:2]

    # Plot clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        ax.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=50,
        )

    # Add ROI labels
    for i, roi_name in enumerate(roi_names):
        ax.annotate(
            roi_name,
            (features_pca[i, 0], features_pca[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Calculate explained variance ratio
    total_var = np.sum(s**2)
    var_ratio = s[:2] ** 2 / total_var

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)")
    ax.set_title("Spike Pattern Clusters (PCA Projection)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_dendrogram(
    ax,
    linkage_matrix: np.ndarray,
    roi_names: list[str],
) -> None:
    """Plot dendrogram of hierarchical clustering."""
    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=roi_names,
        orientation="top",
        distance_sort=True,
        show_leaf_counts=True,
    )
    ax.set_title("Hierarchical Clustering")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Distance")


def _plot_feature_heatmap(
    ax,
    features: np.ndarray,
    roi_names: list[str],
    cluster_labels: np.ndarray,
) -> None:
    """Plot heatmap of features organized by clusters."""
    # Sort ROIs by cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_features = features[sorted_indices]
    sorted_roi_names = [roi_names[i] for i in sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]

    # Create heatmap
    im = ax.imshow(sorted_features.T, cmap="RdBu_r", aspect="auto")

    # Set labels
    feature_names = [
        "Firing Rate",
        "Mean ISI",
        "CV ISI",
        "Burstiness",
        "First Spike Pos",
        "Last Spike Pos",
        "Spike Spread",
        "Activity Periods",
    ]

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xticks(range(len(sorted_roi_names)))
    ax.set_xticklabels(sorted_roi_names, rotation=45, ha="right")
    ax.set_xlabel("ROI (sorted by cluster)")
    ax.set_title("Spike Pattern Features by Cluster")

    # Add cluster boundaries
    cluster_boundaries = []
    current_cluster = sorted_labels[0]
    for i, label in enumerate(sorted_labels):
        if label != current_cluster:
            cluster_boundaries.append(i - 0.5)
            current_cluster = label

    for boundary in cluster_boundaries:
        ax.axvline(x=boundary, color="black", linewidth=2)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Feature Value (z-score)")
