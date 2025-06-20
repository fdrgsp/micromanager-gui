from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np
from scipy import ndimage

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots.\
_plot_calcium_peaks_correlation import (
    _calculate_cross_correlation,
)

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _create_connectivity_matrix(
    correlation_matrix: np.ndarray,
    threshold_percentile: float = 90.0,
) -> np.ndarray:
    """Create binary connectivity matrix from correlation matrix.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Pairwise correlation matrix
    threshold_percentile : float
        Percentile threshold (0-100). Only correlations above this percentile
        become connections.

    Returns
    -------
    np.ndarray
        Binary connectivity matrix (1 = connected, 0 = not connected)
    """
    # Exclude diagonal (self-correlations = 1.0) for threshold calculation
    off_diagonal_mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
    off_diagonal_values = correlation_matrix[off_diagonal_mask]

    if len(off_diagonal_values) == 0:
        return np.eye(correlation_matrix.shape[0])

    # Calculate threshold
    threshold = np.percentile(off_diagonal_values, threshold_percentile)

    # Create binary connectivity matrix
    return (correlation_matrix >= threshold).astype(int)


def _get_roi_coordinates_from_labels(
    labels_image: np.ndarray,
    roi_indices: list[int],
) -> dict[int, tuple[float, float]]:
    """Extract ROI centroid coordinates from label image.

    Parameters
    ----------
    labels_image : np.ndarray
        2D label image where each ROI has a unique integer value
    roi_indices : list[int]
        List of ROI indices to find coordinates for

    Returns
    -------
    dict[int, tuple[float, float]]
        Dictionary mapping ROI index to (x, y) centroid coordinates
    """
    coordinates = {}

    for roi_idx in roi_indices:
        # Find pixels belonging to this ROI
        roi_mask = labels_image == roi_idx

        if np.any(roi_mask):
            # Calculate centroid
            centroid = ndimage.center_of_mass(roi_mask)
            # Convert to (x, y) - note that ndimage returns (row, col) = (y, x)
            coordinates[roi_idx] = (centroid[1], centroid[0])

    return coordinates


def _plot_connectivity_network_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    show_labels_background: bool = True,
) -> None:
    """Plot spatial functional connectivity network.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    data : dict[str, ROIData]
        Dictionary of ROI data
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    show_labels_background : bool
        Whether to show the label image as background
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Calculate correlation matrix
    correlation_matrix, rois_idxs = _calculate_cross_correlation(data, rois)

    if correlation_matrix is None or rois_idxs is None:
        LOGGER.warning(
            "Insufficient data for network connectivity analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        ax.text(0.5, 0.5, "Insufficient data for network analysis",
                ha="center", va="center", transform=ax.transAxes)
        widget.canvas.draw()
        return

    # Get network threshold from first available ROI
    network_threshold = 90.0  # Default
    if rois_idxs:
        first_roi_key = str(rois_idxs[0])
        if (first_roi_key in data and
            hasattr(data[first_roi_key], 'calcium_network_threshold') and
            data[first_roi_key].calcium_network_threshold is not None):
            network_threshold = data[first_roi_key].calcium_network_threshold

    # Ensure network_threshold is never None
    if network_threshold is None:
        network_threshold = 90.0

    # Create connectivity matrix
    connectivity_matrix = _create_connectivity_matrix(
        correlation_matrix, network_threshold
    )

    # Try to get spatial coordinates
    # For now, we'll use a simple grid layout if no spatial data is available
    coordinates = _create_grid_layout(rois_idxs)

    # Filter out ROIs without coordinates
    valid_indices = []
    valid_roi_labels = []
    pos_x = []
    pos_y = []

    for i, roi_idx in enumerate(rois_idxs):
        if roi_idx in coordinates:
            valid_indices.append(i)
            valid_roi_labels.append(roi_idx)
            x, y = coordinates[roi_idx]
            pos_x.append(x)
            pos_y.append(y)

    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    n_valid = len(valid_roi_labels)

    if n_valid < 2:
        LOGGER.warning(
            "Insufficient ROIs with valid coordinates for network visualization."
        )
        return

    # Draw edges first (so they appear behind nodes)
    edge_count = 0
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            orig_i = valid_indices[i]
            orig_j = valid_indices[j]

            if connectivity_matrix[orig_i, orig_j] == 1:
                # Line thickness and alpha based on correlation strength
                corr_strength = abs(correlation_matrix[orig_i, orig_j])
                linewidth = 1 + corr_strength * 3  # Scale from 1 to 4
                alpha = 0.4 + 0.6 * corr_strength  # Scale from 0.4 to 1.0

                # Color based on correlation sign
                color = "green" if correlation_matrix[orig_i, orig_j] > 0 else "red"

                ax.plot(
                    [pos_x[i], pos_x[j]],
                    [pos_y[i], pos_y[j]],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=1,
                )
                edge_count += 1

    # Draw nodes
    scatter = ax.scatter(
        pos_x,
        pos_y,
        s=300,
        c="yellow",
        edgecolors="black",
        linewidth=2,
        alpha=0.9,
        zorder=5,
    )

    # Add ROI labels
    for i, roi_idx in enumerate(valid_roi_labels):
        ax.annotate(
            f"{roi_idx}",
            (pos_x[i], pos_y[i]),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=6,
        )

    # Calculate network statistics
    total_possible_edges = n_valid * (n_valid - 1) // 2
    if total_possible_edges > 0:
        network_density = edge_count / total_possible_edges
    else:
        network_density = 0

    # Set plot properties
    ax.set_aspect("equal")
    ax.set_title(
        f"Functional Connectivity Network\n"
        f"Threshold: {network_threshold:.1f}% | "
        f"Edges: {edge_count}/{total_possible_edges} | "
        f"Density: {network_density:.3f}",
        fontsize=12,
        pad=20
    )
    ax.set_xlabel("X (relative)")
    ax.set_ylabel("Y (relative)")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="Positive correlation"),
        Line2D([0], [0], color="red", lw=2, label="Negative correlation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markersize=10, markeredgecolor="black", label="ROI", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    # Add hover functionality
    _add_hover_functionality_network(
        scatter, widget, valid_roi_labels, correlation_matrix, valid_indices
    )

    widget.figure.tight_layout()
    widget.canvas.draw()


def _create_grid_layout(roi_indices: list[int]) -> dict[int, tuple[float, float]]:
    """Create a grid layout for ROIs when spatial data is not available.

    Parameters
    ----------
    roi_indices : list[int]
        List of ROI indices

    Returns
    -------
    dict[int, tuple[float, float]]
        Dictionary mapping ROI index to (x, y) grid coordinates
    """
    n_rois = len(roi_indices)

    # Calculate grid dimensions (approximately square)
    grid_size = int(np.ceil(np.sqrt(n_rois)))

    coordinates = {}
    for i, roi_idx in enumerate(roi_indices):
        row = i // grid_size
        col = i % grid_size
        coordinates[roi_idx] = (col, row)

    return coordinates


def _add_hover_functionality_network(
    scatter: Any,
    widget: _SingleWellGraphWidget,
    roi_labels: list[int],
    correlation_matrix: np.ndarray,
    valid_indices: list[int],
) -> None:
    """Add hover functionality to network nodes."""
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the index of the clicked point
        point_idx = sel.target.index

        if point_idx < len(roi_labels):
            roi_idx = roi_labels[point_idx]
            orig_idx = valid_indices[point_idx]

            # Calculate node degree (number of connections)
            # Number of connections (excluding self)
            connections = np.sum(correlation_matrix[orig_idx, :] > 0) - 1

            # Find strongest connections
            correlations = correlation_matrix[orig_idx, :]
            correlations[orig_idx] = 0  # Exclude self-correlation
            strongest_idx = np.argmax(np.abs(correlations))
            strongest_corr = correlations[strongest_idx]
            strongest_idx_int = int(strongest_idx)
            if strongest_idx_int in valid_indices:
                strongest_roi = roi_labels[valid_indices.index(strongest_idx_int)]
            else:
                strongest_roi = "N/A"

            sel.annotation.set(
                text=(
                    f"ROI {roi_idx}\n"
                    f"Connections: {connections}\n"
                    f"Strongest: ROI {strongest_roi}\n"
                    f"Correlation: {strongest_corr:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            widget.roiSelected.emit([str(roi_idx)])


def _plot_connectivity_matrix_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot the binary connectivity matrix as a heatmap.

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

    # Calculate correlation matrix
    correlation_matrix, rois_idxs = _calculate_cross_correlation(data, rois)

    if correlation_matrix is None or rois_idxs is None:
        LOGGER.warning(
            "Insufficient data for connectivity matrix analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        widget.canvas.draw()
        return

    # Get network threshold
    network_threshold = 90.0  # Default
    if rois_idxs:
        first_roi_key = str(rois_idxs[0])
        if (first_roi_key in data and
            hasattr(data[first_roi_key], 'calcium_network_threshold') and
            data[first_roi_key].calcium_network_threshold is not None):
            network_threshold = data[first_roi_key].calcium_network_threshold

    # Ensure network_threshold is never None
    if network_threshold is None:
        network_threshold = 90.0

    # Create connectivity matrix
    connectivity_matrix = _create_connectivity_matrix(
        correlation_matrix, network_threshold
    )

    # Calculate network statistics
    n_nodes = len(rois_idxs)
    n_edges = np.sum(connectivity_matrix) - n_nodes  # Exclude diagonal
    total_possible_edges = n_nodes * (n_nodes - 1)
    network_density = n_edges / total_possible_edges if total_possible_edges > 0 else 0

    # Plot connectivity matrix
    img = ax.imshow(connectivity_matrix, cmap="Blues", vmin=0, vmax=1)

    # Add colorbar
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="Blues", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Connection Status")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Not Connected", "Connected"])

    # Set labels and title
    ax.set_title(
        f"Binary Connectivity Matrix\n"
        f"Threshold: {network_threshold:.1f}% | "
        f"Edges: {n_edges // 2} | "  # Divide by 2 since matrix is symmetric
        f"Density: {network_density:.3f}",
        fontsize=12
    )
    ax.set_xlabel("ROI Index")
    ax.set_ylabel("ROI Index")

    # Set tick labels to ROI indices
    ax.set_xticks(range(len(rois_idxs)))
    ax.set_xticklabels([str(roi) for roi in rois_idxs], rotation=45)
    ax.set_yticks(range(len(rois_idxs)))
    ax.set_yticklabels([str(roi) for roi in rois_idxs])

    # Add hover functionality
    _add_hover_functionality_connectivity_matrix(
        img, widget, rois_idxs, connectivity_matrix, correlation_matrix
    )

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_connectivity_matrix(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    connectivity_matrix: np.ndarray,
    correlation_matrix: np.ndarray,
) -> None:
    """Add hover functionality to connectivity matrix heatmap."""
    cursor = mplcursors.cursor(image, hover=True)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        if x < len(rois) and y < len(rois):
            roi_x, roi_y = rois[x], rois[y]
            is_connected = connectivity_matrix[y, x]
            correlation = correlation_matrix[y, x]

            status = "Connected" if is_connected else "Not Connected"

            sel.annotation.set(
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Status: {status}\n"
                    f"Correlation: {correlation:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            widget.roiSelected.emit([str(roi_x), str(roi_y)])
