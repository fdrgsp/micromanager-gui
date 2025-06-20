from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mplcursors
import numpy as np
from skimage import measure

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation import (
    _calculate_cross_correlation,
)
from micromanager_gui._plate_viewer._util import coordinates_to_mask

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_connectivity_network_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
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
        return

    # Get network threshold from first available ROI
    network_threshold: float | None = None
    if rois_idxs:
        first_roi_key = str(rois_idxs[0])
        if (
            first_roi_key in data
            and data[first_roi_key].calcium_network_threshold is not None
        ):
            network_threshold = data[first_roi_key].calcium_network_threshold

    # Ensure network_threshold is never None
    if network_threshold is None:
        network_threshold = 90.0

    # Create connectivity matrix
    connectivity_matrix = _create_connectivity_matrix(
        correlation_matrix, network_threshold
    )

    # Try to get ROI shapes from mask data
    roi_shapes = _get_roi_shapes_from_mask_data(data, rois_idxs)

    # If no shape data available, fall back to coordinates only
    if len(roi_shapes) < len(rois_idxs):
        LOGGER.warning("Some ROIs do not have mask data.")
        return

    # Extract coordinates from shape data
    coordinates = {
        roi_idx: shape_data["centroid"] for roi_idx, shape_data in roi_shapes.items()
    }

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

    # Create composite ROI image using actual mask coordinates
    composite_image, image_centroids, image_shape = _create_roi_composite_image(
        data, rois_idxs
    )

    if composite_image.size == 0 or not image_centroids:
        # Fall back to coordinate-based approach if no mask data available
        LOGGER.info("No mask coordinate data available, using centroids only")
        return

    # Use image-based approach with actual ROI masks
    # Create binary image for ROI display (black ROIs on light gray background)
    roi_display_image = np.ones((*image_shape, 3)) * 0.9  # Light gray background

    valid_indices = []
    valid_roi_labels = []

    for i, roi_idx in enumerate(rois_idxs):
        if roi_idx in image_centroids:
            valid_indices.append(i)
            valid_roi_labels.append(roi_idx)

            # Color the ROI pixels black
            roi_mask = composite_image == roi_idx
            if np.any(roi_mask):
                roi_display_image[roi_mask] = [0, 0, 0]  # Black color

    # Draw edges using image centroids
    edge_count = 0
    n_valid = len(valid_roi_labels)

    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            orig_i = valid_indices[i]
            orig_j = valid_indices[j]

            if connectivity_matrix[orig_i, orig_j] == 1:
                roi_i = valid_roi_labels[i]
                roi_j = valid_roi_labels[j]

                if roi_i in image_centroids and roi_j in image_centroids:
                    # Line thickness and alpha based on correlation strength
                    corr_strength = abs(correlation_matrix[orig_i, orig_j])
                    linewidth = 1 + corr_strength * 1  # Scale from 1 to 4
                    alpha = 0.4 + 0.6 * corr_strength  # Scale from 0.4 to 1.0

                    # Color based on correlation sign
                    color = (
                        "green" if correlation_matrix[orig_i, orig_j] > 0 else "magenta"
                    )
                    x1, y1 = image_centroids[roi_i]
                    x2, y2 = image_centroids[roi_j]

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=3,  # Above image, below labels
                    )
                    edge_count += 1

    # Display the ROI image
    im = ax.imshow(roi_display_image, alpha=1.0, zorder=2)

    # Collect ROI centroids for hover functionality (no visual labels)
    drawn_rois = []
    for roi_idx in valid_roi_labels:
        if roi_idx in image_centroids:
            x, y = image_centroids[roi_idx]
            drawn_rois.append((roi_idx, x, y))

    # Calculate network statistics
    total_possible_edges = n_valid * (n_valid - 1) // 2
    if total_possible_edges > 0:
        network_density = edge_count / total_possible_edges
    else:
        network_density = 0

    # Set plot properties
    ax.set_aspect("equal")
    ax.set_title(
        f"Calcium Peaks Functional Connectivity Network\n"
        f"Threshold: {network_threshold:.1f}% | "
        f"Edges: {edge_count}/{total_possible_edges} | "
        f"Density: {network_density*100:.1f}%",
        fontsize=12,
        pad=20,
    )
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add simple hover functionality using the composite image
    _add_hover_functionality(im, composite_image, widget)

    # Set axis limits with some padding
    if drawn_rois:
        all_x = [x for _, x, _ in drawn_rois]
        all_y = [y for _, _, y in drawn_rois]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add padding (10% of range or minimum 10 pixels)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = max(x_range * 0.1, 10)
        y_pad = max(y_range * 0.1, 10)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Invert y-axis to match image coordinates (origin at top-left)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    widget.figure.tight_layout()
    widget.canvas.draw()


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


def _get_roi_shapes_from_mask_data(
    data: dict[str, ROIData],
    roi_indices: list[int],
) -> dict[int, dict[str, Any]]:
    """Extract ROI shapes and contours from mask coordinate data.

    Parameters
    ----------
    data : dict[str, ROIData]
        Dictionary containing ROI data with mask_coord_and_shape information
    roi_indices : list[int]
        List of ROI indices to extract shapes for

    Returns
    -------
    dict[int, dict[str, Any]]
        Dictionary mapping ROI index to shape data containing:
        - 'centroid': (x, y) centroid coordinates
        - 'contours': list of contour arrays for matplotlib Polygon
        - 'mask': 2D boolean mask array
        - 'bbox': bounding box (x_min, y_min, x_max, y_max)
    """
    roi_shapes = {}

    for roi_idx in roi_indices:
        roi_key = str(roi_idx)
        if roi_key in data:
            roi_data = data[roi_key]

            # Check if mask coordinate and shape data is available
            if roi_data.mask_coord_and_shape is not None:

                # Extract coordinates and shape
                (y_coords, x_coords), (height, width) = roi_data.mask_coord_and_shape

                if len(x_coords) > 0 and len(y_coords) > 0:
                    # Reconstruct the 2D mask
                    mask = coordinates_to_mask((y_coords, x_coords), (height, width))

                    # Calculate centroid
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)

                    # Calculate bounding box
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)

                    # Find contours for plotting
                    contours = measure.find_contours(mask.astype(float), 0.5)

                    # Convert contours to matplotlib format (x, y instead of row, col)
                    matplotlib_contours = []
                    for contour in contours:
                        # Swap row/col to x/y and adjust for plotting
                        matplotlib_contour = np.column_stack(
                            [contour[:, 1], contour[:, 0]]
                        )
                        matplotlib_contours.append(matplotlib_contour)

                    roi_shapes[roi_idx] = {
                        "centroid": (centroid_x, centroid_y),
                        "contours": matplotlib_contours,
                        "mask": mask,
                        "bbox": (x_min, y_min, x_max, y_max),
                    }

    return roi_shapes


def _create_roi_composite_image(
    data: dict[str, ROIData],
    roi_indices: list[int],
) -> tuple[np.ndarray, dict[int, tuple[float, float]], tuple[int, int]]:
    """Create a composite image showing all ROI masks with their actual coordinates.

    Parameters
    ----------
    data : dict[str, ROIData]
        Dictionary containing ROI data with mask_coord_and_shape information
    roi_indices : list[int]
        List of ROI indices to include in the image

    Returns
    -------
    tuple[np.ndarray, dict[int, tuple[float, float]], tuple[int, int]]
        - Composite image array with ROI masks (each ROI has unique integer value)
        - Dictionary mapping ROI index to (x, y) centroid coordinates
        - Image shape (height, width)
    """
    roi_masks = {}
    centroids = {}
    original_shape = None

    # First pass: collect all ROI masks and determine original shape
    for roi_idx in roi_indices:
        roi_key = str(roi_idx)
        if roi_key in data:
            roi_data = data[roi_key]

            # Check if mask coordinate and shape data is available
            if roi_data.mask_coord_and_shape is not None:

                # Extract coordinates and original shape
                (y_coords, x_coords), (height, width) = roi_data.mask_coord_and_shape

                if len(x_coords) > 0 and len(y_coords) > 0:
                    roi_masks[roi_idx] = (y_coords, x_coords)

                    # Calculate centroid
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)
                    centroids[roi_idx] = (centroid_x, centroid_y)

                    # Use the original mask shape
                    if original_shape is None:
                        original_shape = (height, width)

    if not roi_masks or original_shape is None:
        # No valid masks found, return empty image
        return np.zeros((100, 100), dtype=int), {}, (100, 100)

    # Create composite image with original shape
    img_height, img_width = original_shape
    composite_image = np.zeros((img_height, img_width), dtype=int)

    # Paint each ROI mask onto the composite image
    for roi_idx, (y_coords, x_coords) in roi_masks.items():
        # Ensure coordinates are within bounds
        y_coords = np.array(y_coords)
        x_coords = np.array(x_coords)

        valid_mask = (
            (y_coords >= 0)
            & (y_coords < img_height)
            & (x_coords >= 0)
            & (x_coords < img_width)
        )

        if np.any(valid_mask):
            valid_y = y_coords[valid_mask]
            valid_x = x_coords[valid_mask]

            # Paint ROI with its index value
            composite_image[valid_y, valid_x] = roi_idx

    return composite_image, centroids, (img_height, img_width)


def _add_hover_functionality(
    image: Any,
    composite_image: np.ndarray,
    widget: _SingleWellGraphWidget,
) -> None:
    """Add simple hover functionality to show ROI number when mouse is over ROI.

    Parameters
    ----------
    image : Any
        The matplotlib image object
    composite_image : np.ndarray
        The composite image array where each ROI has its index as pixel value
    widget : _SingleWellGraphWidget
        Widget to emit ROI selection events
    """
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the pixel coordinates
        x, y = map(int, np.round([sel.target[0], sel.target[1]]))

        # Check if coordinates are within image bounds
        if 0 <= y < composite_image.shape[0] and 0 <= x < composite_image.shape[1]:
            roi_value = composite_image[y, x]

            if roi_value > 0:  # ROI pixel (not background)
                sel.annotation.set(
                    text=f"ROI {roi_value}",
                    fontsize=8,
                    color="black",
                )
                # Emit selection for only the single hovered ROI
                widget.roiSelected.emit([str(roi_value)])
            else:
                # Hide annotation for background pixels
                sel.annotation.set_visible(False)
        else:
            sel.annotation.set_visible(False)



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
        if (
            first_roi_key in data
            and data[first_roi_key].calcium_network_threshold is not None
        ):
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
    img = ax.imshow(connectivity_matrix, vmin=0, vmax=1)

    # Set labels and title
    ax.set_title(
        f"Binary Connectivity Matrix\n"
        f"Threshold: {network_threshold:.1f}% | "
        f"Edges: {n_edges // 2} | "  # Divide by 2 since matrix is symmetric
        f"Density: {network_density:.3f}",
        fontsize=12,
    )
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

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