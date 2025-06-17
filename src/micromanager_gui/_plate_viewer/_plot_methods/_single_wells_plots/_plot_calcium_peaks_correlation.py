from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.signal import correlate
from scipy.spatial.distance import squareform
from scipy.stats import zscore

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _calculate_cross_correlation(
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for the active ROIs."""
    traces: list[list[float]] = []
    rois_idxs: list[int] = []
    # get the traces for the active rois
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue
        if not roi_data.active or not roi_data.dec_dff:
            continue
        rois_idxs.append(int(roi_key))
        traces.append(roi_data.dec_dff)

    if len(rois_idxs) <= 1:
        return None, None

    traces_array = np.array(traces)  # shape (n_rois, n_frames)

    dff_zero_mean = zscore(traces_array, axis=1)

    n_rois = len(rois_idxs)
    correlation_matrix_active = np.zeros((n_rois, n_rois))
    for i, j in itertools.product(range(n_rois), range(n_rois)):
        x = dff_zero_mean[i]
        y = dff_zero_mean[j]
        corr = correlate(x, y, mode="full", method="fft")
        corr /= np.linalg.norm(x) * np.linalg.norm(y)  # normalises magnitude
        correlation_matrix_active[i, j] = np.max(corr)
    return correlation_matrix_active, rois_idxs


def _plot_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_cross_correlation(data, rois)

    if correlation_matrix is None or rois_idxs is None:
        return

    ax.set_title("Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)")
    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("ROI")
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_box_aspect(1)

    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    img = ax.imshow(correlation_matrix, cmap="viridis", vmin=0, vmax=1)

    _add_hover_functionality_cross_corr(img, widget, rois_idxs, correlation_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_cross_corr(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        roi_x, roi_y = rois[x], rois[y]
        sel.annotation.set(
            text=f"ROI {roi_x} ↔ ROI {roi_y}\nvalue: {values[y, x]:0.2f}",
            fontsize=8,
            color="black",
        )

        widget.roiSelected.emit([str(roi_x), str(roi_y)])


def _plot_hierarchical_clustering_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    use_dendrogram: bool = False,
) -> None:
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    correlation_matrix, rois_idxs = _calculate_cross_correlation(data, rois)

    if correlation_matrix is None or rois_idxs is None:
        return

    if use_dendrogram:
        _plot_hierarchical_clustering_dendrogram(ax, correlation_matrix, rois_idxs)
    else:
        _plot_hierarchical_clustering_map(widget, ax, correlation_matrix, rois_idxs)

    ax.set_xlabel("ROI")

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_hierarchical_clustering_dendrogram(
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering dendrogram."""
    ax.set_title(
        "Pairwise Cross-Correlation - Hierarchical Clustering Dendrogram\n"
        "(Calcium Peaks Events)"
    )
    ax.set_ylabel("Distance")
    correlation_matrix = np.round(correlation_matrix, decimals=8)
    dist_condensed = squareform(1 - np.abs(correlation_matrix))
    Z = linkage(dist_condensed, method="complete")
    labels = [str(i) for i in rois_idxs]
    dendrogram(Z, ax=ax, labels=labels, leaf_rotation=90, leaf_font_size=12)


def _plot_hierarchical_clustering_map(
    widget: _SingleWellGraphWidget,
    ax: Axes,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering map."""
    correlation_matrix = np.round(correlation_matrix, decimals=8)
    dist_condensed = squareform(1 - np.abs(correlation_matrix))
    order = leaves_list(linkage(dist_condensed, method="complete"))
    reordered_matrix = correlation_matrix[order][:, order]
    ax.set_title(
        "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
        "(Calcium Peaks Events)"
    )
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    image = ax.imshow(reordered_matrix, cmap="viridis")

    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Cross-Correlation Index")

    _add_hover_functionality_clustering(
        image, widget, rois_idxs, order, reordered_matrix
    )


def _add_hover_functionality_clustering(
    image: AxesImage,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    order: list[int],
    values: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(image, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        x, y = map(int, np.round(sel.target))
        roi_x, roi_y = rois[order[x]], rois[order[y]]

        sel.annotation.set(
            text=f"ROI {roi_x} ↔ ROI {roi_y}\nvalue: {values[y, x]:0.2f}",
            fontsize=8,
            color="black",
        )

        widget.roiSelected.emit([str(roi_x), str(roi_y)])
