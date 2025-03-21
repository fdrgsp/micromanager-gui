from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import mplcursors
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_synchrony(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot global synchrony."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if rois is None:
        rois = [int(roi) for roi in data.keys() if roi.isdigit()]

    phase_dict = _get_phase_dict_from_rois(data, rois)
    if phase_dict is None:
        return None

    active_rois = list(phase_dict.keys())

    synchrony_matrix = _get_synchrony_matrix(phase_dict)

    if synchrony_matrix is None:
        return None

    synchrony = _get_synchrony(synchrony_matrix)

    ax.imshow(synchrony_matrix, cmap="viridis", aspect="auto")
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap=cm.viridis),
        ax=ax,
    )
    cbar.set_label("Synchrony index")

    ax.set_title(f"Global Synchrony: {synchrony:0.4f}")

    ax.set_ylabel("ROIs")
    ax.set_yticklabels([])
    ax.set_yticks([])

    ax.set_xlabel("ROIs")
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.set_box_aspect(1)

    _add_hover_functionality(ax, widget, active_rois, synchrony_matrix)
    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_synchrony(synchrony_matrix: np.ndarray | None) -> float | None:
    """Calculate the connection matrix."""
    if synchrony_matrix is None or synchrony_matrix.size == 0:
        return None

    # Ensure the matrix is at least 2x2 and square
    if synchrony_matrix.shape[0] < 2 or (
        synchrony_matrix.shape[0] != synchrony_matrix.shape[1]
    ):
        return None

    return float(
        np.median(np.sum(synchrony_matrix, axis=0) - 1)
        / (synchrony_matrix.shape[0] - 1)
    )


def _get_synchrony_matrix(phase_dict: dict[str, list[float]]) -> np.ndarray | None:
    """Calculate global synchrony (FluoroSNNAP)."""
    active_rois = list(phase_dict.keys())  # ROI names

    if len(active_rois) < 2:
        return None

    # Convert phase_dict values into a NumPy array of shape (N, T)
    phase_array = np.array([phase_dict[roi] for roi in active_rois])  # Shape (N, T)

    # Compute pairwise phase difference using broadcasting (Shape: (N, N, T))
    phase_diff = np.expand_dims(phase_array, axis=1) - np.expand_dims(
        phase_array, axis=0
    )

    # Ensure phase difference is within valid range [0, 2π]
    phase_diff = np.mod(np.abs(phase_diff), 2 * np.pi)

    # Compute cosine and sine of the phase differences
    cos_mean = np.mean(np.cos(phase_diff), axis=2)  # Shape: (N, N)
    sin_mean = np.mean(np.sin(phase_diff), axis=2)  # Shape: (N, N)

    return np.array(np.sqrt(cos_mean**2 + sin_mean**2))


def _get_phase_dict_from_rois(
    roi_data_dict: dict[str, ROIData], rois: list[int]
) -> dict[str, list[float]] | None:
    """Organize the phase list from the rois wanted."""
    phase_dict: dict[str, list[float]] = {}

    # if less than two rois input, can't calculate synchrony
    if len(rois) < 2:
        return None

    for roi, roi_data in roi_data_dict.items():
        roi_int = int(roi)
        if roi_int not in rois:
            continue

        phase_list = roi_data.instantaneous_phase

        if not phase_list:
            continue

        phase_dict[roi] = phase_list

    return phase_dict


def _add_hover_functionality(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    rois: list[str],
    synchrony_matrix: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        y, x = int(sel.target[0]), int(sel.target[1])
        roi_x, roi_y = rois[x], rois[y]

        sel.annotation.set(
            text=f"ROI {roi_x} ↔ ROI {roi_y}\nvalue: {synchrony_matrix[y, x]:0.2f}",
            fontsize=8,
            color="black",
        )
        if roi_x.isdigit() and roi_y.isdigit():
            # TODO: display two ROIs?
            widget.roiSelected.emit(roi_x)
            widget.roiSelected.emit(roi_y)
