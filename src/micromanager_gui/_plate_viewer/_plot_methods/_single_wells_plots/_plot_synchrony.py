from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import (
    _get_linear_phase,
    _get_synchrony,
    _get_synchrony_matrix,
)

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_synchrony_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot global synchrony."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    if rois is None:
        rois = [int(roi) for roi in data if roi.isdigit()]

    # if less than two rois input, can't calculate synchrony
    if len(rois) < 2:
        LOGGER.warning(
            "Insufficient ROIs selected for synchrony analysis. "
            "Please select at least two ROIs."
        )
        return None

    phase_dict: dict[str, list[float]] = {}
    for roi, roi_data in data.items():
        if int(roi) not in rois:
            continue
        if (
            not roi_data.dec_dff
            or not roi_data.peaks_dec_dff
            or len(roi_data.peaks_dec_dff) < 1
        ):
            continue
        frames = len(roi_data.dec_dff)
        peaks = np.array(roi_data.peaks_dec_dff)
        phase_dict[roi] = _get_linear_phase(frames, peaks)

    synchrony_matrix = _get_synchrony_matrix(phase_dict)

    if synchrony_matrix is None:
        return None

    linear_synchrony = _get_synchrony(synchrony_matrix)

    title = f"Global Synchrony (Median: {linear_synchrony:0.4f})"

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Synchrony index")

    ax.set_title(title)

    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])

    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.set_box_aspect(1)

    active_rois = list(phase_dict.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)
    widget.figure.tight_layout()
    widget.canvas.draw()


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
        x, y = map(int, np.round(sel.target))  # <-- Snap to nearest pixel center
        roi_x, roi_y = rois[x], rois[y]

        sel.annotation.set(
            text=f"ROI {roi_x} ↔ ROI {roi_y}\nvalue: {synchrony_matrix[y, x]:0.2f}",
            fontsize=8,
            color="black",
        )
        if roi_x.isdigit() and roi_y.isdigit():
            widget.roiSelected.emit([roi_x, roi_y])
