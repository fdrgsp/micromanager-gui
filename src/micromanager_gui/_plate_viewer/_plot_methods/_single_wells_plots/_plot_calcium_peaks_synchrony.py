from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import (
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_calcium_peaks_events_from_rois,
)

if TYPE_CHECKING:
    from matplotlib.image import AxesImage

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_peak_event_synchrony_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot peak event-based synchrony analysis.

    Parameters
    ----------
    widget: _SingleWellGraphWidget
        widget to plot on
    data: dict[str, ROIData]
        Dictionary of ROI data
    rois: list[int] | None
        List of ROI indices to include, None for all
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    peak_trains = _get_calcium_peaks_events_from_rois(data, rois)
    if peak_trains is None or len(peak_trains) < 2:
        LOGGER.warning(
            "Insufficient peak data for synchrony analysis. "
            "Ensure at least two ROIs with calcium peaks are selected."
        )
        return

    jit = _get_jit(data, rois)
    if jit is None:
        LOGGER.warning("No valid jitter window value found for synchrony analysis.")
        return

    # Convert peak trains to peak event data dict for correlation-based synchrony
    peak_event_data_dict = {
        roi_name: cast(list[float], peak_train.astype(float).tolist())
        for roi_name, peak_train in peak_trains.items()
    }

    # Use jitter window method for calcium peaks - better suited for discrete
    # events with inherent timing uncertainty due to biology and frame rate limits
    synchrony_matrix = _get_calcium_peaks_event_synchrony_matrix(
        peak_event_data_dict, method="jitter_window", jitter_window=jit
    )

    if synchrony_matrix is None:
        LOGGER.warning(
            "Failed to calculate synchrony matrix. "
            "Ensure peak event data is valid and contains sufficient data."
        )
        return

    # Calculate global synchrony metric using peak event-specific function
    global_synchrony = _get_calcium_peaks_event_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Calcium Peaks Events - Jitter Window Method)\n"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
        ax=ax,
    )
    cbar.set_label("Peak Event Synchrony Index")

    ax.set_title(title)
    ax.set_ylabel("ROI")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel("ROI")
    ax.set_xticklabels([])
    ax.set_xticks([])

    active_rois = list(peak_trains.keys())
    _add_hover_functionality(img, widget, active_rois, synchrony_matrix)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_jit(roi_data_dict: dict[str, ROIData], rois: list[int] | None) -> int | None:
    """Get the jitter window value for synchrony form ROIData."""
    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]
    # use only the first roi since the burst parameters are the same for all ROIs
    roi_key = str(rois[0]) if rois else None
    if roi_key is None or roi_key not in roi_data_dict:
        LOGGER.warning("No valid ROIs found for synchrony analysis.")
        return None
    roi_data = roi_data_dict[roi_key]
    return roi_data.calcium_sync_jitter_window


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
                text=(
                    f"ROI {roi_x} ↔ ROI {roi_y}\n"
                    f"Peak Event Synchrony: {synchrony_matrix[y, x]:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            if roi_x.isdigit() and roi_y.isdigit():
                widget.roiSelected.emit([roi_x, roi_y])
