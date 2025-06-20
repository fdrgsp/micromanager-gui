from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors
import numpy as np

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import (
    _get_spike_synchrony,
    _get_spike_synchrony_matrix,
    _get_spikes_over_threshold,
)

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
) -> None:
    """Plot spike-based synchrony analysis.

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

    spike_trains = _get_spike_trains_from_rois(data, rois)
    if spike_trains is None or len(spike_trains) < 2:
        LOGGER.warning(
            "Insufficient spike data for synchrony analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        return

    lag = _get_lag(data, rois)
    if lag is None:
        LOGGER.warning("No valid lag value found for synchrony analysis.")
        return

    # Convert spike trains to spike data dict for correlation-based synchrony
    spike_data_dict = {
        roi_name: cast(list[float], spike_train.astype(float).tolist())
        for roi_name, spike_train in spike_trains.items()
    }

    # Use cross-correlation method for inferred spikes - better suited for
    # signal-like data that may have temporal artifacts from deconvolution
    synchrony_matrix = _get_spike_synchrony_matrix(
        spike_data_dict, method="cross_correlation", max_lag=lag
    )

    if synchrony_matrix is None:
        LOGGER.warning(
            "Failed to compute synchrony matrix. "
            "Ensure spike data is valid and contains sufficient ROIs."
        )
        widget.canvas.draw()
        return

    # Calculate global synchrony metric using spike-specific function
    global_synchrony = _get_spike_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Global Synchrony (Median: {global_synchrony:.4f})\n"
        f"(Thresholded Spike Data - Cross-Correlation Method)\n"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=1)),
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


def _get_lag(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
) -> int | None:
    """Get the lag value for synchrony form ROIData."""
    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]
    # use only the first roi since the burst parameters are the same for all ROIs
    roi_key = str(rois[0]) if rois else None
    if roi_key is None or roi_key not in roi_data_dict:
        LOGGER.warning("No valid ROIs found for synchrony analysis.")
        return None
    roi_data = roi_data_dict[roi_key]
    return roi_data.spikes_sync_cross_corr_lag


def _get_spike_trains_from_rois(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract spike trains from ROI data.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all

    Returns
    -------
        Dictionary mapping ROI names to binary spike arrays
    """
    spike_trains: dict[str, np.ndarray] = {}

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None

    for roi_key, roi_data in roi_data_dict.items():
        try:
            roi_id = int(roi_key)
            if roi_id not in rois or not roi_data.active:
                continue
        except ValueError:
            # Skip non-numeric ROI keys when rois filter is specified
            continue

        if (spike_probs := _get_spikes_over_threshold(roi_data)) is not None:
            # Convert spike probabilities to binary spike train
            spike_train = np.array(spike_probs) > 0.0
            if np.sum(spike_train) > 0:  # Only include ROIs with at least one spike
                spike_trains[roi_key] = spike_train

    return spike_trains if len(spike_trains) >= 2 else None


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
                    f"Spike Synchrony: {synchrony_matrix[y, x]:.3f}"
                ),
                fontsize=8,
                color="black",
            )
            if roi_x.isdigit() and roi_y.isdigit():
                widget.roiSelected.emit([roi_x, roi_y])
