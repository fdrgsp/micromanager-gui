"""Spike-based synchrony analysis for network analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

from micromanager_gui._plate_viewer._util import (
    _get_synchrony_matrix,
    get_linear_phase,
    get_synchrony,
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
    spike_threshold: float = 0.0, # TODO: remove and add parameter for analysis
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
    spike_threshold: float
        Threshold for spike detection (default 0.1)
    time_window: float
        Time window for synchrony detection in seconds (default 0.1)
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    spike_trains = _get_spike_trains_from_rois(data, rois, spike_threshold)
    if spike_trains is None or len(spike_trains) < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient spike data for synchrony analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Get exposure time from data for temporal resolution
    exposure_time_ms = _get_exposure_time_from_data(data)
    # Use exposure time as the synchrony window (convert from ms to seconds)
    time_window = exposure_time_ms / 1000.0 if exposure_time_ms > 0 else 0.1

    synchrony_matrix = _calculate_spike_synchrony_matrix(spike_trains, time_window)

    if synchrony_matrix is None:
        ax.text(
            0.5,
            0.5,
            "Unable to calculate spike synchrony",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        widget.canvas.draw()
        return

    # Calculate global synchrony metric using existing function
    global_synchrony = get_synchrony(synchrony_matrix)
    if global_synchrony is None:
        global_synchrony = 0.0

    title = (
        f"Spike-based Median Global Synchrony ({global_synchrony:.3f})\n"
        f"threshold={spike_threshold:.1f}, window={time_window*1000:.1f}ms)\n"
    )

    img = ax.imshow(synchrony_matrix, cmap="viridis", vmin=0, vmax=1)
    cbar = widget.figure.colorbar(
        cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1)),
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


def _get_spike_trains_from_rois(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
    spike_threshold: float = 0.5,
) -> dict[str, np.ndarray] | None:
    """Extract spike trains from ROI data.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all
        spike_threshold: Threshold for spike detection

    Returns
    -------
        Dictionary mapping ROI names to binary spike arrays
    """
    spike_trains: dict[str, np.ndarray] = {}

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None

    for roi, roi_data in roi_data_dict.items():
        if int(roi) not in rois or not roi_data.active:
            continue

        if (spike_probs := roi_data.inferred_spikes) is not None:
            # Convert spike probabilities to binary spike train
            spike_train = np.array(spike_probs) > spike_threshold
            if np.sum(spike_train) > 0:  # Only include ROIs with at least one spike
                spike_trains[roi] = spike_train

    return spike_trains if len(spike_trains) >= 2 else None


def _get_exposure_time_from_data(roi_data_dict: dict[str, ROIData]) -> float:
    """Extract exposure time from ROI data.

    Attempts to estimate exposure time from total recording time and number of frames.
    Falls back to a default if no temporal information is available.

    Args:
        roi_data_dict: Dictionary of ROI data

    Returns
    -------
        Exposure time in milliseconds
    """
    for roi_data in roi_data_dict.values():
        if (
            roi_data.total_recording_time_sec is not None
            and roi_data.inferred_spikes is not None
            and len(roi_data.inferred_spikes) > 0
        ):
            total_time_sec = roi_data.total_recording_time_sec
            n_frames = len(roi_data.inferred_spikes)
            # Calculate frame interval (exposure + any delay between frames)
            frame_interval_ms = (total_time_sec * 1000) / n_frames
            return frame_interval_ms

    # Default fallback (100ms frame interval = 10 Hz)
    return 100.0


def _calculate_spike_synchrony_matrix(
    spike_trains: dict[str, np.ndarray],
    time_window: float,
) -> np.ndarray | None:
    """Calculate pairwise spike synchrony matrix using PLV approach.

    Converts spike trains to instantaneous phases and uses the existing
    PLV-based synchrony calculation for consistency with calcium analysis.

    Args:
        spike_trains: Dictionary of binary spike trains
        time_window: Time window for synchrony detection (seconds)

    Returns
    -------
        Square matrix of synchrony values using PLV method
    """
    roi_names = list(spike_trains.keys())
    n_rois = len(roi_names)

    if n_rois < 2:
        return None

    # Convert spike trains to phase representations
    phase_dict = {}

    for roi_name, spike_train in spike_trains.items():
        # Find spike indices
        spike_indices = np.where(spike_train)[0]

        if len(spike_indices) == 0:
            # No spikes - assign zero phase
            phase_dict[roi_name] = [0.0] * len(spike_train)
        else:
            # Use existing get_linear_phase function to create phase from spikes
            phase_dict[roi_name] = get_linear_phase(len(spike_train), spike_indices)

    # Use existing PLV-based synchrony matrix calculation
    synchrony_matrix = _get_synchrony_matrix(phase_dict)

    return synchrony_matrix


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
