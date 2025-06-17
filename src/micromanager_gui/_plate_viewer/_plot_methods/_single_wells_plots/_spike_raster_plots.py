from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import mplcursors
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import _get_spikes_over_threshold

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import _SingleWellGraphWidget
    from micromanager_gui._plate_viewer._util import ROIData

# Colors for stimulated vs non-stimulated traces
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _generate_spike_raster_plot(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a spike raster plot using thresholded spike data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    ax.set_title(
        "Inferred Spikes Raster Plot Colored by Amplitude"
        if amplitude_colors
        else "Inferred Spikes Raster Plot"
    )

    # initialize required lists and variables
    event_data: list[list[int]] = []
    colors: list = []  # Colors for eventplot (can be strings or tuples)
    rois_rec_time: list[float] = []
    total_frames: int = 0

    # if amplitude colors are used, determine min/max amplitude range
    min_amp, max_amp = (float("inf"), float("-inf")) if amplitude_colors else (0, 0)

    active_rois = []
    # loop over the ROIData and get the spike events for each ROI
    for roi_key, roi_data in data.items():
        roi_id = int(roi_key)
        if rois is not None and roi_id not in rois:
            continue

        # Get thresholded spikes
        thresholded_spikes = _get_spikes_over_threshold(roi_data)
        if not thresholded_spikes:
            continue

        # Find spike event times (indices where spike values are above threshold)
        spike_times = []
        spike_amplitudes = []

        for i, spike_val in enumerate(thresholded_spikes):
            if spike_val > 0:  # Above threshold
                spike_times.append(i)
                spike_amplitudes.append(spike_val)

        if not spike_times:
            continue

        # store the active ROIs
        active_rois.append(roi_id)

        # convert the x-axis frames to seconds
        if roi_data.total_recording_time_sec is not None:
            rois_rec_time.append(roi_data.total_recording_time_sec)

        # assuming all traces have the same number of frames
        if not total_frames and roi_data.raw_trace is not None:
            total_frames = len(roi_data.raw_trace)

        # store event data (spike times)
        event_data.append(spike_times)

        if amplitude_colors and spike_amplitudes:
            # calculate min and max amplitudes for color normalization
            min_amp = min(min_amp, min(spike_amplitudes))
            max_amp = max(max_amp, max(spike_amplitudes))
        else:
            # assign default color if not using amplitude-based coloring
            colors.append(f"C{roi_id - 1}")

    if not event_data:
        LOGGER.warning(
            "No spike data available for the selected ROIs. "
            "Please check the data or ROI selection."
        )
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # create the color palette for the raster plot
    if amplitude_colors:
        _generate_spike_amplitude_colors(data, rois, min_amp, max_amp, colors)

    # plot the raster plot
    ax.eventplot(event_data, colors=colors)

    # set the axis labels
    ax.set_ylabel("ROIs")
    ax.set_yticklabels([])
    ax.set_yticks([])

    # use any trace to get total number of frames (they should all be the same)
    sample_trace = None
    for roi_data in data.values():
        if roi_data.raw_trace is not None:
            sample_trace = roi_data.raw_trace
            break

    _update_time_axis(ax, rois_rec_time, sample_trace)

    # add the colorbar if amplitude colors are used
    if amplitude_colors and colorbar:
        # Use the same logic as in color generation
        vmax = min_amp + (max_amp - min_amp) * 0.6  # Use 50% of the range
        cbar = widget.figure.colorbar(
            cm.ScalarMappable(norm=Normalize(vmin=min_amp, vmax=vmax), cmap="viridis"),
            ax=ax,
        )
        cbar.set_label("Spike Amplitude")

    widget.figure.tight_layout()
    _add_hover_functionality(ax, widget, active_rois)
    widget.canvas.draw()


def _generate_spike_amplitude_colors(
    data: dict[str, ROIData],
    rois: list[int] | None,
    min_amp: float,
    max_amp: float,
    colors: list,
) -> None:
    """Assign colors based on individual spike amplitudes for raster plot."""
    # Always use a reduced range to make yellow colors more visible
    # Use the midpoint between min and max as vmax for better color distribution
    vmax = min_amp + (max_amp - min_amp) * 0.5  # Use 50% of the range
    norm_amp_color = Normalize(vmin=min_amp, vmax=vmax)
    cmap = colormaps.get_cmap("viridis")

    for roi in rois or data.keys():
        roi_data = data[str(roi)]
        if thresholded_spikes := _get_spikes_over_threshold(roi_data):
            # Get individual spike amplitudes and create colors for each spike event
            spike_colors = []
            for spike_val in thresholded_spikes:
                if spike_val > 0:  # This is a spike event
                    # Color each spike based on its individual amplitude
                    color = cmap(norm_amp_color(spike_val))
                    spike_colors.append(color)

            if spike_colors:
                colors.append(spike_colors)


def _add_hover_functionality(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for valid ROI elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi_parts = label.split(" ")
            if len(roi_parts) > 1 and roi_parts[1].isdigit():
                widget.roiSelected.emit(roi_parts[1])
        else:
            # For raster plots, map the position to an ROI
            if hasattr(sel, "target") and active_rois:
                with contextlib.suppress(ValueError, AttributeError, IndexError):
                    y_pos = int(sel.target[1])  # Get y-coordinate (ROI index)
                    if 0 <= y_pos < len(active_rois):
                        roi_id = active_rois[y_pos]
                        hover_text = f"ROI {roi_id}"
                        sel.annotation.set(text=hover_text, fontsize=8, color="black")
                        widget.roiSelected.emit(str(roi_id))
                        return
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update the x-axis to show time in seconds if recording time is available."""
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    # get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # get total number of frames from the trace
    total_frames = len(trace) if trace is not None else 1
    # compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _plot_stimulated_vs_non_stimulated_spike_traces(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot thresholded spike traces: green=stimulated, magenta=non-stimulated."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    rois_rec_time: list[float] = []
    sample_trace = None

    # Filter and sort ROIs: non-stimulated first, then stimulated
    sorted_items = sorted(
        [
            (roi_key, roi_data)
            for roi_key, roi_data in data.items()
            if roi_data.active
            and roi_data.inferred_spikes is not None
            and roi_data.inferred_spikes_threshold is not None
            and (rois is None or int(roi_key) in rois)
        ],
        key=lambda item: item[1].stimulated,
    )

    if not sorted_items:
        ax.text(
            0.5,
            0.5,
            "No spike data available for stimulated/non-stimulated analysis",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Get stimulation frames from first ROI
    stimulations_frames_and_powers: dict[str, int] = {}

    # Plot each ROI trace with thresholded spikes and vertical offset
    for count, (roi_key, roi_data) in enumerate(sorted_items):
        if (
            roi_data.inferred_spikes is None
            or roi_data.inferred_spikes_threshold is None
        ):
            continue

        if not stimulations_frames_and_powers:
            stimulations_frames_and_powers = (
                roi_data.stimulations_frames_and_powers or {}
            )

        # Get thresholded spikes (values above threshold, 0 otherwise)
        thresholded_spikes = _get_spikes_over_threshold(roi_data)
        if not thresholded_spikes:
            continue

        # Store a sample trace for time axis update
        if sample_trace is None:
            sample_trace = thresholded_spikes

        # Create vertical offset for each ROI
        offset = count * 1.1
        trace_offset = np.array(thresholded_spikes) + offset

        if (ttime := roi_data.total_recording_time_sec) is not None:
            rois_rec_time.append(ttime)

        # Color based on stimulation status
        color = STIMULATED_COLOR if roi_data.stimulated else NON_STIMULATED_COLOR
        ax.plot(trace_offset, label=f"ROI {roi_key}", color=color, linewidth=1.5)

    # Plot stimulation frames as vertical lines
    for frame in stimulations_frames_and_powers:
        ax.axvline(x=int(frame), color="blue", linestyle="--", alpha=0.7, linewidth=2)

    ax.set_title(
        "Stimulated vs Non-Stimulated ROIs Spike Traces\n"
        "(Thresholded Inferred Spikes)"
    )
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel("ROIs")

    # Create legend
    legend_patches = [
        Patch(facecolor=STIMULATED_COLOR, label="Stimulated ROIs"),
        Patch(facecolor=NON_STIMULATED_COLOR, label="Non-Stimulated ROIs"),
        Patch(facecolor="blue", label="Stimulation Pulse"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        frameon=True,
        fontsize="small",
        edgecolor="black",
        facecolor="white",
    )

    # Update time axis using the same utility function
    _update_time_axis(ax, rois_rec_time, sample_trace)

    widget.figure.tight_layout()
    _add_hover_functionality(ax, widget, [int(roi_key) for roi_key, _ in sorted_items])
    widget.canvas.draw()
