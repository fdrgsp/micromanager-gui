from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.cm as cm
import mplcursors
import numpy as np
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import _SingleWellGraphWidget
    from micromanager_gui._plate_viewer._util import ROIData


def _generate_raster_plot(
    widget: _SingleWellGraphWidget,
    data: dict,
    rois: list[int] | None = None,
    amplitude_colors: bool = False,
) -> None:
    """Generate a raster plot."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    ax.set_title(
        "Raster Plot Colored by Amplitude" if amplitude_colors else "Raster Plot"
    )

    # initialize required lists and variables
    event_data: list[list[float]] = []
    colors: list[list[str]] = []
    rois_rec_time: list[float] = []
    total_frames: int = 0
    trace: list[float] | None = None

    # if amplitude colors are used, determine min/max amplitude range
    if amplitude_colors:
        min_amp, max_amp = float("inf"), float("-inf")

    # loop over the ROIData and get the peaks and their colors for each ROI
    for roi_key, roi_data in data.items():
        roi_id = int(roi_key)
        if rois is not None and roi_id not in rois:
            continue

        roi_data = cast("ROIData", roi_data)

        if not roi_data.peaks_dec_dff or not roi_data.peaks_amplitudes_dec_dff:
            continue

        # this is to then convert the x-axis frames to seconds
        if roi_data.total_recording_time_in_sec is not None:
            rois_rec_time.append(roi_data.total_recording_time_in_sec)

        # assuming all traces have the same number of frames
        if not total_frames and roi_data.raw_trace is not None:
            total_frames = len(roi_data.raw_trace)

        # store event data
        event_data.append(roi_data.peaks_dec_dff)

        if amplitude_colors:
            # calculate min and max amplitudes for color normalization
            min_amp = min(min_amp, min(roi_data.peaks_amplitudes_dec_dff))
            max_amp = max(max_amp, max(roi_data.peaks_amplitudes_dec_dff))
        else:
            # assign default color if not using amplitude-based coloring
            colors.append([f"C{roi_id - 1}"] * len(roi_data.peaks_dec_dff))

    # create the color palette for the raster plot
    if amplitude_colors:
        # multiply 0.5 to max_amp to make the color range more visible
        norm_amp_color = Normalize(vmin=min_amp, vmax=max_amp * 0.5)
        cmap = cm.viridis  # choose colormap
        # scalar mappable for colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm_amp_color)
        sm.set_array([])  # required for colorbar

        # assign colors based on amplitude
        colors = [
            [cmap(norm_amp_color(amp)) for amp in roi_data.peaks_amplitudes_dec_dff]
            for roi_data in (
                cast("ROIData", data[str(roi)]) for roi in rois or data.keys()
            )
            if roi_data.peaks_amplitudes_dec_dff
        ]

    # plot the raster plot
    ax.eventplot(event_data, colors=colors)

    # set the axis labels
    ax.set_ylabel("ROIs")
    # hide the y-axis labels and ticks
    ax.set_yticklabels([])
    ax.set_yticks([])

    # use the last trace to get total number of frames (they should all be the same)
    trace = roi_data.raw_trace
    update_time_axis(ax, rois_rec_time, trace)

    # add the colorbar if amplitude colors are used
    if amplitude_colors:
        cbar = widget.figure.colorbar(sm, ax=ax)
        cbar.set_label("Amplitude")

    widget.figure.tight_layout()

    # add hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        # emit the graph widget roiSelected signal
        if sel.artist.get_label():
            # sel.artist.get_label() returns a _child0
            child = int(sel.artist.get_label()[6:])
            roi = str(rois[child]) if rois is not None else str(child + 1)
            sel.annotation.set(text=f"ROI {roi}", fontsize=8, color="black")
            if roi.isdigit():
                widget.roiSelected.emit(roi)

    widget.canvas.draw()


def update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
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
