from __future__ import annotations

from typing import TYPE_CHECKING

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
    data: dict[str, ROIData],
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
    min_amp, max_amp = (float("inf"), float("-inf")) if amplitude_colors else (0, 0)

    # loop over the ROIData and get the peaks and their colors for each ROI
    for roi_key, roi_data in data.items():
        roi_id = int(roi_key)
        if rois is not None and roi_id not in rois:
            continue

        if not roi_data.peaks_dec_dff or not roi_data.peaks_amplitudes_dec_dff:
            continue

        # convert the x-axis frames to seconds
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
        _generate_amplitude_colors(data, rois, min_amp, max_amp, colors)

    # plot the raster plot
    ax.eventplot(event_data, colors=colors)

    # set the axis labels
    ax.set_ylabel("ROIs")
    ax.set_yticklabels([])
    ax.set_yticks([])

    # use the last trace to get total number of frames (they should all be the same)
    trace = roi_data.raw_trace
    _update_time_axis(ax, rois_rec_time, trace)

    # add the colorbar if amplitude colors are used
    if amplitude_colors:
        cbar = widget.figure.colorbar(
            cm.ScalarMappable(
                norm=Normalize(vmin=min_amp, vmax=max_amp * 0.5), cmap=cm.viridis
            ),
            ax=ax,
        )
        cbar.set_label("Amplitude")

    widget.figure.tight_layout()
    _add_hover_functionality(ax, widget, rois)
    widget.canvas.draw()


def _generate_amplitude_colors(
    data: dict[str, ROIData],
    rois: list[int] | None,
    min_amp: float,
    max_amp: float,
    colors: list[list[str]],
) -> None:
    """Assign colors based on amplitude for raster plot."""
    norm_amp_color = Normalize(vmin=min_amp, vmax=max_amp * 0.5)
    cmap = cm.viridis
    for roi in rois or data.keys():
        roi_data = data[str(roi)]
        if roi_data.peaks_amplitudes_dec_dff:
            colors.append(
                [cmap(norm_amp_color(amp)) for amp in roi_data.peaks_amplitudes_dec_dff]
            )


def _add_hover_functionality(
    ax: Axes, widget: _SingleWellGraphWidget, rois: list[int] | None
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        if sel.artist.get_label():
            child = int(sel.artist.get_label()[6:])
            roi = str(rois[child]) if rois is not None else str(child + 1)
            sel.annotation.set(text=f"ROI {roi}", fontsize=8, color="black")
            if roi.isdigit():
                widget.roiSelected.emit(roi)


def _update_time_axis(
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
