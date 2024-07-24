from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

if TYPE_CHECKING:
    from ._graph_widget_cond import _GraphWidget_cond
    from ._util import Peaks, ROIData

COUNT_INCREMENT = 2


# def get_trace(
#     roi_data: ROIData,
#     dff: bool,
#     photobleach_corrected: bool,
#     used_for_bleach_correction: bool,
# ) -> list[float] | None:
#     """Get the appropriate trace based on the flags."""
#     if used_for_bleach_correction:
#         trace = roi_data.use_for_bleach_correction
#         return trace[0] if trace is not None else None
#     elif dff and not photobleach_corrected:
#         return roi_data.dff
#     elif photobleach_corrected and not dff:
#         return roi_data.bleach_corrected_trace
#     else:
#         return roi_data.raw_trace


def compare_conditions(
    widget: _GraphWidget_cond,
    data: dict,
    x_axis: str | None | None,
    y_axis: str | None = None,
    colors: list[str] | None = None,
    amplitude: bool = False,
    frequency: bool = False,
    cell_size: bool = False,
    max_slope: bool = False,
    rise_time: bool = False,
    decay_time: bool = False,
    iei: bool = False,
    global_connectivity: bool = False,
    width: bool = False
) -> None:
    """Plot various types of traces."""
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Set the title
    title_parts = []
    if amplitude:
        title_parts.append("Average Amplitude")
    if frequency:
        title_parts.append("Average frequency")
    if cell_size:
        title_parts.append("Average cell size")
    if max_slope:
        title_parts.append("Average max slope")
    if rise_time:
        title_parts.append("Average rise time")
    if decay_time:
        title_parts.append("Average decay time")
    if global_connectivity:
        title_parts.append("Global connectivity")
    if iei:
        title_parts.append("Average Interevent Interval")
    if y_axis:
        title_parts.append(f"compared by {y_axis}")

    ax.set_title(" - ".join(title_parts))

    count = 0
    data_to_plot = {}

    for fov, fov_data in data.items():
        for roi in fov_data:
            roi_data = cast("ROIData", data[fov][roi])


            if x_axis == "Genotype":
                cond_to_plot = roi_data.condition_1
            elif x_axis == "Treatment":
                cond_to_plot = roi_data.condition_2
            print(f"        condition: {cond_to_plot}")
            if not data_to_plot.get(cond_to_plot):
                data_to_plot[cond_to_plot] = []
                count += COUNT_INCREMENT

            if amplitude:
                data_to_plot[cond_to_plot].append(roi_data.mean_amplitude)
                print(f"roi: {roi}, amplitude: {roi_data.mean_amplitude}")
            if frequency:
                data_to_plot[cond_to_plot].append(roi_data.frequency)

            if cell_size:
                data_to_plot[cond_to_plot].append(roi_data.cell_size)

            if max_slope:
                data_to_plot[cond_to_plot].append(roi_data.mean_max_slope)

            if rise_time:
                data_to_plot[cond_to_plot].append(roi_data.mean_raise_time)

            if decay_time:
                data_to_plot[cond_to_plot].append(roi_data.mean_decay_time)

            if iei:
                data_to_plot[cond_to_plot].append(roi_data.mean_iei)

            if global_connectivity:
                data_to_plot[cond_to_plot].append(roi_data.global_connectivity)

    groups = list(data_to_plot.keys())
    values = list(data_to_plot.values())
    # print(f"        group length: {len(groups)}, groups: {groups}")
    # print(f"        values length: {len(values)}, groups: {values[0]}")

    bp = ax.boxplot(values, labels=groups, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    widget.figure.legend([bp["boxes"][i] for i in range(len(groups))],
                         groups, loc='upper right')

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    # Add hover functionality using mplcursors
    # cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    # @cursor.connect("add")  # type: ignore [misc]
    # def on_add(sel: mplcursors.Selection) -> None:
    #     sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
    #     # emit the graph widget roiSelected signal
    #     if sel.artist.get_label():

    #         roi = cast(str, sel.artist.get_label().split(" ")[1])

    #         if roi.isdigit():
    #             widget.roiSelected.emit(roi)

    widget.canvas.draw()
