from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from micromanager_gui._plate_viewer._util import (
    STIMULATION_MASK,
)

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData

DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    data: dict,
    rois: list[int] | None = None,
    with_rois: bool = False,
) -> None:
    """Visualize Stimulated area."""
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # get analysis path
    analysis_path = widget._plate_viewer.analysis_files_path
    if analysis_path is None:
        return

    # get the stimulation mask
    stimulation_mask_path = Path(analysis_path) / STIMULATION_MASK
    if not stimulation_mask_path.exists():
        return
    stim_mask = tifffile.imread(stimulation_mask_path)

    color_mapping = {0: "black", 1: "white"}  # 0: background, 1: stimulated area

    if with_rois:
        # get the labels file path
        labels_image_path = widget._plate_viewer.labels_path
        if labels_image_path is None:
            return

        stim, non_stim = _group_rois(data, rois)

        # open label image
        r = str(rois[0]) if rois is not None else "1"
        label_name = cast("ROIData", data[r]).well_fov_position
        if not label_name:
            return
        labels = tifffile.imread(Path(labels_image_path) / label_name)

        # create a color mapping for the labels
        labels_range = np.unique(labels[labels != 0])
        for roi in labels_range:
            if roi in stim:
                color_mapping[roi] = STIMULATED_COLOR
            elif roi in non_stim:
                color_mapping[roi] = NON_STIMULATED_COLOR
            else:
                color_mapping[roi] = DEFAULT_COLOR

        # plot the labels image with the color mapping
        unique_labels = np.unique(labels)
        colors = [color_mapping.get(lbl, DEFAULT_COLOR) for lbl in unique_labels]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            boundaries=np.append(unique_labels, unique_labels[-1] + 1),
            ncolors=len(colors),
        )
        ax.imshow(labels, cmap=cmap, norm=norm)

        # create and add legend
        legend_patches = [
            Patch(color="green", label="Stimulated ROIs"),
            Patch(color="magenta", label="Non-Stimulated ROIs"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),  # moves it above the plot (x, y)
            ncol=2,  # single row
            frameon=True,
            fontsize="small",
            edgecolor="black",
        )

    else:
        ax.imshow(stim_mask, cmap="gray")

    ax.axis("off")

    widget.figure.tight_layout()

    if with_rois:
        # Add hover functionality using mplcursors
        cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

        @cursor.connect("add")  # type: ignore [misc]
        def on_add(sel: mplcursors.Selection) -> None:
            roi_val = None
            # emit the graph widget roiSelected signal
            x, y = int(sel.target[0]), int(sel.target[1])
            if 0 <= y < stim_mask.shape[0] and 0 <= x < stim_mask.shape[1]:
                roi_val = str(labels[y, x]) if labels[y, x] > 0 else None
            if roi_val:
                sel.annotation.set(text=f"ROI {roi_val}", fontsize=8, color="yellow")
                sel.annotation.arrow_patch.set_color("yellow")
                sel.annotation.arrow_patch.set_alpha(1)  # arrow is visible
            else:
                sel.annotation.set_visible(False)  # hide annotation
                sel.annotation.arrow_patch.set_alpha(0)  # hide arrow
            # emit the graph widget roiSelected signal
            if roi_val and roi_val.isdigit():
                widget.roiSelected.emit(roi_val)

    widget.canvas.draw()


def _group_rois(data: dict, rois: list[int] | None) -> tuple[list[int], list[int]]:
    """To group the ROIs based on stimulated state."""
    stimulated_rois: list[int] = []
    non_stimulated_rois: list[int] = []

    for roi_key in data:
        if rois is not None and int(roi_key) not in rois:
            continue

        roi_data = cast("ROIData", data[roi_key])

        if roi_data.stimulated:
            stimulated_rois.append(int(roi_key))
        else:
            non_stimulated_rois.append(int(roi_key))

    return stimulated_rois, non_stimulated_rois
