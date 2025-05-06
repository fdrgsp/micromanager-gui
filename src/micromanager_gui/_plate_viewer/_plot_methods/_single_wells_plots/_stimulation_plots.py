from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from skimage.measure import find_contours

from micromanager_gui._plate_viewer._util import STIMULATION_MASK

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"


def _plot_stimulated_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Visualize stimulated peak amplitudes per ROI per stimulation condition."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # get analysis path
    analysis_path = widget._plate_viewer.analysis_files_path
    if analysis_path is None:
        return

    # {power_pulselength: [(ROI1, amp1), (ROI2, amp2), ...]}
    # e.g. {"10_100": [(1, 0.5), (2, 0.6)], "20_200": [(3, 0.7)]}
    power_pulse_and_amps: dict[str, list[tuple[int, float]]] = {}
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        if not roi_data.stimulated or roi_data.amplitudes_stimulated_peaks is None:
            continue
        # power_pulse is f"{power}_{pulse_len}"
        for power_pulse, amp_list in roi_data.amplitudes_stimulated_peaks.items():
            for amp_val in amp_list:
                power_pulse_and_amps.setdefault(power_pulse, []).append(
                    (int(roi_key), amp_val)
                )

    # sort the power_pulse_and_amps dictionary by power
    power_pulse_and_amps = dict(
        sorted(power_pulse_and_amps.items(), key=lambda x: int(x[0].split("_")[0]))
    )

    # rename as power_pulse = "10_100" -> "10% 100ms"
    renamed_power_pulse_and_amps: dict[str, list[tuple[int, float]]] = {}
    # e.g. {"10% 100ms": [(1, 0.5), (2, 0.6)], "20% 200ms": [(3, 0.7)]}
    for power_pulse in power_pulse_and_amps:
        power_pulse_spit = power_pulse.split("_")
        x_name = f"{power_pulse_spit[0]}% {power_pulse_spit[1]}ms pulse"
        renamed_power_pulse_and_amps[x_name] = power_pulse_and_amps[power_pulse]

    # plot each power_pulse group as a scatter
    all_artists = []
    all_metadata = []
    for power_pulse_label, roi_amp_pairs in renamed_power_pulse_and_amps.items():
        rois_, amps = zip(*roi_amp_pairs)

        scatter = ax.scatter(
            [power_pulse_label] * len(amps), amps, label=power_pulse_label
        )
        all_artists.append(scatter)
        all_metadata.append((rois_, amps))

    _add_hover_to_amplitude_plot(ax, widget, all_artists, all_metadata)

    ax.set_ylabel("Amplitude")
    ax.set_title("Stimulated Peak Amplitudes per Power/Pulse Length Condition")
    ax.tick_params(axis="x", rotation=45)
    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_to_amplitude_plot(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    artists: list,
    metadata: list[tuple[list[int], list[float]]],
) -> None:
    """Add hover tooltips to amplitude scatter plot points."""
    cursor = mplcursors.cursor(artists, hover=True)

    @cursor.connect("add")  # type: ignore
    def on_add(sel: mplcursors.Selection) -> None:
        artist = sel.artist
        index = sel.index
        group_index = artists.index(artist)
        rois_, amps = metadata[group_index]
        roi = rois_[index]
        amp_val = amps[index]

        sel.annotation.set(
            text=f"ROI {roi}\nAmp: {amp_val:.3f}", fontsize=8, color="black"
        )
        sel.annotation.arrow_patch.set_alpha(0.5)

        widget.roiSelected.emit(str(roi))


def _visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    with_rois: bool = False,
    stimulated_area: bool = False,
) -> None:
    """Visualize Stimulated area."""
    # clear the figure
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

    if with_rois:
        _plot_stimulated_rois(ax, widget, data, rois, stim_mask, stimulated_area)
    else:
        ax.imshow(stim_mask, cmap="gray")

    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_stimulated_rois(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None,
    stim_mask: np.ndarray,
    with_stimulated_area: bool,
) -> None:
    """Plot the ROIs with stimulated and non-stimulated areas."""
    # get the labels file path
    labels_image_path = widget._plate_viewer.labels_path
    if labels_image_path is None:
        return

    stim, non_stim = _group_rois(data, rois)

    # open label image
    r = str(rois[0]) if rois is not None else "1"
    label_name = f"{data[r].well_fov_position}.tif"
    if not label_name:
        return
    labels = tifffile.imread(Path(labels_image_path) / label_name)

    # create a color mapping for the labels
    color_mapping = _generate_color_mapping(labels, stim, non_stim)

    # plot the labels image with the color mapping
    unique_labels = np.unique(labels)
    colors = [color_mapping.get(lbl, DEFAULT_COLOR) for lbl in unique_labels]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=np.append(unique_labels, unique_labels[-1] + 1),
        ncolors=len(colors),
    )

    if with_stimulated_area:
        stim_area_contours = find_contours(stim_mask.astype(float), level=0.5)
        for contour in stim_area_contours:
            ax.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=1)
    ax.imshow(labels, cmap=cmap, norm=norm)

    _add_legend(ax)
    _add_hover_functionality(ax, widget, labels, stim_mask)


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


def _generate_color_mapping(
    labels: np.ndarray, stim: list[int], non_stim: list[int]
) -> dict[int, str]:
    """Generate a color mapping for the labels."""
    color_mapping = {0: "black", 1: "white"}  # 0: background, 1: stimulated area
    labels_range = np.unique(labels[labels != 0])
    for roi in labels_range:
        if roi in stim:
            color_mapping[roi] = STIMULATED_COLOR
        elif roi in non_stim:
            color_mapping[roi] = NON_STIMULATED_COLOR
        else:
            color_mapping[roi] = DEFAULT_COLOR
    return color_mapping


def _add_legend(ax: Axes) -> None:
    """Add legend to the plot."""
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


def _add_hover_functionality(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    labels: np.ndarray,
    stim_mask: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        roi_val = None
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
        if roi_val and roi_val.isdigit():
            widget.roiSelected.emit(roi_val)
