from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import mplcursors
import numpy as np
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from skimage.measure import find_contours

from micromanager_gui._plate_viewer._logger._pv_logger import LOGGER
from micromanager_gui._plate_viewer._util import (
    MWCM,
    STIMULATION_MASK,
    _get_spikes_over_threshold,
    equation_from_str,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"
P1 = 5
P2 = 100


def _plot_evoked_experiment_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    stimulated_area: bool = False,
    with_rois: bool = False,
    stimulated: bool = False,
    with_peaks: bool = False,
) -> None:
    """Plot evoked experiment data."""
    if stimulated_area or with_rois:
        _visualize_stimulated_area(widget, data, rois, with_rois, stimulated_area)

    if with_peaks:
        _plot_stimulated_vs_non_stimulated_roi_amp(widget, data, rois, with_peaks)

    else:
        _plot_stim_or_not_stim_peaks_amplitude(widget, data, rois, stimulated)


def _plot_stim_or_not_stim_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    stimulated: bool = False,
) -> None:
    """Visualize stimulated peak amplitudes per ROI per stimulation parameters."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # get analysis path
    analysis_path = widget._plate_viewer.analysis_path
    if analysis_path is None:
        return

    pulse: str = ""
    led_power_equation_str: str = ""
    led_power_equation: Callable | None = None
    # {power_pulselength: [(ROI1, amp1), (ROI2, amp2), ...]}
    # e.g. {"10_100": [(1, 0.5), (2, 0.6)], "20_200": [(3, 0.7)]}
    # or {"1mW/cm²_100": [(1, 0.5), (2, 0.6)], ...}
    power_pulse_and_amps: dict[str, list[tuple[int, float]]] = {}
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        if not led_power_equation_str:
            led_power_equation_str = roi_data.led_power_equation or ""
            led_power_equation = equation_from_str(led_power_equation_str)

        from micromanager_gui._plate_viewer._util import (
            get_stimulated_amplitudes_from_roi_data,
        )

        # Compute amplitudes on-demand
        amps_stim, amps_non_stim = get_stimulated_amplitudes_from_roi_data(
            roi_data, led_power_equation=led_power_equation
        )

        amplitudes = amps_stim if stimulated else amps_non_stim

        if not amplitudes:
            continue

        if not pulse:
            pulse = next(iter(amplitudes.keys())).split("_")[1]

        # power_pulse is f"{power}_{pulse_len}"
        for power_pulse, amp_list in amplitudes.items():
            for amp_val in amp_list:
                power_pulse_and_amps.setdefault(power_pulse, []).append(
                    (int(roi_key), amp_val)
                )

    # sort the power_pulse_and_amps dictionary by power
    power_pulse_and_amps = dict(
        sorted(power_pulse_and_amps.items(), key=lambda x: extract_leading_number(x[0]))
    )
    x_axis_label = ""
    # rename as power_pulse = "10_100" -> "10% 100ms"
    renamed_power_pulse_and_amps: dict[str, list[tuple[int, float]]] = {}
    # e.g. {"10% 100ms": [(1, 0.5), (2, 0.6)], "20% 200ms": [(3, 0.7)]}
    for power_pulse in power_pulse_and_amps:
        power_pulse_spit = power_pulse.split("_")
        # x_name = f"{power_pulse_spit[0]}% {power_pulse_spit[1]}ms"
        x_name = f"{power_pulse_spit[0]}"
        if MWCM in x_name:  # e.g. "30mW/cm²"
            x_name = x_name.split(MWCM)[0]
            if not x_axis_label:
                x_axis_label = "Irradiance (mW/cm²)"
        elif not x_axis_label:
            x_axis_label = "Power (%)"
        renamed_power_pulse_and_amps[x_name] = power_pulse_and_amps[power_pulse]

    # plot each power_pulse group as a scatter
    all_artists: list = []
    all_metadata: list[tuple[list[int], list[float]]] = []
    for power_pulse_label, roi_amp_pairs in renamed_power_pulse_and_amps.items():
        amps = np.array([amp for _, amp in roi_amp_pairs])
        rois_ = [roi for roi, _ in roi_amp_pairs]

        mean_amp = np.mean(amps)
        std_amp = np.std(amps)
        n = len(amps)
        error = std_amp / np.sqrt(n)
        errorbar = ax.errorbar(
            power_pulse_label,
            mean_amp,
            yerr=error,
            fmt="o",
            capsize=5,
            label=power_pulse_label,
        )
        all_artists.append(errorbar)
        all_metadata.append((rois_, [mean_amp]))
        scatter = ax.scatter(
            [power_pulse_label] * len(amps),
            amps,
            label=power_pulse_label,
            color="lightgray",
            s=30,
        )
        all_artists.append(scatter)
        all_metadata.append((rois_, cast(list[float], amps.tolist())))

    _add_hover_to_stimulated_amp_plot(widget, all_artists, all_metadata)

    ax.set_ylabel("Mean Amplitude ± SEM")
    ax.set_xlabel(x_axis_label)
    if x_axis_label == "Irradiance (mW/cm²)":
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    title = (
        "Stimulated" if stimulated else "Non-Stimulated"
    ) + " Calcium Peaks Mean Amplitudes ± SEM"
    title += "\n("
    title += "Per LED Intensity - "
    if pulse:
        title += f"{pulse} ms pulses - "
    title += "Deconvolved ΔF/F)"
    ax.set_title(title)
    widget.figure.tight_layout()
    widget.canvas.draw()


def extract_leading_number(key: str) -> float:
    """Extract leading number from key (before '_'), stripping units if present."""
    if match := re.match(r"(\d+(?:\.\d+)?)", key.split("_")[0]):
        return float(match[1])
    raise ValueError(f"Could not extract a valid number from key: {key}")


def _add_hover_to_stimulated_amp_plot(
    widget: _SingleWellGraphWidget,
    artists: list,
    metadata: list[tuple[list[int], list[float]]],
) -> None:
    """Add hover tooltips to amplitude scatter plot points."""
    cursor = mplcursors.cursor(artists, hover=mplcursors.HoverMode.Transient)

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
    analysis_path = widget._plate_viewer.analysis_path
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
        ax.imshow(stim_mask, cmap="gray", clim=(0, 1))

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
    labels_image_path = widget._plate_viewer.pv_labels_path
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
    _add_hover_functionality_plot_stim_roi(ax, widget, labels, stim_mask)


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


def _add_hover_functionality_plot_stim_roi(
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
        if roi_val and "ROI" in roi_val:
            sel.annotation.set(text=f"ROI {roi_val}", fontsize=8, color="yellow")
            sel.annotation.arrow_patch.set_color("yellow")
            sel.annotation.arrow_patch.set_alpha(1)  # arrow is visible
        else:
            sel.annotation.set_visible(False)  # hide annotation
            sel.annotation.arrow_patch.set_alpha(0)  # hide arrow
        if roi_val and roi_val.isdigit():
            widget.roiSelected.emit(roi_val)


def _plot_stimulated_vs_non_stimulated_roi_amp(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    with_peaks: bool = False,
) -> None:
    """Plot dec ΔF/F traces with global percentile normalization (5th-100th)."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    trace: list[float] | None = None
    rois_rec_time: list[float] = []

    # Filter and sort ROIs: non-stimulated first
    sorted_items = sorted(
        [
            (roi_key, roi_data)
            for roi_key, roi_data in data.items()
            if roi_data.active
            and roi_data.dec_dff is not None
            and roi_data.peaks_dec_dff is not None
            and (rois is None or int(roi_key) in rois)
        ],
        key=lambda item: item[1].stimulated,
    )

    # gather all dec_dff values from included ROIs
    all_values: list[float] = []
    for _, roi_data in sorted_items:
        if roi_data.dec_dff is None:
            continue
        all_values.extend(roi_data.dec_dff)

    # compute nth and nth percentiles globally
    if all_values:
        percentiles = np.percentile(all_values, [P1, P2])
        p1, p2 = float(percentiles[0]), float(percentiles[1])
    else:
        p1, p2 = 0.0, 1.0

    stimulations_frames_and_powers: dict[str, int] = {}
    # plot each ROI trace with normalized and vertically offset values
    for count, (roi_key, roi_data) in enumerate(sorted_items):
        if roi_data.dec_dff is None:
            continue
        if not stimulations_frames_and_powers:
            stimulations_frames_and_powers = (
                roi_data.stimulations_frames_and_powers or {}
            )
        trace = _normalize_trace_percentile(roi_data.dec_dff, p1, p2)
        offset = count * 1.1
        trace_offset = np.array(trace) + offset

        if (ttime := roi_data.total_recording_time_sec) is not None:
            rois_rec_time.append(ttime)

        color = STIMULATED_COLOR if roi_data.stimulated else NON_STIMULATED_COLOR
        ax.plot(trace_offset, label=f"ROI {roi_key}", color=color)

        if with_peaks and roi_data.peaks_dec_dff is not None:
            peaks_indices = [int(p) for p in roi_data.peaks_dec_dff]
            ax.plot(
                peaks_indices,
                np.array(trace)[peaks_indices] + offset,
                "x",
                color="k",
            )

    # plot the stimulation frames as vertical lines
    for frame in stimulations_frames_and_powers:
        ax.axvline(x=float(frame) - 0.5, color="blue", linestyle="--", alpha=0.5)

    ax.set_title(
        "Stimulated vs Non-Stimulated ROIs Calcium Traces \n"
        "(Normalized Deconvolved ΔF/F)"
    )
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel("ROIs")

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

    _update_time_axis(ax, rois_rec_time, trace)

    _add_hover_functionality_stim_vs_non_stim(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _normalize_trace_percentile(
    trace: list[float], p1: float, p2: float
) -> list[float]:
    """Normalize a trace using the global 5th and 100th percentiles."""
    tr = np.array(trace)
    denom = p2 - p1
    if denom == 0:
        return cast(list[float], np.zeros_like(tr).tolist())
    normalized = (tr - p1) / denom
    normalized = np.clip(normalized, 0, 1)  # ensure values in [0, 1]
    return cast(list[float], normalized.tolist())


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


def _add_hover_functionality_stim_vs_non_stim(
    ax: Axes, widget: _SingleWellGraphWidget
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for peaks or other elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi = cast(str, label.split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


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
        LOGGER.warning(
            "No active ROIs with inferred spikes found for stimulated/non-stimulated "
            "analysis."
        )
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
        ax.axvline(
            x=float(frame) - 0.5, color="blue", linestyle="--", alpha=0.7, linewidth=2
        )

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

    # Update time axis using the utility function
    _update_time_axis_spike_traces(ax, rois_rec_time, sample_trace)

    widget.figure.tight_layout()
    active_roi_ids = [int(roi_key) for roi_key, _ in sorted_items]
    _add_hover_functionality_spike_traces(ax, widget, active_roi_ids)
    widget.canvas.draw()


def _add_hover_functionality_spike_traces(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using mplcursors for spike traces."""
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


def _update_time_axis_spike_traces(
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
