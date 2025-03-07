from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

from ._util import (
    DEC_DFF,
    DEC_DFF_AMPLITUDE,
    DEC_DFF_AMPLITUDE_VS_FREQUENCY,
    DEC_DFF_FREQUENCY,
    DEC_DFF_IEI,
    DEC_DFF_NORMALIZED,
    DEC_DFF_NORMALIZED_WITH_PEAKS,
    DEC_DFF_WITH_PEAKS,
    DFF,
    DFF_NORMALIZED,
    NORMALIZED_TRACES,
    RAW_TRACES,
    STIMULATED_AREA,
    STIMULATED_ROIS,
)

if TYPE_CHECKING:
    from ._graph_widgets import _MultilWellGraphWidget, _SingleWellGraphWidget
    from ._util import ROIData

COUNT_INCREMENT = 1

SINGLE_WELL_GRAPHS_OPTIONS: dict[str, dict[str, bool]] = {
    RAW_TRACES: {},
    NORMALIZED_TRACES: {"normalize": True},
    DFF: {"dff": True},
    DFF_NORMALIZED: {"dff": True, "normalize": True},
    DEC_DFF: {"dec": True},
    DEC_DFF_WITH_PEAKS: {"dec": True, "with_peaks": True},
    DEC_DFF_NORMALIZED: {"dec": True, "normalize": True},
    DEC_DFF_NORMALIZED_WITH_PEAKS: {"dec": True, "normalize": True, "with_peaks": True},
    DEC_DFF_AMPLITUDE: {"dec": True, "amp": True},
    DEC_DFF_FREQUENCY: {"dec": True, "freq": True},
    DEC_DFF_AMPLITUDE_VS_FREQUENCY: {"dec": True, "amp": True, "freq": True},
    DEC_DFF_IEI: {"dec": True, "iei": True},
    STIMULATED_AREA: {"with_rois": False},
    STIMULATED_ROIS: {"with_rois": True},
}

MULTI_WELL_GRAPHS_OPTIONS: dict[str, dict[str, bool]] = {
    DEC_DFF_AMPLITUDE_VS_FREQUENCY: {"amp": True, "freq": True},
    DEC_DFF_AMPLITUDE: {"amp": True},
    DEC_DFF_FREQUENCY: {"freq": True},
    DEC_DFF_IEI: {"iei": True},
}


# ------------------------------SINGLE-WELL PLOTTING------------------------------------
# To add a new option to the dropdown menu in the graph widget, add the option to
# the SINGLE_WELL_COMBO_OPTIONS list in _util.py. Then, add the corresponding key-value
# pair to the SINGLE_WELL_GRAPHS_OPTIONS dictionary in this file. Finally, add the
# corresponding plotting logic to the `plot_single_well_traces` or
# `_plot_single_well_traces` functions below.
# --------------------------------------------------------------------------------------


def plot_single_well_traces(
    widget: _SingleWellGraphWidget,
    data: dict,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot traces based on the text."""
    if not text or text == "None":
        return

    if "stimulated" in text.lower():
        return visualize_stimulated_area(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    # get the options for the text using the SINGLE_WELL_GRAPHS_OPTIONS dictionary that
    # maps the text to the options
    return _plot_single_well_traces(
        widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
    )


def _plot_single_well_traces(
    widget: _SingleWellGraphWidget,
    data: dict,
    rois: list[int] | None = None,
    dff: bool = False,
    dec: bool = False,
    normalize: bool = False,
    with_peaks: bool = False,
    amp: bool = False,
    freq: bool = False,
    iei: bool = False,
) -> None:
    """Plot various types of traces."""
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Collect the title parts --------------------------
    title_parts = []
    if normalize:
        title_parts.append("Normalized Traces [0, 1]")
    if with_peaks:
        title_parts.append("Peaks")
    ax.set_title(" - ".join(title_parts))
    # --------------------------------------------------

    # loop over the ROIData and plot the traces per ROI
    count = 0
    rois_rec_time: list[float] = []
    for key in data:
        if rois is not None and int(key) not in rois:
            continue

        roi_data = cast("ROIData", data[key])

        # get the correct trace based on the flags
        trace = get_trace(roi_data, dff, dec)
        if trace is None:
            continue

        if (ttime := roi_data.total_recording_time_in_sec) is not None:
            rois_rec_time.append(ttime)

        if amp and freq:
            # plot amp vs freq
            if roi_data.peaks_amplitudes_dec_dff is None:
                continue
            amp_list = roi_data.peaks_amplitudes_dec_dff
            roi_freq_list = [roi_data.dec_dff_frequency] * len(amp_list)
            ax.plot(amp_list, roi_freq_list, "o", label=f"ROI {key}")

        elif amp:
            # plot amplitude
            if roi_data.peaks_amplitudes_dec_dff is None:
                continue
            ax.plot(
                [int(key)] * len(roi_data.peaks_amplitudes_dec_dff),
                roi_data.peaks_amplitudes_dec_dff,
                "o",
                label=f"ROI {key}",
            )

        elif freq:
            # plot frequency
            ax.plot(int(key), roi_data.dec_dff_frequency, "o", label=f"ROI {key}")

        elif iei:
            # plot inter-event intervals
            if roi_data.iei is None:
                continue
            ax.plot(
                [int(key)] * len(roi_data.iei), roi_data.iei, "o", label=f"ROI {key}"
            )

        else:
            # normalize if the flag is set
            if normalize:
                trace = normalize_trace(trace)
                ax.plot(np.array(trace) + count, label=f"ROI {key}")
                # hide the y-axis labels
                ax.set_yticklabels([])
                # hide the ticks
                ax.set_yticks([])
            else:
                ax.plot(trace, label=f"ROI {key}")

            # plot the peaks if the flag is set
            if with_peaks:
                if roi_data.peaks_dec_dff is None:
                    continue
                peaks_indices = np.array(roi_data.peaks_dec_dff)
                ax.plot(
                    peaks_indices,
                    np.array(trace)[peaks_indices.astype(int)]
                    + (count if normalize else 0),
                    "x",
                    label=f"Peaks ROI {key}",
                )

        count += COUNT_INCREMENT

    # set the axis labels
    if amp and freq:
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Frequency")
    elif amp:
        ax.set_xlabel("ROIs")
        ax.set_ylabel("Amplitude")
    elif freq:
        ax.set_xlabel("ROIs")
        ax.set_ylabel("Frequency")
    elif iei:
        ax.set_xlabel("ROIs")
        ax.set_ylabel("Inter-event intervals (sec)")
    else:
        if dff:
            ax.set_ylabel("dF/F")
        elif dec:
            ax.set_ylabel("Deconvolved dF/F")
        else:
            ax.set_ylabel("Fluorescence Intensity")

        if sum(rois_rec_time) > 0:
            # get the average total recording time in seconds
            avg_rec_time = int(np.mean(rois_rec_time))
            # get total number of frames from last trace (they should all be the same)
            total_frames = len(trace) if trace is not None else 1
            # compute tick positions
            tick_interval = avg_rec_time / total_frames
            x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
            x_labels = [str(int(t * tick_interval)) for t in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel("Frames")

    widget.figure.tight_layout()

    # Add hover functionality using mplcursors
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        # emit the graph widget roiSelected signal
        if sel.artist.get_label():
            roi = cast(str, sel.artist.get_label().split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)

    widget.canvas.draw()


def get_trace(roi_data: ROIData, dff: bool, dec: bool) -> list[float] | None:
    """Get the appropriate trace based on the flags."""
    if dff and dec:
        return None
    if dff:
        return roi_data.dff
    if dec:
        return roi_data.dec_dff
    return roi_data.raw_trace


def normalize_trace(trace: list[float]) -> list[float]:
    """Normalize the trace to the range [0, 1]."""
    tr = np.array(trace)
    normalized = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
    return cast(list[float], normalized.tolist())


def visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    data: dict,
    rois: list[int] | None = None,
    with_rois: bool = False,
) -> None:
    """Visualize Stimulated area."""
    fov = widget._plate_viewer._fov_table.value()
    if fov is None:
        return

    if (
        widget._plate_viewer._datastore is None
        or widget._plate_viewer._datastore.sequence is None
        or widget._plate_viewer._datastore.sequence.stage_positions is None
    ):
        return

    t = int(len(widget._plate_viewer._datastore.sequence.stage_positions) / 3 * 2)
    fov_snapshot = cast(
        np.ndarray, widget._plate_viewer._datastore.isel(p=fov.pos_idx, t=t, c=0)
    )
    st_area = widget._plate_viewer._analysis_wdg._stimulated_area

    if st_area is None:
        return

    if fov_snapshot is None:
        return

    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    ax.imshow(fov_snapshot, cmap="gray")
    ax.imshow(st_area, cmap="Blues", alpha=0.5)

    if with_rois:
        label = widget._plate_viewer._get_labels(fov)
        if not isinstance(label, np.ndarray):
            return

        if rois is not None:
            st_rois, ust_rois = _group_rois(data, rois)

        st_color = [1, 0, 0]
        ust_color = [0, 0, 1]

        mask_overlay = np.zeros((label.shape[0], label.shape[1], 3))

        for roi in st_rois:
            mask_overlay[label == roi] = st_color

        for roi in ust_rois:
            mask_overlay[label == roi] = ust_color

        ax.imshow(mask_overlay, interpolation="none", alpha=0.5)

        # Add hover functionality using mplcursors
        cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

        @cursor.connect("add")  # type: ignore [misc]
        def on_add(sel: mplcursors.Selection) -> None:
            if label is None:
                return
            x, y = int(sel.target[0]), int(sel.target[1])
            if 0 <= y < label.shape[0] and 0 <= x < label.shape[1]:
                roi = str(label[y, x]) if label[y, x] > 0 else None
            else:
                roi = None

            if not roi:
                sel.annotation.set_text("")
                return

            sel.annotation.set(text=roi, fontsize=8, color="black")
            # emit the graph widget roiSelected signal
            # if sel.artist.get_label():
            if roi.isdigit():
                widget.roiSelected.emit(roi)

    ax.axis("off")
    widget.canvas.draw()


def _group_rois(data: dict, rois: list[int]) -> tuple[list[int], list[int]]:
    """To group the ROIs based on stimulated state."""
    st_rois: list[int] = []
    ust_rois: list[int] = []

    for key in data:
        if rois is not None and int(key) not in rois:
            continue

        roi_data = cast("ROIData", data[key])

        if roi_data.stimulated is None:
            continue

        if roi_data.stimulated:
            st_rois.append(int(key))
        else:
            ust_rois.append(int(key))

    return st_rois, ust_rois


# ------------------------------MULTI-WELL PLOTTING-------------------------------------
# To add a new option to the dropdown menu in the graph widget, add the option to
# the MULTI_WELL_COMBO_OPTIONS list in _util.py. Then, add the corresponding key-value
# pair to the MULTI_WELL_GRAPHS_OPTIONS dictionary in this file. Finally, add the
# corresponding plotting logic to the `plot_multi_well_data` or
# `_plot_multi_well_data` functions below.
# --------------------------------------------------------------------------------------


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    data: dict[str, dict[str, ROIData]],
    positions: list[int] | None = None,
) -> None:
    """Plot the multi-well data."""
    if not text or text == "None" or not data:
        return

    # get the options for the text using the MULTI_WELL_GRAPHS_OPTIONS dictionary that
    # maps the text to the options
    return _plot_multi_well_data(
        widget, data, positions, **MULTI_WELL_GRAPHS_OPTIONS[text]
    )


def _plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    data: dict[str, dict[str, ROIData]],
    positions: list[int] | None = None,
    amp: bool = False,
    freq: bool = False,
    iei: bool = False,
) -> None:
    """Plot the multi-well data."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    well_count: int = 0
    wells_names: list[str] = []
    # loop over the data and plot the data
    for well_and_fov, roi_data in data.items():
        if positions is not None and int(well_and_fov) not in positions:
            continue

        # this is to store the well names to then use them as x axis labels
        if (well := well_and_fov.split("_")[0]) not in wells_names:
            wells_names.append(well)
            # we increment the well count only when we find a new well
            well_count += 1

        for roi in roi_data:
            r_data = roi_data[roi]

            if amp and freq:
                if r_data.peaks_amplitudes_dec_dff is None:
                    continue
                amp_list = r_data.peaks_amplitudes_dec_dff
                well_freq_list = [r_data.dec_dff_frequency] * len(amp_list)
                ax.plot(
                    amp_list, well_freq_list, "o", label=f"{well_and_fov} - roi{roi}"
                )
                ax.set_xlabel("Amplitude")
                ax.set_ylabel("Frequency")

            elif amp:
                if r_data.peaks_amplitudes_dec_dff is None:
                    continue
                ax.plot(
                    [well_count] * len(r_data.peaks_amplitudes_dec_dff),
                    r_data.peaks_amplitudes_dec_dff,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel("Amplitude")

            elif freq:
                ax.plot(
                    well_count,
                    r_data.dec_dff_frequency,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel("Frequency")

            elif iei:
                if r_data.iei is None:
                    continue
                ax.plot(
                    [well_count] * len(r_data.iei),
                    r_data.iei,
                    "o",
                    label=f"{well_and_fov} - roi{roi}",
                )
                ax.set_xlabel("Wells")
                ax.set_ylabel("Inter-event intervals (sec)")

    # set ticks labels as well names
    if (amp or freq or iei) and not (amp and freq):
        ax.set_xticks(range(1, len(wells_names) + 1))
        ax.set_xticklabels(wells_names)

    widget.figure.tight_layout()

    # add hover functionality to display the well position
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")

    widget.canvas.draw()
