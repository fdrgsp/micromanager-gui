from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

from micromanager_gui._plate_viewer._util import (
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
    RASTER_PLOT,
    RASTER_PLOT_AMP,
    RAW_TRACES,
    STIMULATED_AREA,
    STIMULATED_ROIS,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData

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
    RASTER_PLOT: {"amplitude_colors": False},
    RASTER_PLOT_AMP: {"amplitude_colors": True},
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


def _plot_single_well_data(
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

    for roi_key in data:
        if rois is not None and int(roi_key) not in rois:
            continue

        roi_data = cast("ROIData", data[roi_key])

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
            ax.plot(amp_list, roi_freq_list, "o", label=f"ROI {roi_key}")

        elif amp:
            # plot amplitude
            if roi_data.peaks_amplitudes_dec_dff is None:
                continue
            ax.plot(
                [int(roi_key)] * len(roi_data.peaks_amplitudes_dec_dff),
                roi_data.peaks_amplitudes_dec_dff,
                "o",
                label=f"ROI {roi_key}",
            )

        elif freq:
            # plot frequency
            ax.plot(
                int(roi_key), roi_data.dec_dff_frequency, "o", label=f"ROI {roi_key}"
            )

        elif iei:
            # plot inter-event intervals
            if roi_data.iei is None:
                continue
            ax.plot(
                [int(roi_key)] * len(roi_data.iei),
                roi_data.iei,
                "o",
                label=f"ROI {roi_key}",
            )

        else:
            # normalize if the flag is set
            if normalize:
                trace = normalize_trace(trace)
                ax.plot(np.array(trace) + count, label=f"ROI {roi_key}")
                # hide the y-axis labels
                ax.set_yticklabels([])
                # hide the ticks
                ax.set_yticks([])
            else:
                ax.plot(trace, label=f"ROI {roi_key}")

            # plot the peaks if the flag is set
            if with_peaks:
                if roi_data.peaks_dec_dff is None:
                    continue
                peaks_indices = [int(peak_ind) for peak_ind in roi_data.peaks_dec_dff]
                ax.plot(
                    peaks_indices,
                    np.array(trace)[peaks_indices] + (count if normalize else 0),
                    "x",
                    label=f"Peaks ROI {roi_key}",
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
        # update the x-axis to show time in seconds or frames
        update_time_axis(ax, rois_rec_time, trace)

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


def update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    print("updating time axis", trace is not None, sum(rois_rec_time) > 0)
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
