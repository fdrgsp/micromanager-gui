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
    data: dict[str, ROIData],
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
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    title = [
        "Normalized Traces [0, 1]" if normalize else "",
        "Peaks" if with_peaks else "",
    ]
    ax.set_title(" - ".join(filter(None, title)))

    rois_rec_time: list[float] = []
    count = 0

    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        trace: list[float] | None = _get_trace(roi_data, dff, dec)
        if trace is None:
            continue

        if (ttime := roi_data.total_recording_time_in_sec) is not None:
            rois_rec_time.append(ttime)

        if amp or freq or iei:
            _plot_metrics(ax, roi_key, roi_data, amp, freq, iei)
        else:
            _plot_trace(ax, roi_key, trace, normalize, with_peaks, roi_data, count)
            count += COUNT_INCREMENT

    _set_axis_labels(ax, amp, freq, iei, dff, dec)
    if not (amp or freq or iei):
        _update_time_axis(ax, rois_rec_time, trace)
    widget.figure.tight_layout()

    _add_hover_functionality(ax, widget)
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes, roi_key: str, roi_data: ROIData, amp: bool, freq: bool, iei: bool
) -> None:
    """Plot amplitude, frequency, or inter-event intervals."""
    if amp and freq:
        if roi_data.peaks_amplitudes_dec_dff:
            ax.plot(
                roi_data.peaks_amplitudes_dec_dff,
                [roi_data.dec_dff_frequency] * len(roi_data.peaks_amplitudes_dec_dff),
                "o",
                label=f"ROI {roi_key}",
            )
    elif amp:
        if roi_data.peaks_amplitudes_dec_dff:
            ax.plot(
                [int(roi_key)] * len(roi_data.peaks_amplitudes_dec_dff),
                roi_data.peaks_amplitudes_dec_dff,
                "o",
                label=f"ROI {roi_key}",
            )
    elif freq:
        ax.plot(int(roi_key), roi_data.dec_dff_frequency, "o", label=f"ROI {roi_key}")
    elif iei and roi_data.iei:
        ax.plot(
            [int(roi_key)] * len(roi_data.iei),
            roi_data.iei,
            "o",
            label=f"ROI {roi_key}",
        )


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float],
    normalize: bool,
    with_peaks: bool,
    roi_data: ROIData,
    count: int,
) -> None:
    """Plot trace data with optional normalization and peaks."""
    if normalize:
        trace = _normalize_trace(trace)
        ax.plot(np.array(trace) + count, label=f"ROI {roi_key}")
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    if with_peaks and roi_data.peaks_dec_dff:
        peaks_indices = [int(peak) for peak in roi_data.peaks_dec_dff]
        ax.plot(
            peaks_indices,
            np.array(trace)[peaks_indices] + (count if normalize else 0),
            "x",
            label=f"Peaks ROI {roi_key}",
        )


def _set_axis_labels(
    ax: Axes, amp: bool, freq: bool, iei: bool, dff: bool, dec: bool
) -> None:
    """Set axis labels based on the plotted data."""
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
        ax.set_ylabel(
            "dF/F" if dff else "Deconvolved dF/F" if dec else "Fluorescence Intensity"
        )


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        if sel.artist.get_label():
            roi = cast(str, sel.artist.get_label().split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)


def _get_trace(roi_data: ROIData, dff: bool, dec: bool) -> list[float] | None:
    """Get the appropriate trace based on the flags."""
    if dff and dec:
        return None
    if dff:
        return roi_data.dff
    if dec:
        return roi_data.dec_dff
    return roi_data.raw_trace


def _normalize_trace(trace: list[float]) -> list[float]:
    """Normalize the trace to the range [0, 1]."""
    tr = np.array(trace)
    normalized = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
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
