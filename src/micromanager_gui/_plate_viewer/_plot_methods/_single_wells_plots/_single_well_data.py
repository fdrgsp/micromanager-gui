from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


COUNT_INCREMENT = 1
P1 = 5
P2 = 99


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
    sem: bool = False,
    std: bool = False,
) -> None:
    """Plot various types of traces."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # compute nth and nth percentiles globally
    p1 = p2 = 0.0
    if normalize:
        all_values = []
        for roi_key, roi_data in data.items():
            if rois is not None and int(roi_key) not in rois:
                continue
            trace = _get_trace(roi_data, dff, dec)
            if trace is not None:
                all_values.extend(trace)
        if all_values:
            percentiles = np.percentile(all_values, [P1, P2])
            p1, p2 = float(percentiles[0]), float(percentiles[1])
        else:
            p1, p2 = 0.0, 1.0

    rois_rec_time: list[float] = []
    count = 0

    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        trace = _get_trace(roi_data, dff, dec)
        if trace is None:
            continue

        if (ttime := roi_data.total_recording_time_in_sec) is not None:
            rois_rec_time.append(ttime)

        if amp or freq or iei:
            _plot_metrics(ax, roi_key, roi_data, amp, freq, iei, std, sem)
        else:
            # plot only active neurons if asked to plot peaks
            if with_peaks and not roi_data.active:
                continue
            _plot_trace(
                ax,
                roi_key,
                trace,
                normalize,
                with_peaks,
                roi_data,
                count,
                p1,
                p2,
            )
            count += COUNT_INCREMENT

    _set_graph_title_and_labels(
        ax, amp, freq, iei, dff, dec, normalize, with_peaks, std, sem
    )
    if not (amp or freq or iei):
        _update_time_axis(ax, rois_rec_time, trace)
    widget.figure.tight_layout()

    _add_hover_functionality(ax, widget)
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi_key: str,
    roi_data: ROIData,
    amp: bool,
    freq: bool,
    iei: bool,
    std: bool,
    sem: bool,
) -> None:
    """Plot amplitude, frequency, or inter-event intervals."""
    if amp and freq:
        if not roi_data.peaks_amplitudes_dec_dff or roi_data.dec_dff_frequency is None:
            return
        # plot mean amplitude +- std of each ROI vs frequency
        if std:
            mean_amp = np.mean(roi_data.peaks_amplitudes_dec_dff)
            std_amp = np.std(roi_data.peaks_amplitudes_dec_dff)
            ax.errorbar(
                mean_amp,
                roi_data.dec_dff_frequency,
                xerr=std_amp,
                fmt="o",
                label=f"ROI {roi_key}",
                capsize=5,
            )
        # plot mean amplitude +- sem of each ROI vs frequency
        elif sem:
            mean_amp = np.mean(roi_data.peaks_amplitudes_dec_dff)
            sem_amp = mean_amp / np.sqrt(len(roi_data.peaks_amplitudes_dec_dff))
            ax.errorbar(
                mean_amp,
                roi_data.dec_dff_frequency,
                xerr=sem_amp,
                fmt="o",
                label=f"ROI {roi_key}",
                capsize=5,
            )
        else:
            ax.plot(
                roi_data.peaks_amplitudes_dec_dff,
                [roi_data.dec_dff_frequency] * len(roi_data.peaks_amplitudes_dec_dff),
                "o",
                label=f"ROI {roi_key}",
            )
    elif amp:
        if not roi_data.peaks_amplitudes_dec_dff:
            return
        # plot mean amplitude +- std of each ROI
        if std:
            mean_amp = np.mean(roi_data.peaks_amplitudes_dec_dff)
            std_amp = np.std(roi_data.peaks_amplitudes_dec_dff)
            ax.errorbar(
                [int(roi_key)],
                mean_amp,
                yerr=std_amp,
                fmt="o",
                label=f"ROI {roi_key}",
                capsize=5,
            )
        # plot mean amplitude +- sem of each ROI
        elif sem:
            mean_amp = np.mean(roi_data.peaks_amplitudes_dec_dff)
            sem_amp = mean_amp / np.sqrt(len(roi_data.peaks_amplitudes_dec_dff))
            ax.errorbar(
                [int(roi_key)],
                mean_amp,
                yerr=sem_amp,
                fmt="o",
                label=f"ROI {roi_key}",
                capsize=5,
            )
        else:
            ax.plot(
                [int(roi_key)] * len(roi_data.peaks_amplitudes_dec_dff),
                roi_data.peaks_amplitudes_dec_dff,
                "o",
                label=f"ROI {roi_key}",
            )
    elif freq:
        if roi_data.dec_dff_frequency is None:
            return
        ax.plot(int(roi_key), roi_data.dec_dff_frequency, "o", label=f"ROI {roi_key}")
    elif iei and roi_data.iei:
        # # plot mean inter-event intervals +- std of each ROI
        # if std:
        #     mean_iei = np.mean(roi_data.iei)
        #     std_iei = np.std(roi_data.iei)
        #     ax.errorbar(
        #         [int(roi_key)],
        #         mean_iei,
        #         yerr=std_iei,
        #         fmt="o",
        #         label=f"ROI {roi_key}",
        #         capsize=5,
        #     )
        # # plot mean inter-event intervals +- sem of each ROI
        # elif sem:
        #     mean_iei = np.mean(roi_data.iei)
        #     sem_iei = mean_iei / np.sqrt(len(roi_data.iei))
        #     ax.errorbar(
        #         [int(roi_key)],
        #         mean_iei,
        #         yerr=sem_iei,
        #         fmt="o",
        #         label=f"ROI {roi_key}",
        #         capsize=5,
        #     )
        # else:
        ax.plot(
            [int(roi_key)] * len(roi_data.iei),
            roi_data.iei,
            "o",
            label=f"ROI {roi_key}",
        )


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float] | np.ndarray,
    normalize: bool,
    with_peaks: bool,
    roi_data: ROIData,
    count: int,
    p1: float,
    p2: float,
) -> None:
    """Plot trace data with optional percentile-based normalization and peaks."""
    offset = count * 1.1  # vertical offset

    if normalize:
        trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    if with_peaks and roi_data.peaks_dec_dff:
        peaks_indices = [int(p) for p in roi_data.peaks_dec_dff]
        ax.plot(peaks_indices, np.array(trace)[peaks_indices], "x")


def _set_graph_title_and_labels(
    ax: Axes,
    amp: bool,
    freq: bool,
    iei: bool,
    dff: bool,
    dec: bool,
    normalize: bool,
    with_peaks: bool,
    std: bool,
    sem: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    x_lbl: str | None = None
    if amp and freq:
        if std:
            title = "ROIs Mean Amplitude ± StD vs Frequency"
            x_lbl = "Mean Amplitude ± StD"
        elif sem:
            title = "ROIs Mean Amplitude ± SEM vs Frequency"
            x_lbl = "Mean Amplitude ± SEM"
        else:
            title = "ROIs Amplitude vs Frequency"
            x_lbl = "Amplitude"
        title += " (Deconvolved dF/F)" if dec else ""
        y_lbl = "Frequency (Hz)"
    elif amp:
        if std:
            title = "Mean Amplitude ± StD"
            y_lbl = "Mean Amplitude ± StD"
        elif sem:
            title = "Mean Amplitude ± SEM"
            y_lbl = "Mean Amplitude ± SEM"
        else:
            title = "Amplitude"
            y_lbl = "Amplitude"
        title += " (Deconvolved dF/F)" if dec else ""
        x_lbl = "ROIs"
    elif freq:
        title = "Frequency (Deconvolved dF/F)" if dec else "Frequency"
        x_lbl = "ROIs"
        y_lbl = "Frequency (Hz)"
    elif iei:
        if std:
            title = "Inter-event intervals (Sec - Mean ± StD"
            y_lbl = "Inter-event intervals ± StD (Sec)"
        elif sem:
            title = "Inter-event intervals (Sec - Mean ± SEM"
            y_lbl = "Inter-event intervals ± SEM (Sec)"
        else:
            title = "Inter-event intervals (Sec"
            y_lbl = "Inter-event intervals (Sec)"
        title += " - (Deconvolved dF/F)" if dec else ")"
        x_lbl = "ROIs"
    else:
        if dff:
            title = "Normalized Traces (dF/F)" if normalize else "Traces (dF/F)"
            y_lbl = "ROIs" if normalize else "dF/F"
        elif dec:
            title = (
                "Normalized Traces (Deconvolved dF/F)"
                if normalize
                else "Traces (Deconvolved dF/F)"
            )
            y_lbl = "ROIs" if normalize else "Deconvolved dF/F"
        else:
            title = "Normalized Traces" if normalize else "Raw Traces"
            y_lbl = "ROIs" if normalize else "Fluorescence Intensity"
        if with_peaks:
            title += " with Peaks"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)
    if x_lbl is not None:
        ax.set_xlabel(x_lbl)


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        if (lbl := sel.artist.get_label()) and "ROI" in lbl:
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


def _normalize_trace_percentile(
    trace: list[float] | np.ndarray, p1: float, p2: float
) -> np.ndarray:
    """Normalize a trace using 5th-100th percentile, clipped to [0, 1]."""
    tr = np.array(trace) if isinstance(trace, list) else trace
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


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
