from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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


def _plot_amplitude_and_frequency_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    amp: bool = False,
    freq: bool = False,
    std: bool = False,
    sem: bool = False,
) -> None:
    """Plot traces data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # compute nth and nth percentiles globally
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        _plot_metrics(ax, roi_key, roi_data, amp, freq, std, sem)

    _set_graph_title_and_labels(ax, amp, freq, std, sem)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi_key: str,
    roi_data: ROIData,
    amp: bool,
    freq: bool,
    std: bool,
    sem: bool,
) -> None:
    """Plot amplitude, frequency, or inter-event intervals."""
    if amp and freq:
        if not roi_data.peaks_amplitudes_dec_dff or roi_data.dec_dff_frequency is None:
            return
        # plot mean amplitude +- std of each ROI vs frequency
        if std:
            mean_amp = cast(list[float], np.mean(roi_data.peaks_amplitudes_dec_dff))
            std_amp = np.std(roi_data.peaks_amplitudes_dec_dff)
            _plot_errorbars(
                ax, mean_amp, roi_data.dec_dff_frequency, std_amp, f"ROI {roi_key}"
            )
        # plot mean amplitude +- sem of each ROI vs frequency
        elif sem:
            mean_amp = cast(list[float], np.mean(roi_data.peaks_amplitudes_dec_dff))
            sem_amp = mean_amp / np.sqrt(len(roi_data.peaks_amplitudes_dec_dff))
            _plot_errorbars(
                ax, mean_amp, roi_data.dec_dff_frequency, sem_amp, f"ROI {roi_key}"
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
            mean_amp = cast(list[float], np.mean(roi_data.peaks_amplitudes_dec_dff))
            std_amp = np.std(roi_data.peaks_amplitudes_dec_dff)
            _plot_errorbars(ax, [int(roi_key)], mean_amp, std_amp, f"ROI {roi_key}")
        # plot mean amplitude +- sem of each ROI
        elif sem:
            mean_amp = cast(list[float], np.mean(roi_data.peaks_amplitudes_dec_dff))
            sem_amp = mean_amp / np.sqrt(len(roi_data.peaks_amplitudes_dec_dff))
            _plot_errorbars(ax, [int(roi_key)], mean_amp, sem_amp, f"ROI {roi_key}")
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


def _plot_errorbars(
    ax: Axes, x: list[float], y: float | list[float], yerr: Any, label: str
) -> None:
    """Plot error bars graph."""
    ax.errorbar(x, y, yerr=yerr, label=label, fmt="o", capsize=5)


def _set_graph_title_and_labels(
    ax: Axes,
    amp: bool,
    freq: bool,
    std: bool,
    sem: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    if amp and freq:
        if std:
            title = "ROIs Mean Amplitude ± StD vs Frequency"
        elif sem:
            title = "ROIs Mean Amplitude ± SEM vs Frequency"
        else:
            title = "ROIs Amplitude vs Frequency"
        title += " (Deconvolved dF/F)"
        x_lbl = "Amplitude"
        y_lbl = "Frequency (Hz)"
    elif amp:
        if std:
            title = "Mean Amplitude ± StD"
        elif sem:
            title = "Mean Amplitude ± SEM"
        else:
            title = "Amplitude"
        title += " (Deconvolved dF/F)"
        x_lbl = "ROIs"
        y_lbl = "Amplitude"
    elif freq:
        title = "Frequency (Deconvolved dF/F)"
        x_lbl = "ROIs"
        y_lbl = "Frequency (Hz)"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)
    ax.set_xlabel(x_lbl)
    if x_lbl == "ROIs":
        ax.set_xticks([])
        ax.set_xticklabels([])


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
