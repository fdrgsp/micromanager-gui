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


def _plot_amplitude_and_frequency_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    amp: bool = False,
    freq: bool = False,
) -> None:
    """Plot traces data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # compute nth and nth percentiles globally
    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        _plot_metrics(ax, roi_key, roi_data, amp, freq)

    _set_graph_title_and_labels(ax, amp, freq)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi_key: str,
    roi_data: ROIData,
    amp: bool,
    freq: bool,
) -> None:
    """Plot amplitude or frequency."""
    if amp and freq:
        if not roi_data.peaks_amplitudes_dec_dff or roi_data.dec_dff_frequency is None:
            return
        mean_amp = cast("list[float]", np.mean(roi_data.peaks_amplitudes_dec_dff))
        sem_amp = mean_amp / np.sqrt(len(roi_data.peaks_amplitudes_dec_dff))
        _plot_errorbars(
            ax, mean_amp, roi_data.dec_dff_frequency, sem_amp, f"ROI {roi_key}"
        )
    elif amp:
        if not roi_data.peaks_amplitudes_dec_dff:
            return

        # plot mean amplitude +- sem of each ROI
        mean_amp = cast("list[float]", np.mean(roi_data.peaks_amplitudes_dec_dff))
        sem_amp = mean_amp / np.sqrt(len(roi_data.peaks_amplitudes_dec_dff))
        _plot_errorbars(ax, [int(roi_key)], mean_amp, sem_amp, f"ROI {roi_key}")
        ax.scatter(
            [int(roi_key)] * len(roi_data.peaks_amplitudes_dec_dff),
            roi_data.peaks_amplitudes_dec_dff,
            alpha=0.5,
            s=30,
            color="lightgray",
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
) -> None:
    """Set axis labels based on the plotted data."""
    title = x_lbl = y_lbl = ""
    if amp and freq:
        title = (
            "ROIs Mean Calcium Peaks Amplitude ± SEM vs Frequency (Deconvolved ΔF/F)"
        )
        x_lbl = "Amplitude"
        y_lbl = "Frequency (Hz)"
    elif amp:
        title = "Calcium Peaks Mean Amplitude ± SEM (Deconvolved ΔF/F)"
        x_lbl = "ROIs"
        y_lbl = "Amplitude"
    elif freq:
        title = "Calcium Peaks Frequency (Deconvolved ΔF/F)"
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
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for peaks or other elements
        if label and "ROI" in label and not label.startswith("_"):
            # Get the data point coordinates
            x, y = sel.target

            # Create hover text with ROI and value information
            roi = cast("str", label.split(" ")[1])

            # Determine what type of plot this is based on axis labels
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()

            if "Amplitude" in x_label and "Frequency" in y_label:
                # Amplitude vs Frequency plot
                hover_text = f"{label}\nAmp: {x:.3f}\nFreq: {y:.3f} Hz"
            elif "Amplitude" in y_label:
                # Amplitude plot
                hover_text = f"{label}\nAmp: {y:.3f}"
            elif "Frequency" in y_label:
                # Frequency plot
                hover_text = f"{label}\nFreq: {y:.3f} Hz"
            else:
                # Fallback to just ROI label
                hover_text = label

            sel.annotation.set(text=hover_text, fontsize=8, color="black")

            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
