from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

if TYPE_CHECKING:
    from ._graph_widget import _GraphWidget
    from ._util import ROIData

COUNT_INCREMENT = 1


def get_trace(
    roi_data: ROIData,
    dff: bool,
    photobleach_corrected: bool,
    used_for_bleach_correction: bool,
) -> list[float] | None:
    """Get the appropriate trace based on the flags."""
    if used_for_bleach_correction:
        trace = roi_data.use_for_bleach_correction
        return trace[0] if trace is not None else None
    elif dff and not photobleach_corrected:
        return roi_data.dff
    elif photobleach_corrected and not dff:
        return roi_data.bleach_corrected_trace
    else:
        return roi_data.raw_trace


def normalize_trace(trace: list[float]) -> list[float]:
    """Normalize the trace to the range [0, 1]."""
    tr = np.array(trace)
    normalized = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))
    return cast(list[float], normalized.tolist())


def plot_traces(
    widget: _GraphWidget,
    data: dict,
    rois: list[int] | None = None,
    dff: bool = False,
    normalize: bool = False,
    photobleach_corrected: bool = False,
    with_peaks: bool = False,
    used_for_bleach_correction: bool = False,
) -> None:
    """Plot various types of traces."""
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Set the title
    title_parts = []
    if used_for_bleach_correction:
        title_parts.append("Traces Used for Bleach Correction")
    if normalize:
        title_parts.append("Normalized Traces [0, 1]")
    if dff and not used_for_bleach_correction:
        title_parts.append("ΔF/F - Photobleach Correction")
    if photobleach_corrected and not dff and not used_for_bleach_correction:
        title_parts.append("Photobleach Correction")
    if with_peaks:
        title_parts.append("Peaks")
    ax.set_title(" - ".join(title_parts))

    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue

        roi_data = cast("ROIData", data[key])
        trace = get_trace(
            roi_data, dff, photobleach_corrected, used_for_bleach_correction
        )

        if trace is None:
            continue

        if normalize:
            trace = normalize_trace(trace)
            ax.plot(
                np.array(trace) + (0 if used_for_bleach_correction else count),
                label=f"ROI {key}",
            )
        else:
            ax.plot(trace, label=f"ROI {key}")

        if with_peaks and roi_data.peaks is not None:
            peaks = [pk.peak for pk in roi_data.peaks if pk.peak is not None]
            ax.plot(
                peaks,
                np.array(trace)[peaks]
                + (count if normalize and not used_for_bleach_correction else 0),
                "x",
                label=f"Peaks ROI {key}",
            )

        if used_for_bleach_correction:
            curve = data[next(iter(data.keys()))].average_photobleaching_fitted_curve
            if curve is not None:
                if normalize:
                    curve = normalize_trace(curve)
                ax.plot(
                    curve,
                    label="Fitted Curve",
                    linestyle="--",
                    color="black",
                    linewidth=2,
                )

        count += COUNT_INCREMENT

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
