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
    dec: bool,
) -> list[float] | None:
    """Get the appropriate trace based on the flags."""
    # NOTE: dff and dec can't be True at the same time
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


def plot_traces(
    widget: _GraphWidget,
    data: dict,
    rois: list[int] | None = None,
    dff: bool = False,
    dec: bool = False,
    normalize: bool = False,
    with_peaks: bool = False,
    amp: bool = False,
    freq: bool = False,
) -> None:
    """Plot various types of traces."""
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Set the title
    title_parts = []
    if normalize:
        title_parts.append("Normalized Traces [0, 1]")
    if with_peaks:
        title_parts.append("Peaks")
    ax.set_title(" - ".join(title_parts))

    count = 0
    for key in data:
        if rois is not None and int(key) not in rois:
            continue

        roi_data = cast("ROIData", data[key])
        trace = get_trace(roi_data, dff, dec)

        if trace is None:
            continue

        if amp:
            if roi_data.peaks_amplitudes_dec_dff is None:
                continue
            ax.plot(
                [int(key)] * len(roi_data.peaks_amplitudes_dec_dff),
                roi_data.peaks_amplitudes_dec_dff,
                "o",
                label=f"ROI {key}",
            )
            ax.set_xlabel("ROIs")
            ax.set_ylabel("Amplitude")
        elif freq:
            ax.plot(
                int(key),
                roi_data.dec_dff_frequency,
                "o",
                label=f"ROI {key}",
            )
            ax.set_xlabel("ROIs")
            ax.set_ylabel("Frequency")
        else:
            if normalize:
                trace = normalize_trace(trace)
                ax.plot(np.array(trace) + count, label=f"ROI {key}")
            else:
                ax.plot(trace, label=f"ROI {key}")

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
