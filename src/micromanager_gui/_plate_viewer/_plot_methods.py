from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np

from ._util import (
    DEC_DFF,
    DEC_DFF_AMPLITUDE,
    DEC_DFF_AMPLITUDE_VS_FREQUENCY,
    DEC_DFF_FREQUENCY,
    DEC_DFF_NORMALIZED,
    DEC_DFF_NORMALIZED_WITH_PEAKS,
    DEC_DFF_WITH_PEAKS,
    DFF,
    DFF_NORMALIZED,
    NORMALIZED_TRACES,
    RAW_TRACES,
)

if TYPE_CHECKING:
    from ._graph_widget import _GraphWidget
    from ._util import ROIData

COUNT_INCREMENT = 1

GRAPHS_OPTIONS: dict[str, dict[str, bool]] = {
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
}


# --------------------------------------NOTE--------------------------------------
# To add a new option to the dropdown menu in the graph widget, add the option to
# the COMBO_OPTIONS list in _util.py. Then, add the corresponding key-value pair to
# the GRAPHS_OPTIONS dictionary in this file. Finally, add the corresponding plotting
# logic to the `plot_traces` or `_plot_traces` functions below.
# --------------------------------------------------------------------------------


def plot_traces(
    widget: _GraphWidget,
    data: dict,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot traces based on the text."""
    # get the options for the text using the GRAPHS_OPTIONS dictionary that maps the
    # text to the options
    if not text or text == "None":
        return

    # TODO: add raster plot
    # if "raster" in text.lower():
    # return _plot_raster(...)

    return _plot_traces(widget, data, rois, **GRAPHS_OPTIONS[text])


def _plot_traces(
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
    for key in data:
        if rois is not None and int(key) not in rois:
            continue

        roi_data = cast("ROIData", data[key])

        # get the correct trace based on the flags
        trace = get_trace(roi_data, dff, dec)
        if trace is None:
            continue

        if amp and freq:
            # plot amp vs freq
            if roi_data.peaks_amplitudes_dec_dff is None:
                continue
            amp_list = roi_data.peaks_amplitudes_dec_dff
            roi_freq_list = [roi_data.dec_dff_frequency] * len(amp_list)
            ax.plot(roi_freq_list, amp_list, "o", label=f"ROI {key}")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Amplitude")

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
            ax.set_xlabel("ROIs")
            ax.set_ylabel("Amplitude")

        elif freq:
            # plot frequency
            ax.plot(
                int(key),
                roi_data.dec_dff_frequency,
                "o",
                label=f"ROI {key}",
            )
            ax.set_xlabel("ROIs")
            ax.set_ylabel("Frequency")

        else:
            # normalize if the flag is set
            if normalize:
                trace = normalize_trace(trace)
                ax.plot(np.array(trace) + count, label=f"ROI {key}")
                ax.set_ylabel("ROI")
            else:
                ax.plot(trace, label=f"ROI {key}")
                # set the y-axis label depending on the flags
                if dff:
                    ax.set_ylabel("dF/F")
                elif dec:
                    ax.set_ylabel("Deconvolved dF/F")
                else:
                    # this in case or raw traces
                    ax.set_ylabel("Fluorescence Intensity")

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
            ax.set_xlabel("Frames")

        count += COUNT_INCREMENT

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
