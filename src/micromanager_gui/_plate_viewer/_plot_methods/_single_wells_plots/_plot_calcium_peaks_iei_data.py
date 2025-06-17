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


def _plot_iei_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot traces data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        _plot_metrics(ax, roi_key, roi_data)

    _set_graph_title_and_labels(ax)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi_key: str,
    roi_data: ROIData,
) -> None:
    """Plot amplitude, frequency, or inter-event intervals."""
    if not roi_data.iei:
        return
    # plot mean inter-event intervals +- sem of each ROI
    mean_iei = np.mean(roi_data.iei)
    sem_iei = mean_iei / np.sqrt(len(roi_data.iei))
    ax.errorbar(
        [int(roi_key)],
        mean_iei,
        yerr=sem_iei,
        fmt="o",
        label=f"ROI {roi_key}",
        capsize=5,
    )
    ax.scatter(
        [int(roi_key)] * len(roi_data.iei),
        roi_data.iei,
        alpha=0.5,
        color="lightgray",
        s=30,
        label=f"ROI {roi_key}",
    )


def _set_graph_title_and_labels(
    ax: Axes,
) -> None:
    """Set axis labels based on the plotted data."""
    title = "Calcium Peaks Inter-event intervals (Sec - Mean ± SEM - Deconvolved ΔF/F)"
    x_lbl = "ROIs"
    ax.set_title(title)
    ax.set_ylabel("Inter-event intervals (Sec)")
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
            roi = cast(str, label.split(" ")[1])

            # Show IEI value in seconds
            hover_text = f"{label}\nIEI: {y:.3f} sec"

            sel.annotation.set(text=hover_text, fontsize=8, color="black")

            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
