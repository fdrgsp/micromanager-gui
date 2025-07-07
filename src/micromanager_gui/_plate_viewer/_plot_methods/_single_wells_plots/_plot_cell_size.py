from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from micromanager_gui._plate_viewer._graph_widgets import (
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


def _plot_cell_size_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot global synchrony."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    units = ""

    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue
        if not roi_data.cell_size:
            continue
        if not units and roi_data.cell_size_units:
            units = roi_data.cell_size_units
        ax.scatter(int(roi_key), roi_data.cell_size, label=f"ROI {roi_key}")

    ax.set_xlabel("ROI")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel(f"Cell Size ({units})")
    ax.set_title("Cell Size per ROI")

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


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

            # Get the units from the y-axis label
            y_label = ax.get_ylabel()
            # Extract units from the y-axis label (e.g., "Cell Size (μm²)" -> "μm²")
            if "(" in y_label and ")" in y_label:
                units = y_label.split("(")[1].split(")")[0]
                hover_text = f"{label}\nSize: {y:.3f} {units}"
            else:
                hover_text = f"{label}\nSize: {y:.3f}"

            sel.annotation.set(text=hover_text, fontsize=8, color="black")

            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
