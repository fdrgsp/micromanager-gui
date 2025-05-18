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


def _plot_iei_data(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
    std: bool = False,
    sem: bool = False,
) -> None:
    """Plot traces data."""
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    for roi_key, roi_data in data.items():
        if rois is not None and int(roi_key) not in rois:
            continue

        _plot_metrics(ax, roi_key, roi_data, std, sem)

    _set_graph_title_and_labels(ax, std, sem)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_metrics(
    ax: Axes,
    roi_key: str,
    roi_data: ROIData,
    std: bool,
    sem: bool,
) -> None:
    """Plot amplitude, frequency, or inter-event intervals."""
    if not roi_data.iei:
        return
    # plot mean inter-event intervals +- std of each ROI
    if std:
        mean_iei = np.mean(roi_data.iei)
        std_iei = np.std(roi_data.iei)
        ax.errorbar(
            [int(roi_key)],
            mean_iei,
            yerr=std_iei,
            fmt="o",
            label=f"ROI {roi_key}",
            capsize=5,
        )
    # plot mean inter-event intervals +- sem of each ROI
    elif sem:
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
    else:
        ax.plot(
            [int(roi_key)] * len(roi_data.iei),
            roi_data.iei,
            "o",
            label=f"ROI {roi_key}",
        )


def _set_graph_title_and_labels(
    ax: Axes,
    std: bool,
    sem: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    if std:
        title = "Inter-event intervals (Sec - Mean ± StD - Deconvolved dF/F)"
    elif sem:
        title = "Inter-event intervals (Sec - Mean ± SEM - Deconvolved dF/F)"
    else:
        title = "Inter-event intervals (Sec - Deconvolved dF/F)"
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
        sel.annotation.set(text=sel.artist.get_label(), fontsize=8, color="black")
        if (lbl := sel.artist.get_label()) and "ROI" in lbl:
            roi = cast(str, sel.artist.get_label().split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)
