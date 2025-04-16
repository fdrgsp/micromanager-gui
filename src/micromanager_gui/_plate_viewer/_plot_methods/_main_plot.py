from __future__ import annotations

from typing import TYPE_CHECKING

from micromanager_gui._plate_viewer._util import (
    CELL_SIZE_ALL,
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
    RASTER_PLOT_AMP_WITH_COLORBAR,
    RAW_TRACES,
    STIMULATED_AREA,
    STIMULATED_ROIS,
    STIMULATED_ROIS_WITH_STIMULATED_AREA,
    SYNCHRONY_ALL,
)

from ._multi_wells_plots._multi_well_data_plot import (
    _plot_multi_cond_data,
    _plot_multi_well_data,
)
from ._single_wells_plots._raster_plots import _generate_raster_plot
from ._single_wells_plots._single_well_data import _plot_single_well_data
from ._single_wells_plots._stimulation_plots import _visualize_stimulated_area

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData

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
    RASTER_PLOT_AMP: {"amplitude_colors": True, "colorbar": False},
    RASTER_PLOT_AMP_WITH_COLORBAR: {"amplitude_colors": True, "colorbar": True},
    DEC_DFF_IEI: {"dec": True, "iei": True},
    STIMULATED_AREA: {"with_rois": False, "stimulated_area": False},
    STIMULATED_ROIS: {"with_rois": True, "stimulated_area": False},
    STIMULATED_ROIS_WITH_STIMULATED_AREA: {"with_rois": True, "stimulated_area": True},
}

MULTI_WELL_GRAPHS_OPTIONS: dict[str, dict[str, bool]] = {
    DEC_DFF_AMPLITUDE_VS_FREQUENCY: {"amp": True, "freq": True},
    DEC_DFF_AMPLITUDE: {"amp": True},
    DEC_DFF_FREQUENCY: {"freq": True},
    DEC_DFF_IEI: {"iei": True},
    CELL_SIZE_ALL: {"cell_size": True},
    SYNCHRONY_ALL: {"sync": True},
}


# ------------------------------SINGLE-WELL PLOTTING------------------------------------
# To add a new option to the dropdown menu in the graph widget, add the option to
# the SINGLE_WELL_COMBO_OPTIONS list in _util.py. Then, add the corresponding key-value
# pair to the SINGLE_WELL_GRAPHS_OPTIONS dictionary in this file. Finally, add the
# corresponding plotting logic to the `plot_single_well_data` or
# `plot_multi_well_data` functions below.
# --------------------------------------------------------------------------------------


def plot_single_well_data(
    widget: _SingleWellGraphWidget,
    data: dict,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot traces based on the text."""
    if not text or text == "None":
        return

    # get the options for the text using the SINGLE_WELL_GRAPHS_OPTIONS dictionary
    # that maps the text to the options

    # plot stimulated area/ROIs
    if text in {STIMULATED_AREA, STIMULATED_ROIS, STIMULATED_ROIS_WITH_STIMULATED_AREA}:
        return _visualize_stimulated_area(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    # plot raster plot
    if text in {RASTER_PLOT, RASTER_PLOT_AMP, RASTER_PLOT_AMP_WITH_COLORBAR}:
        return _generate_raster_plot(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    # plot other types of graphs
    else:
        return _plot_single_well_data(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )


# ------------------------------MULTI-WELL PLOTTING-------------------------------------
# To add a new option to the dropdown menu in the graph widget, add the option to
# the MULTI_WELL_COMBO_OPTIONS list in _util.py. Then, add the corresponding key-value
# pair to the MULTI_WELL_GRAPHS_OPTIONS dictionary in this file. Finally, add the
# corresponding plotting logic to the `plot_multi_well_data` or
# `_plot_multi_well_data` functions below.
# --------------------------------------------------------------------------------------


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    data: dict[str, dict[str, ROIData]],
    # positions: list[int] | None = None,
) -> None:
    """Plot the multi-well data."""
    if not text or text == "None" or not data:
        return

    # get the options for the text using the MULTI_WELL_GRAPHS_OPTIONS dictionary that
    # maps the text to the options
    return _plot_multi_well_data(widget, data, **MULTI_WELL_GRAPHS_OPTIONS[text])


def plot_multi_cond_data(
    widget: _MultilWellGraphWidget,
    text: str,
    data: dict[str, list[ROIData]],
    cond_order: list[str],
    plate_map_color: dict[str, str],
) -> None:
    """Plot the multi-well data."""
    if not text or text == "None" or not data:
        return

    # get the options for the text using the MULTI_WELL_GRAPHS_OPTIONS dictionary that
    # maps the text to the options
    return _plot_multi_cond_data(
        widget, data, cond_order, plate_map_color, **MULTI_WELL_GRAPHS_OPTIONS[text]
    )
