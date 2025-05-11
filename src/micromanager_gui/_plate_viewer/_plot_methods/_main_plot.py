from __future__ import annotations

from typing import TYPE_CHECKING

from micromanager_gui._plate_viewer._util import (
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
    GLOBAL_SYNCHRONY,
    NORMALIZED_TRACES,
    RASTER_PLOT,
    RASTER_PLOT_AMP,
    RASTER_PLOT_AMP_WITH_COLORBAR,
    RAW_TRACES,
    SPONTANEOUS_PEAKS_AMP,
    STIMULATED_AREA,
    STIMULATED_PEAKS_AMP,
    STIMULATED_ROIS,
    STIMULATED_ROIS_WITH_STIMULATED_AREA,
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED,
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS,
)

from ._multi_wells_plots._multi_well_data_plot import _plot_multi_well_data
from ._single_wells_plots._raster_plots import _generate_raster_plot
from ._single_wells_plots._single_well_data import _plot_single_well_data
from ._single_wells_plots._stimulation_plots import (
    _plot_spontaneous_peaks_amplitude,
    _plot_stimulated_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_roi_amp,
    _visualize_stimulated_area,
)
from ._single_wells_plots._synchrony_plots import _plot_synchrony

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
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED: {"with_peaks": False},
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS: {"with_peaks": True},
    STIMULATED_PEAKS_AMP: {},
    SPONTANEOUS_PEAKS_AMP: {},
    GLOBAL_SYNCHRONY: {},
}

MULTI_WELL_GRAPHS_OPTIONS: dict[str, dict[str, bool]] = {
    DEC_DFF_AMPLITUDE_VS_FREQUENCY: {"amp": True, "freq": True},
    DEC_DFF_AMPLITUDE: {"amp": True},
    DEC_DFF_FREQUENCY: {"freq": True},
    DEC_DFF_IEI: {"iei": True},
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

    # plot stimulated peaks amplitude
    if text == STIMULATED_PEAKS_AMP:
        return _plot_stimulated_peaks_amplitude(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    # plot spontaneous peaks amplitude (non due to stimulation)
    if text == SPONTANEOUS_PEAKS_AMP:
        return _plot_spontaneous_peaks_amplitude(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    if text in {
        STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS,
        STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED,
    }:
        return _plot_stimulated_vs_non_stimulated_roi_amp(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    # plot raster plot
    if text in {RASTER_PLOT, RASTER_PLOT_AMP, RASTER_PLOT_AMP_WITH_COLORBAR}:
        return _generate_raster_plot(
            widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text]
        )

    if text in {GLOBAL_SYNCHRONY}:
        return _plot_synchrony(widget, data, rois, **SINGLE_WELL_GRAPHS_OPTIONS[text])

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
    positions: list[int] | None = None,
) -> None:
    """Plot the multi-well data."""
    if not text or text == "None" or not data:
        return

    # get the options for the text using the MULTI_WELL_GRAPHS_OPTIONS dictionary that
    # maps the text to the options
    return _plot_multi_well_data(
        widget, data, positions, **MULTI_WELL_GRAPHS_OPTIONS[text]
    )
