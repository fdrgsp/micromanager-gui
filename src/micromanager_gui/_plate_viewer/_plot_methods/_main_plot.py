from __future__ import annotations

from typing import TYPE_CHECKING

from ._single_wells_plots._plolt_evoked_evperiment_data_plots import (
    _plot_stim_or_not_stim_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_roi_amp,
    _visualize_stimulated_area,
)
from ._single_wells_plots._plot_amplitudes_and_frequencies_data import (
    _plot_amplitude_and_frequency_data,
)
from ._single_wells_plots._plot_iei_data import _plot_iei_data
from ._single_wells_plots._plot_traces_data import _plot_traces_data
from ._single_wells_plots._raster_plots import _generate_raster_plot
from ._single_wells_plots._synchrony_plots import _plot_synchrony_data

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )
    from micromanager_gui._plate_viewer._util import ROIData


# TITLES FOR THE PLOTS THAT WILL BE SHOWN IN THE COMBOBOX
# fmt: off
RAW_TRACES = "Raw Traces"
NORMALIZED_TRACES = "Normalized Traces"
DFF = "DeltaF/F0"
DFF_NORMALIZED = "DeltaF/F0 Normalized"
DEC_DFF = "Deconvolved DeltaF/F0"
DEC_DFF_WITH_PEAKS = "Deconvolved DeltaF/F0 with Peaks"
DEC_DFF_NORMALIZED = "Deconvolved DeltaF/F0 Normalized"
DEC_DFF_NORMALIZED_WITH_PEAKS = "Deconvolved DeltaF/F0 Normalized with Peaks"
DEC_DFF_AMPLITUDE = "Deconvolved DeltaF/F0 Amplitudes"
DEC_DFF_AMPLITUDE_STD = "Deconvolved DeltaF/F0 Amplitudes (Mean ± StD)"
DEC_DFF_AMPLITUDE_SEM = "Deconvolved DeltaF/F0 Amplitudes (Mean ± SEM)"
DEC_DFF_FREQUENCY = "Deconvolved DeltaF/F0 Frequencies"
DEC_DFF_AMPLITUDE_VS_FREQUENCY = "Deconvolved DeltaF/F0 Amplitudes vs Frequencies"
DEC_DFF_AMPLITUDE_STD_VS_FREQUENCY = "Deconvolved DeltaF/F0 Amplitudes (Mean ± StD) vs Frequencies"  # noqa: E501
DEC_DFF_AMPLITUDE_SEM_VS_FREQUENCY = "Deconvolved DeltaF/F0 Amplitudes (Mean ± SEM) vs Frequencies"  # noqa: E501
DEC_DFF_IEI = "Deconvolved DeltaF/F0 Inter-event Interval"
DEC_DFF_IEI_STD = "Deconvolved DeltaF/F0 Inter-event Interval (Mean ± StD)"
DEC_DFF_IEI_SEM = "Deconvolved DeltaF/F0 Inter-event Interval (Mean ± SEM)"
RASTER_PLOT = "Raster plot Colored by ROI"
RASTER_PLOT_AMP = "Raster plot Colored by Amplitude"
RASTER_PLOT_AMP_WITH_COLORBAR = "Raster plot Colored by Amplitude with Colorbar"
STIMULATED_AREA = "Stimulated Area"
STIMULATED_ROIS = "Stimulated vs Non-Stimulated ROIs"
STIMULATED_ROIS_WITH_STIMULATED_AREA = "Stimulated vs Non-Stimulated ROIs with Stimulated Area"  # noqa: E501
STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED = "Stimulated vs Non-Stimulated Normalized (Deconvolved DeltaF/F0)"  # noqa: E501
STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS = "Stimulated vs Non-Stimulated Normalized with Peaks (Deconvolved DeltaF/F0)"  # noqa: E501
STIMULATED_PEAKS_AMP = "Stimulated Peaks Amplitudes"
STIMULATED_PEAKS_AMP_STD = "Stimulated Peaks Amplitudes (Mean ± StD)"
STIMULATED_PEAKS_AMP_SEM = "Stimulated Peaks Amplitudes (Mean ± SEM)"
NON_STIMULATED_PEAKS_AMP = "Non-Stimulated Peaks Amplitudes"
NON_STIMULATED_PEAKS_AMP_STD = "Non-Stimulated Peaks Amplitudes (Mean ± StD)"
NON_STIMULATED_PEAKS_AMP_SEM = "Non-Stimulated Peaks Amplitudes (Mean ± SEM)"
GLOBAL_SYNCHRONY = "Global Synchrony"
GLOBAL_SYNCHRONY_P_VALUE = "Global Synchrony (with p-value)"
# fmt: on


# GROUPS OF PLOTTING OPTIONS (SEE `SINGLE_WELL_COMBO_OPTIONS_DICT` BELOW)
TRACES_GROUP = {
    RAW_TRACES: {},
    NORMALIZED_TRACES: {"normalize": True},
    DFF: {"dff": True},
    DFF_NORMALIZED: {"dff": True, "normalize": True},
    DEC_DFF: {"dec": True},
    DEC_DFF_WITH_PEAKS: {"dec": True, "with_peaks": True},
    DEC_DFF_NORMALIZED: {"dec": True, "normalize": True},
    DEC_DFF_NORMALIZED_WITH_PEAKS: {"dec": True, "normalize": True, "with_peaks": True},
}

AMPLITUDE_GROUP = {
    DEC_DFF_AMPLITUDE: {"amp": True},
    DEC_DFF_AMPLITUDE_STD: {"amp": True, "std": True},
    DEC_DFF_AMPLITUDE_SEM: {"amp": True, "sem": True},
}

FREQUENCY_GROUP = {
    DEC_DFF_FREQUENCY: {"freq": True},
}


AMPLITUDE_AND_FREQUENCY_GROUP = {
    DEC_DFF_AMPLITUDE_VS_FREQUENCY: {"amp": True, "freq": True},
    DEC_DFF_AMPLITUDE_STD_VS_FREQUENCY: {"amp": True, "freq": True, "std": True},
    DEC_DFF_AMPLITUDE_SEM_VS_FREQUENCY: {"amp": True, "freq": True, "sem": True},
}

RASTER_PLOT_GROUP = {
    RASTER_PLOT: {},
    RASTER_PLOT_AMP: {"amplitude_colors": True, "colorbar": False},
    RASTER_PLOT_AMP_WITH_COLORBAR: {"amplitude_colors": True, "colorbar": True},
}

OTHERS_GROUP = {
    DEC_DFF_IEI: {},
    DEC_DFF_IEI_STD: {"std": True},
    DEC_DFF_IEI_SEM: {"sem": True},
    GLOBAL_SYNCHRONY: {},
    # GLOBAL_SYNCHRONY_P_VALUE: {"with_p_value": True},
}

EVOKED_GROUP = {
    STIMULATED_AREA: {"stimulated_area": False},
    STIMULATED_ROIS: {"with_rois": True},
    STIMULATED_ROIS_WITH_STIMULATED_AREA: {"with_rois": True, "stimulated_area": True},
    STIMULATED_PEAKS_AMP: {"stimulated": True},
    STIMULATED_PEAKS_AMP_STD: {"stimulated": True, "std": True},
    STIMULATED_PEAKS_AMP_SEM: {"stimulated": True, "sem": True},
    NON_STIMULATED_PEAKS_AMP: {},
    NON_STIMULATED_PEAKS_AMP_STD: {"std": True},
    NON_STIMULATED_PEAKS_AMP_SEM: {"sem": True},
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED: {},
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS: {"with_peaks": True},
}


# Dictionary to group the options in the graph widgets combobox
# The keys are sections that wont be selectable but are used as dividers
SINGLE_WELL_COMBO_OPTIONS_DICT = {
    "------------Traces---------------------------": TRACES_GROUP.keys(),
    "------------Amplitude------------------------": AMPLITUDE_GROUP.keys(),
    "------------Frequency------------------------": FREQUENCY_GROUP.keys(),
    "------------Amplitude and Frequency----------": AMPLITUDE_AND_FREQUENCY_GROUP.keys(),  # noqa: E501
    "------------Raster Plots---------------------": RASTER_PLOT_GROUP.keys(),
    "------------Others---------------------------": OTHERS_GROUP.keys(),
    "------------Evoked Experiment----------------": EVOKED_GROUP.keys(),
}


MULTI_WELL_COMBO_OPTIONS = [
    DEC_DFF_AMPLITUDE_VS_FREQUENCY,
    DEC_DFF_AMPLITUDE,
    DEC_DFF_FREQUENCY,
    DEC_DFF_IEI,
]
# ------------------------------------------------------------------------------------


def plot_single_well_data(
    widget: _SingleWellGraphWidget,
    data: dict,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot traces based on the text."""
    if not text or text == "None" or text in SINGLE_WELL_COMBO_OPTIONS_DICT.keys():
        return

    # TRACES GROUP
    if text in TRACES_GROUP:
        return _plot_traces_data(widget, data, rois, **TRACES_GROUP[text])

    # AMPLITUDE GROUP
    if text in AMPLITUDE_GROUP:
        return _plot_amplitude_and_frequency_data(
            widget, data, rois, **AMPLITUDE_GROUP[text]
        )

    # FREQUENCY GROUP
    if text in FREQUENCY_GROUP:
        return _plot_amplitude_and_frequency_data(
            widget, data, rois, **FREQUENCY_GROUP[text]
        )

    # AMPLITUDE AND FREQUENCY GROUP
    if text in AMPLITUDE_AND_FREQUENCY_GROUP:
        return _plot_amplitude_and_frequency_data(
            widget, data, rois, **AMPLITUDE_AND_FREQUENCY_GROUP[text]
        )

    # RASTER PLOT GROUP
    if text in RASTER_PLOT_GROUP:
        return _generate_raster_plot(widget, data, rois, **RASTER_PLOT_GROUP[text])

    # EVOKED EXPERIMENT GROUP
    if text in EVOKED_GROUP:
        if text in {
            STIMULATED_AREA,
            STIMULATED_ROIS,
            STIMULATED_ROIS_WITH_STIMULATED_AREA,
        }:
            return _visualize_stimulated_area(widget, data, rois, **EVOKED_GROUP[text])
        elif text in {
            STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS,
            STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED,
        }:
            return _plot_stimulated_vs_non_stimulated_roi_amp(
                widget, data, rois, **EVOKED_GROUP[text]
            )
        elif text in {
            STIMULATED_PEAKS_AMP,
            NON_STIMULATED_PEAKS_AMP,
            STIMULATED_PEAKS_AMP_STD,
            NON_STIMULATED_PEAKS_AMP_STD,
            STIMULATED_PEAKS_AMP_SEM,
            NON_STIMULATED_PEAKS_AMP_SEM,
        }:
            return _plot_stim_or_not_stim_peaks_amplitude(
                widget, data, rois, **EVOKED_GROUP[text]
            )

    # OTHERS GROUP
    if text in OTHERS_GROUP:
        if text in {GLOBAL_SYNCHRONY, GLOBAL_SYNCHRONY_P_VALUE}:
            return _plot_synchrony_data(widget, data, rois, **OTHERS_GROUP[text])
        elif text in {DEC_DFF_IEI, DEC_DFF_IEI_STD, DEC_DFF_IEI_SEM}:
            return _plot_iei_data(widget, data, rois, **OTHERS_GROUP[text])


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
    # return _plot_multi_well_data(
    #     widget, data, positions, **MULTI_WELL_GRAPHS_OPTIONS[text]
    # )
