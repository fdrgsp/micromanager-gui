from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from micromanager_gui._plate_viewer._logger import LOGGER

from ._multi_wells_plots._csv_bar_plot import plot_csv_bar_plot
from ._single_wells_plots._plolt_evoked_experiment_data_plots import (
    _plot_stim_or_not_stim_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_roi_amp,
    _visualize_stimulated_area,
)
from ._single_wells_plots._plot_amplitudes_and_frequencies_data import (
    _plot_amplitude_and_frequency_data,
)
from ._single_wells_plots._plot_cell_size import _plot_cell_size_data
from ._single_wells_plots._plot_correlation import (
    _plot_cross_correlation_data,
    _plot_hierarchical_clustering_data,
)
from ._single_wells_plots._plot_iei_data import _plot_iei_data
from ._single_wells_plots._plot_synchrony import _plot_synchrony_data
from ._single_wells_plots._plot_traces_data import _plot_traces_data
from ._single_wells_plots._raster_plots import _generate_raster_plot

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )


# TITLES FOR THE PLOTS THAT WILL BE SHOWN IN THE COMBOBOX
# fmt: off
RAW_TRACES = "Raw Traces"
NORMALIZED_TRACES = "Normalized Traces"
DFF = "DeltaF/F0"
DFF_NORMALIZED = "DeltaF/F0 Normalized"
DEC_DFF_NORMALIZED_ACTIVE_ONLY = "Deconvolved DeltaF/F0 Normalized (Active Only)"
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
GLOBAL_SYNCHRONY = "Global Synchrony"
CROSS_CORRELATION = "Cross-Correlation"
CLUSTERING = "Hierarchical Clustering"
CLUSTERING_DENDROGRAM = "Hierarchical Clustering (Dendrogram)"
CELL_SIZE = "Cell Size"

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
STIMULATED_PEAKS_FREQ = "Stimulated Peaks Frequencies"
NON_STIMULATED_PEAKS_FREQ = "Non-Stimulated Peaks Frequencies"


# GROUPS OF PLOTTING OPTIONS (SEE `SINGLE_WELL_COMBO_OPTIONS_DICT` BELOW)
TRACES_GROUP = {
    RAW_TRACES: {},
    NORMALIZED_TRACES: {"normalize": True},
    DFF: {"dff": True},
    DFF_NORMALIZED: {"dff": True, "normalize": True},
    DEC_DFF: {"dec": True},
    DEC_DFF_WITH_PEAKS: {"dec": True, "with_peaks": True,}, # "active_only": True default with "with_peaks" # noqa: E501
    DEC_DFF_NORMALIZED: {"dec": True, "normalize": True},
    DEC_DFF_NORMALIZED_ACTIVE_ONLY: {"dec": True, "normalize": True, "active_only": True},  # noqa: E501
    DEC_DFF_NORMALIZED_WITH_PEAKS: {"dec": True, "normalize": True, "with_peaks": True},  # "active_only": True default with "with_peaks" # noqa: E501
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

INTEREVENT_INTERVAL_GROUP = {
    DEC_DFF_IEI: {},
    DEC_DFF_IEI_STD: {"std": True},
    DEC_DFF_IEI_SEM: {"sem": True},
}

CELL_SIZE_GROUP: dict[str, dict] = {
    CELL_SIZE: {}
    }

CORRELATION_GROUP = {
    GLOBAL_SYNCHRONY: {},
    CROSS_CORRELATION: {},
    CLUSTERING: {},
    CLUSTERING_DENDROGRAM: {"use_dendrogram": True},
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
# fmt: on

# SINGLE WELLS PLOTS ------------------------------------------------------------------

# Dictionary to group the options in the graph widgets combobox
# The keys are sections that wont be selectable but are used as dividers
SINGLE_WELL_COMBO_OPTIONS_DICT = {
    "------------Traces---------------------------": TRACES_GROUP.keys(),
    "------------Amplitude------------------------": AMPLITUDE_GROUP.keys(),
    "------------Frequency------------------------": FREQUENCY_GROUP.keys(),
    "------------Amplitude and Frequency----------": AMPLITUDE_AND_FREQUENCY_GROUP.keys(),  # noqa: E501
    "------------Raster Plots---------------------": RASTER_PLOT_GROUP.keys(),
    "------------Interevent Interval--------------": INTEREVENT_INTERVAL_GROUP.keys(),
    "------------Cell Size------------------------": CELL_SIZE_GROUP.keys(),
    "------------Correlation----------------------": CORRELATION_GROUP.keys(),
    "------------Evoked Experiment----------------": EVOKED_GROUP.keys(),
}


def plot_single_well_data(
    widget: _SingleWellGraphWidget,
    data: dict,
    text: str,
    rois: list[int] | None = None,
) -> None:
    """Plot traces based on the text."""
    if not text or text == "None" or text in SINGLE_WELL_COMBO_OPTIONS_DICT.keys():
        widget.figure.clear()
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

    # INTEREVENT_INTERVAL GROUP
    if text in INTEREVENT_INTERVAL_GROUP:
        if text in {GLOBAL_SYNCHRONY}:
            return _plot_synchrony_data(
                widget, data, rois, **INTEREVENT_INTERVAL_GROUP[text]
            )
        elif text in {DEC_DFF_IEI, DEC_DFF_IEI_STD, DEC_DFF_IEI_SEM}:
            return _plot_iei_data(widget, data, rois, **INTEREVENT_INTERVAL_GROUP[text])

    # CELL SIZE GROUP
    if text in CELL_SIZE_GROUP:
        return _plot_cell_size_data(widget, data, rois, **CELL_SIZE_GROUP[text])

    # CORRELATION GROUP
    if text in CORRELATION_GROUP:
        if text == GLOBAL_SYNCHRONY:
            return _plot_synchrony_data(widget, data, rois, **CORRELATION_GROUP[text])
        elif text == CROSS_CORRELATION:
            return _plot_cross_correlation_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text in {CLUSTERING, CLUSTERING_DENDROGRAM}:
            return _plot_hierarchical_clustering_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )

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


# MULTI WELLS PLOTS -------------------------------------------------------------------

# fmt: off
CSV_BAR_PLOT_AMPLITUDE = "Amplitude Bar Plot"
CSV_BAR_PLOT_FREQUENCY = "Frequency Bar Plot"
CSV_BAR_PLOT_IEI = "Inter-event Interval Bar Plot"
CSV_BAR_PLOT_CELL_SIZE = "Cell Size Bar Plot"
CSV_BAR_PLOT_GLOBAL_SYNCHRONY = "Global Synchrony Bar Plot"
CSV_BAR_PLOT_PERCENTAGE_ACTIVE_CELLS = "Percentage of Active Cells"
CSV_BAR_PLOT_STIMULATED_AMPLITUDE = "Stimulated Amplitude Bar Plot"
CSV_BAR_PLOT_NON_STIMULATED_AMPLITUDE = "Non-Stimulated Amplitude Bar Plot"

MW_GENERAL_GROUP = {
    CSV_BAR_PLOT_AMPLITUDE: {"parameter": "Amplitude",  "suffix": "amplitude", "add_to_title": " (Deconvolved ΔF/F)"},  # noqa: E501
    CSV_BAR_PLOT_FREQUENCY: {"parameter": "Frequency",  "suffix": "frequency",  "add_to_title": " (Deconvolved ΔF/F)",  "units": "Hz"},  # noqa: E501
    CSV_BAR_PLOT_IEI: { "parameter": "Inter-Event Interval",  "suffix": "iei",  "add_to_title": " (Deconvolved ΔF/F)",  "units": "Sec"},  # noqa: E501
    CSV_BAR_PLOT_CELL_SIZE: { "parameter": "Cell Size",  "suffix": "cell_size",  "units": "μm²"},  # noqa: E501
    CSV_BAR_PLOT_GLOBAL_SYNCHRONY: {"parameter": "Global Synchrony",  "suffix": "synchrony",  "add_to_title": "(Median)",  "units": "Index"},  # noqa: E501
    CSV_BAR_PLOT_PERCENTAGE_ACTIVE_CELLS: {"parameter": "Percentage of Active Cells", "suffix": "percentage_active"},  # noqa: E501
}

MW_EVOKED_GROUP = {
    CSV_BAR_PLOT_STIMULATED_AMPLITUDE: {"stimulated": True, "parameter": "Amplitude", "suffix": "amplitudes_stimulated_peaks", "add_to_title": " (Stimulated - Deconvolved ΔF/F)"},  # noqa: E501
    CSV_BAR_PLOT_NON_STIMULATED_AMPLITUDE: {"stimulated": False, "parameter": "Amplitude", "suffix": "amplitudes_non_stimulated_peaks", "add_to_title": " (Non-Stimulated - Deconvolved ΔF/F)"},  # noqa: E501
}
# fmt: on


# Dictionary to group the options in the graph widgets combobox
# The keys are sections that wont be selectable but are used as dividers
MULTI_WELL_COMBO_OPTIONS_DICT = {
    "------------General-----------------------": MW_GENERAL_GROUP.keys(),
    "------------Evoked Experiment-------------": MW_EVOKED_GROUP.keys(),
}


def plot_multi_well_data(
    widget: _MultilWellGraphWidget,
    text: str,
    analysis_path: str | None,
) -> None:
    """Plot multi-well data based on the text."""
    if not text or text == "None" or text in MULTI_WELL_COMBO_OPTIONS_DICT.keys():
        widget.figure.clear()
        return

    if not analysis_path:
        widget.figure.clear()
        return

    # MW_GENERAL_GROUP
    if text in MW_GENERAL_GROUP:
        return _plot_csv_bar_plot_data(
            widget, text, analysis_path, **MW_GENERAL_GROUP[text]
        )

    # MW_EVOKED_GROUP
    if text in MW_EVOKED_GROUP:
        return _plot_csv_bar_plot_data(
            widget, text, analysis_path, **MW_EVOKED_GROUP[text]
        )


def _plot_csv_bar_plot_data(
    widget: _MultilWellGraphWidget, text: str, analysis_path: str, **kwargs: Any
) -> None:
    """Helper function to plot CSV bar plot data."""
    suffix = kwargs.get("suffix")
    if not suffix:
        print(f"No suffix provided for {text}.")
        widget.figure.clear()
        return

    # Determine CSV path based on whether it's stimulated data
    stimulated = kwargs.get("stimulated", False)
    if stimulated or "stimulated" in suffix:
        csv_path = Path(analysis_path) / "grouped_evk"
    else:
        csv_path = Path(analysis_path) / "grouped"

    if not csv_path.exists():
        LOGGER.error(f"CSV path {csv_path} does not exist.")
        widget.figure.clear()
        return

    csv_file = next(
        (f for f in csv_path.glob("*.csv") if f.name.endswith(f"_{suffix}.csv")),
        None,
    )

    if not csv_file:
        widget.figure.clear()
        return

    # Create plot options from kwargs, filtering out non-plot parameters
    plot_options = {
        k: v for k, v in kwargs.items() if k not in ["stimulated", "per_led_power"]
    }

    # Special handling for certain plot types that don't use mean_n_sem
    if "synchrony" in suffix:
        return plot_csv_bar_plot(
            widget,
            csv_file,
            plot_options,
            mean_n_sem=False,
        )

    if "percentage_active" in suffix:
        return plot_csv_bar_plot(
            widget,
            csv_file,
            plot_options,
            value_n=True,
        )

    return plot_csv_bar_plot(widget, csv_file, plot_options)
