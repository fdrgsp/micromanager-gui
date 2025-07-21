from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from micromanager_gui._plate_viewer._logger import LOGGER

from ._multi_wells_plots._csv_bar_plot import plot_csv_bar_plot
from ._single_wells_plots._plolt_evoked_experiment_data_plots import (
    _plot_stim_or_not_stim_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_roi_amp,
    _plot_stimulated_vs_non_stimulated_spike_traces,
    _visualize_stimulated_area,
)
from ._single_wells_plots._plot_calcium_amplitudes_and_frequencies_data import (
    _plot_amplitude_and_frequency_data,
)
from ._single_wells_plots._plot_calcium_network_connectivity import (
    _plot_connectivity_matrix_data,
    _plot_connectivity_network_data,
)
from ._single_wells_plots._plot_calcium_peaks_correlation import (
    _plot_cross_correlation_data,
    _plot_hierarchical_clustering_data,
)
from ._single_wells_plots._plot_calcium_peaks_iei_data import _plot_iei_data
from ._single_wells_plots._plot_calcium_peaks_raster_plots import _generate_raster_plot
from ._single_wells_plots._plot_calcium_peaks_synchrony import (
    _plot_peak_event_synchrony_data,
)
from ._single_wells_plots._plot_calcium_traces_data import _plot_traces_data
from ._single_wells_plots._plot_cell_size import _plot_cell_size_data
from ._single_wells_plots._plot_inferred_spike_burst_activity import (
    _plot_inferred_spike_burst_activity,
)
from ._single_wells_plots._plot_inferred_spike_correlation import (
    _plot_spike_cross_correlation_data,
    _plot_spike_hierarchical_clustering_data,
)
from ._single_wells_plots._plot_inferred_spike_raster_plots import (
    _generate_spike_raster_plot,
)
from ._single_wells_plots._plot_inferred_spike_synchrony import (
    _plot_spike_synchrony_data,
)
from ._single_wells_plots._plot_inferred_spikes import (
    _plot_inferred_spikes,
    _plot_inferred_spikes_normalized_with_bursts,
)

if TYPE_CHECKING:
    from micromanager_gui._plate_viewer._graph_widgets import (
        _MultilWellGraphWidget,
        _SingleWellGraphWidget,
    )


# TITLES FOR THE PLOTS THAT WILL BE SHOWN IN THE COMBOBOX
# fmt: off
RAW_TRACES = "Calcium Raw Traces"
NORMALIZED_TRACES = "Calcium Normalized Traces"
DFF = "Calcium ΔF/F0 Traces"
DFF_NORMALIZED = "Calcium ΔF/F0 Normalized  Traces "
DEC_DFF_NORMALIZED_ACTIVE_ONLY = "Calcium Deconvolved ΔF/F0 Traces Normalized (Active Only)"  # noqa: E501
DEC_DFF = "Calcium Deconvolved ΔF/F0 Traces"
DEC_DFF_WITH_PEAKS = "Calcium Deconvolved ΔF/F0 Traces with Peaks"
DEC_DFF_WITH_PEAKS_AND_THRESHOLDS = "Calcium Deconvolved ΔF/F0 Traces with Peaks and Thresholds (If 1 ROI selected)"  # noqa: E501
DEC_DFF_NORMALIZED = "Calcium Deconvolved ΔF/F0 Normalized Traces "
DEC_DFF_NORMALIZED_WITH_PEAKS = "Calcium Deconvolved ΔF/F0 Normalized Traces with Peaks"
DEC_DFF_AMPLITUDE = "Calcium Peaks Amplitudes (Deconvolved ΔF/F0)"
DEC_DFF_FREQUENCY = "Calcium Peaks Frequencies (Deconvolved ΔF/F0)"
DEC_DFF_AMPLITUDE_VS_FREQUENCY = "Calcium Peaks Amplitudes vs Frequencies (Deconvolved ΔF/F0)"  # noqa: E501
DEC_DFF_IEI = "Calcium Peaks Inter-event Interval (Deconvolved ΔF/F0)"
INFERRED_SPIKES_RAW = "Inferred Spikes Raw"
INFERRED_SPIKES_THRESHOLDED = "Inferred Spikes (Thresholded)"
INFERRED_SPIKES_RAW_WITH_THRESHOLD = "Inferred Spikes Raw (with Thresholds - If 1 ROI selected)"  # noqa: E501
INFERRED_SPIKES_THRESHOLDED_WITH_DEC_DFF = "Inferred Spikes (Thresholded) with Deconvolved ΔF/F0 Traces"  # noqa: E501
INFERRED_SPIKES_THRESHOLDED_NORMALIZED = "Inferred Spikes (Thresholded) Normalized"
INFERRED_SPIKES_THRESHOLDED_ACTIVE_ONLY = "Inferred Spikes (Thresholded) Normalized (Active Only)"  # noqa: E501
INFERRED_SPIKES_NORMALIZED_WITH_BURSTS = "Inferred Spikes (Thresholded) Normalized with Network Bursts"  # noqa: E501
INFERRED_SPIKES_THRESHOLDED_SYNCHRONY = "Inferred Spikes (Thresholded) Global Synchrony"
INFERRED_SPIKE_CROSS_CORRELATION = "Inferred Spikes (Thresholded) Cross-Correlation"
INFERRED_SPIKE_CLUSTERING = "Inferred Spikes (Thresholded) Hierarchical Clustering"
INFERRED_SPIKE_CLUSTERING_DENDROGRAM = "Inferred Spikes (Thresholded) Hierarchical Clustering (Dendrogram)"  # noqa: E501
INFERRED_SPIKE_BURST_ANALYSIS = "Inferred Spikes (Thresholded) Burst Activity Analysis"
RASTER_PLOT = "Calcium Peaks Raster plot Colored by ROI"
RASTER_PLOT_AMP = "Calcium Peaks Raster plot Colored by Amplitude"
RASTER_PLOT_AMP_WITH_COLORBAR = "Calcium Peaks Raster plot Colored by Amplitude with Colorbar"  # noqa: E501
INFERRED_SPIKE_RASTER_PLOT = "Inferred Spikes Raster plot Colored by ROI"
INFERRED_SPIKE_RASTER_PLOT_AMP = "Inferred Spikes Raster plot Colored by Amplitude"
INFERRED_SPIKE_RASTER_PLOT_AMP_WITH_COLORBAR = "Inferred Spikes Raster plot Colored by Amplitude with Colorbar"  # noqa: E501
STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES = "Stimulated vs Non-Stimulated Spike Traces"
CALCIUM_PEAKS_GLOBAL_SYNCHRONY = "Calcium Peaks Global Synchrony"
CALCIUM_NETWORK_CONNECTIVITY = "Calcium Network Connectivity"
CALCIUM_CONNECTIVITY_MATRIX = "Calcium Network Connectivity Matrix"
CROSS_CORRELATION = "Calcium Peaks Cross-Correlation"
CLUSTERING = "Calcium Peaks Hierarchical Clustering"
CLUSTERING_DENDROGRAM = "Calcium Peaks Hierarchical Clustering (Dendrogram)"
CELL_SIZE = "Cell Size"

STIMULATED_AREA = "Stim Area"
STIMULATED_ROIS = "Stim vs Non-Stim ROIs"
STIMULATED_ROIS_WITH_STIMULATED_AREA = "Stim vs Non-Stim ROIs with Stim Area"
STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED = "Stim vs Non-Stim Normalized Calcium Traces (Deconvolved ΔF/F0)"  # noqa: E501
STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS = "Stim vs Non-Stim Normalized Calcium Traces with Peaks (Deconvolved ΔF/F0)"  # noqa: E501
STIMULATED_PEAKS_AMP = "Stim Calcium Peaks Amplitudes"
NON_STIMULATED_PEAKS_AMP = "Non-Stim Calcium Peaks Amplitudes"
STIMULATED_PEAKS_FREQ = "Stim Calcium Peaks Frequencies"
NON_STIMULATED_PEAKS_FREQ = "Non-Stim Calcium Peaks Frequencies"


# GROUPS OF PLOTTING OPTIONS (SEE `SINGLE_WELL_COMBO_OPTIONS_DICT` BELOW)
CALCIUM_TRACES_GROUP = {
    RAW_TRACES: {},
    NORMALIZED_TRACES: {"normalize": True},
    DFF: {"dff": True},
    DFF_NORMALIZED: {"dff": True, "normalize": True},
    DEC_DFF: {"dec": True},
    DEC_DFF_WITH_PEAKS: {"dec": True, "with_peaks": True,},
    DEC_DFF_WITH_PEAKS_AND_THRESHOLDS: {"dec": True, "with_peaks": True, "thresholds": True},  # noqa: E501
    DEC_DFF_NORMALIZED: {"dec": True, "normalize": True},
    DEC_DFF_NORMALIZED_ACTIVE_ONLY: {"dec": True, "normalize": True, "active_only": True},  # noqa: E501
    DEC_DFF_NORMALIZED_WITH_PEAKS: {"dec": True, "normalize": True, "with_peaks": True},
}

INFERRED_SPIKES_GROUP = {
    INFERRED_SPIKES_RAW: {"raw": True},
    INFERRED_SPIKES_THRESHOLDED: {},
    INFERRED_SPIKES_RAW_WITH_THRESHOLD: {"raw": True, "thresholds": True},
    INFERRED_SPIKES_THRESHOLDED_NORMALIZED: {"normalize": True},
    INFERRED_SPIKES_THRESHOLDED_ACTIVE_ONLY: {"normalize": True, "active_only": True},
    INFERRED_SPIKES_NORMALIZED_WITH_BURSTS: {},
    INFERRED_SPIKES_THRESHOLDED_WITH_DEC_DFF: {"dec_dff": True},
}

AMPLITUDE_AND_FREQUENCY_GROUP = {
    DEC_DFF_AMPLITUDE: {"amp": True},
    DEC_DFF_FREQUENCY: {"freq": True},
    DEC_DFF_AMPLITUDE_VS_FREQUENCY: {"amp": True, "freq": True},
}

RASTER_PLOT_GROUP = {
    RASTER_PLOT: {},
    RASTER_PLOT_AMP: {"amplitude_colors": True, "colorbar": False},
    RASTER_PLOT_AMP_WITH_COLORBAR: {"amplitude_colors": True, "colorbar": True},
    INFERRED_SPIKE_RASTER_PLOT: {},
    INFERRED_SPIKE_RASTER_PLOT_AMP: {"amplitude_colors": True, "colorbar": False},
    INFERRED_SPIKE_RASTER_PLOT_AMP_WITH_COLORBAR: {"amplitude_colors": True, "colorbar": True},  # noqa: E501
}

INTEREVENT_INTERVAL_GROUP: dict[str, dict] = {
    DEC_DFF_IEI: {},
}

CELL_SIZE_GROUP: dict[str, dict] = {
    CELL_SIZE: {}
    }

CORRELATION_GROUP = {
    CALCIUM_PEAKS_GLOBAL_SYNCHRONY: {},
    CALCIUM_NETWORK_CONNECTIVITY: {},
    CALCIUM_CONNECTIVITY_MATRIX: {},
    CROSS_CORRELATION: {},
    CLUSTERING: {},
    CLUSTERING_DENDROGRAM: {"use_dendrogram": True},
    INFERRED_SPIKES_THRESHOLDED_SYNCHRONY: {},
    INFERRED_SPIKE_CROSS_CORRELATION: {},
    INFERRED_SPIKE_CLUSTERING: {},
    INFERRED_SPIKE_CLUSTERING_DENDROGRAM: {"use_dendrogram": True},
    INFERRED_SPIKE_BURST_ANALYSIS: {},
    }

EVOKED_GROUP = {
    STIMULATED_AREA: {"stimulated_area": False},
    STIMULATED_ROIS: {"with_rois": True},
    STIMULATED_ROIS_WITH_STIMULATED_AREA: {"with_rois": True, "stimulated_area": True},
    STIMULATED_PEAKS_AMP: {"stimulated": True},
    NON_STIMULATED_PEAKS_AMP: {},
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED: {},
    STIMULATED_VS_NON_STIMULATED_DEC_DFF_NORMALIZED_WITH_PEAKS: {"with_peaks": True},
    STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES: {},
}
# fmt: on

# SINGLE WELLS PLOTS ------------------------------------------------------------------

# Dictionary to group the options in the graph widgets combobox
# The keys are sections that wont be selectable but are used as dividers
SINGLE_WELL_COMBO_OPTIONS_DICT = {
    "----------Calcium Traces-----------------------------------": CALCIUM_TRACES_GROUP.keys(),  # noqa: E501
    "----------Calcium Peaks Amplitude and Frequency---------": AMPLITUDE_AND_FREQUENCY_GROUP.keys(),  # noqa: E501
    "----------Calcium Peaks Interevent Interval----------------": INTEREVENT_INTERVAL_GROUP.keys(),  # noqa: E501
    "----------Inferred Spikes Traces---------------------------": INFERRED_SPIKES_GROUP.keys(),  # noqa: E501
    "----------Raster Plots-------------------------------------": RASTER_PLOT_GROUP.keys(),  # noqa: E501
    "----------Correlation Analysis-----------------------------": CORRELATION_GROUP.keys(),  # noqa: E501
    "----------Cell Size----------------------------------------": CELL_SIZE_GROUP.keys(),  # noqa: E501
    "----------Evoked Experiment------------------------------": EVOKED_GROUP.keys(),
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
    if text in CALCIUM_TRACES_GROUP:
        return _plot_traces_data(widget, data, rois, **CALCIUM_TRACES_GROUP[text])

    # AMPLITUDE AND FREQUENCY GROUP
    if text in AMPLITUDE_AND_FREQUENCY_GROUP:
        return _plot_amplitude_and_frequency_data(
            widget, data, rois, **AMPLITUDE_AND_FREQUENCY_GROUP[text]
        )

    # RASTER PLOT GROUP
    if text in RASTER_PLOT_GROUP:
        if text in {RASTER_PLOT, RASTER_PLOT_AMP, RASTER_PLOT_AMP_WITH_COLORBAR}:
            return _generate_raster_plot(widget, data, rois, **RASTER_PLOT_GROUP[text])
        else:
            return _generate_spike_raster_plot(
                widget, data, rois, **RASTER_PLOT_GROUP[text]
            )

    # INFERRED SPIKES GROUP
    if text in INFERRED_SPIKES_GROUP and text in {
        INFERRED_SPIKES_RAW,
        INFERRED_SPIKES_THRESHOLDED,
        INFERRED_SPIKES_RAW_WITH_THRESHOLD,
        INFERRED_SPIKES_THRESHOLDED_NORMALIZED,
        INFERRED_SPIKES_THRESHOLDED_ACTIVE_ONLY,
        INFERRED_SPIKES_THRESHOLDED_WITH_DEC_DFF,
    }:
        return _plot_inferred_spikes(widget, data, rois, **INFERRED_SPIKES_GROUP[text])

    if text == INFERRED_SPIKES_NORMALIZED_WITH_BURSTS:
        return _plot_inferred_spikes_normalized_with_bursts(
            widget, data, rois, **INFERRED_SPIKES_GROUP[text]
        )

    # INTEREVENT_INTERVAL GROUP
    if text in INTEREVENT_INTERVAL_GROUP:
        return _plot_iei_data(widget, data, rois, **INTEREVENT_INTERVAL_GROUP[text])

    # CELL SIZE GROUP
    if text in CELL_SIZE_GROUP:
        return _plot_cell_size_data(widget, data, rois, **CELL_SIZE_GROUP[text])

    # CORRELATION GROUP
    if text in CORRELATION_GROUP:
        # if text == GLOBAL_SYNCHRONY:
        #     return _plot_synchrony_data(widget, data, rois, **CORRELATION_GROUP[text])
        if text == CALCIUM_PEAKS_GLOBAL_SYNCHRONY:
            return _plot_peak_event_synchrony_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text == CALCIUM_NETWORK_CONNECTIVITY:
            return _plot_connectivity_network_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text == CALCIUM_CONNECTIVITY_MATRIX:
            return _plot_connectivity_matrix_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text == CROSS_CORRELATION:
            return _plot_cross_correlation_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text in {CLUSTERING, CLUSTERING_DENDROGRAM}:
            return _plot_hierarchical_clustering_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text == INFERRED_SPIKES_THRESHOLDED_SYNCHRONY:
            return _plot_spike_synchrony_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text == INFERRED_SPIKE_CROSS_CORRELATION:
            return _plot_spike_cross_correlation_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text in {INFERRED_SPIKE_CLUSTERING, INFERRED_SPIKE_CLUSTERING_DENDROGRAM}:
            return _plot_spike_hierarchical_clustering_data(
                widget, data, rois, **CORRELATION_GROUP[text]
            )
        elif text == INFERRED_SPIKE_BURST_ANALYSIS:
            return _plot_inferred_spike_burst_activity(
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

        elif text in {STIMULATED_PEAKS_AMP, NON_STIMULATED_PEAKS_AMP}:
            return _plot_stim_or_not_stim_peaks_amplitude(
                widget, data, rois, **EVOKED_GROUP[text]
            )

        elif text == STIMULATED_VS_NON_STIMULATED_SPIKE_TRACES:
            return _plot_stimulated_vs_non_stimulated_spike_traces(
                widget, data, rois, **EVOKED_GROUP[text]
            )


# MULTI WELLS PLOTS -------------------------------------------------------------------

# fmt: off
CSV_BAR_PLOT_AMPLITUDE = "Calcium Peaks Amplitude Bar Plot"
CSV_BAR_PLOT_FREQUENCY = "Calcium Peaks Frequency Bar Plot"
CSV_BAR_PLOT_IEI = "Calcium Peaks Inter-event Interval Bar Plot"
CSV_BAR_PLOT_CELL_SIZE = "Cell Size Bar Plot"
CSV_BAR_PLOT_CALCIUM_PEAKS_EVENT_SYNCHRONY = "Calcium Peaks Events Global Synchrony Bar Plot"  # noqa: E501
CSV_BAR_PLOT_INFERRED_SPIKE_SYNCHRONY = "Inferred Spikes Global Synchrony Bar Plot"
CSV_BAR_PLOT_PERCENTAGE_ACTIVE_CELLS = "Percentage of Active Cells (Based on Calcium Peaks) Bar Plot"  # noqa: E501
# CSV_BAR_PLOT_CALCIUM_NETWORK_DENSITY = "Calcium Network Density Bar Plot"
CSV_BAR_PLOT_BURST_COUNT = "Burst Count Bar Plot"
CSV_BAR_PLOT_BURST_DURATION = "Burst Average Duration Bar Plot"
CSV_BAR_PLOT_BURST_INTERVAL = "Burst Average Interval Bar Plot"
CSV_BAR_PLOT_BURST_RATE = "Burst Rate Bar Plot"
CSV_BAR_PLOT_CALCIUM_STIMULATED_AMPLITUDE = "Stimulated Calcium Peaks Amplitude Bar Plot"  # noqa: E501
CSV_BAR_PLOT_CALCIUM_NON_STIMULATED_AMPLITUDE = "Non-Stimulated Calcium Peaks Amplitude Bar Plot"  # noqa: E501

MW_GENERAL_GROUP = {
    CSV_BAR_PLOT_CELL_SIZE: { "parameter": "Cell Size",  "suffix": "cell_size",  "units": "μm²"},  # noqa: E501
    CSV_BAR_PLOT_PERCENTAGE_ACTIVE_CELLS: {"parameter": "Percentage of Active Cells", "suffix": "percentage_active", "add_to_title": "Based on Calcium Peaks"},  # noqa: E501
    CSV_BAR_PLOT_AMPLITUDE: {"parameter": "Calcium Peaks Amplitude",  "suffix": "amplitude", "add_to_title": " (Deconvolved ΔF/F)"},  # noqa: E501
    CSV_BAR_PLOT_FREQUENCY: {"parameter": "Calcium Peaks Frequency",  "suffix": "frequency",  "add_to_title": " (Deconvolved ΔF/F)",  "units": "Hz"},  # noqa: E501
    CSV_BAR_PLOT_IEI: { "parameter": "Calcium Peaks Inter-Event Interval",  "suffix": "iei",  "add_to_title": " (Deconvolved ΔF/F)",  "units": "Sec"},  # noqa: E501
    CSV_BAR_PLOT_CALCIUM_PEAKS_EVENT_SYNCHRONY: {"parameter": "Calcium Peak Events Global Synchrony",  "suffix": "calcium_peaks_synchrony",  "add_to_title": "(Median)",  "units": "Index"},  # noqa: E501
    CSV_BAR_PLOT_INFERRED_SPIKE_SYNCHRONY: {"parameter": "Inferred Spikes Global Synchrony",  "suffix": "spike_synchrony",  "add_to_title": "(Median - Thresholded Data)",  "units": "Index"},  # noqa: E501
    # CSV_BAR_PLOT_CALCIUM_NETWORK_DENSITY: {"parameter": "Calcium Network Density",  "suffix": "calcium_network_density",  "add_to_title": "(Percentile-Based Threshold)",  "units": "%"},  # noqa: E501
    CSV_BAR_PLOT_BURST_COUNT: {"parameter": "Burst Count",  "suffix": "burst_activity",  "burst_metric": "count",  "add_to_title": "(Inferred Spikes)",  "units": "Count"},  # noqa: E501
    CSV_BAR_PLOT_BURST_DURATION: {"parameter": "Burst Average Duration",  "suffix": "burst_activity",  "burst_metric": "avg_duration_sec",  "add_to_title": "(Inferred Spikes)",  "units": "Sec"},  # noqa: E501
    CSV_BAR_PLOT_BURST_INTERVAL: {"parameter": "Burst Average Interval",  "suffix": "burst_activity",  "burst_metric": "avg_interval_sec",  "add_to_title": "(Inferred Spikes)",  "units": "Sec"},  # noqa: E501
    CSV_BAR_PLOT_BURST_RATE: {"parameter": "Burst Rate",  "suffix": "burst_activity",  "burst_metric": "rate_burst_per_min",  "add_to_title": "(Inferred Spikes)",  "units": "Bursts/min"},  # noqa: E501
}

MW_EVOKED_GROUP = {
    CSV_BAR_PLOT_CALCIUM_STIMULATED_AMPLITUDE: {"stimulated": True, "parameter": "Calcium Peaks Amplitude", "suffix": "calcium_peaks_amplitudes_stimulated", "add_to_title": " (Stimulated - Deconvolved ΔF/F)"},  # noqa: E501
    CSV_BAR_PLOT_CALCIUM_NON_STIMULATED_AMPLITUDE: {"stimulated": False, "parameter": "Calcium Peaks Amplitude", "suffix": "calcium_peaks_amplitudes_non_stimulated", "add_to_title": " (Non-Stimulated - Deconvolved ΔF/F)"},  # noqa: E501
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
        LOGGER.error(f"CSV file for suffix '{suffix}' not found in {csv_path}.")
        widget.figure.clear()
        return

    # Create plot options from kwargs, filtering out non-plot parameters
    plot_options = {
        k: v
        for k, v in kwargs.items()
        if k not in ["stimulated", "per_led_power", "burst_metric"]
    }

    # Special handling for burst activity plots
    burst_metric = kwargs.get("burst_metric")
    if suffix == "burst_activity" and burst_metric:
        # Add burst_metric to plot_options for handling in plot_csv_bar_plot
        plot_options["burst_metric"] = burst_metric
        return plot_csv_bar_plot(
            widget,
            csv_file,
            plot_options,
            mean_n_sem=False,
        )

    # Special handling for certain plot types that don't use mean_n_sem
    synchrony_suffixes = [
        "synchrony",
        "spike_synchrony",
        "calcium_network_density",
        "calcium_peaks_synchrony" "",
    ]
    if any(sync_suffix in suffix for sync_suffix in synchrony_suffixes):
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
