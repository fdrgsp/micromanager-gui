from __future__ import annotations

import re
from itertools import zip_longest
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from ._logger import LOGGER
from ._plot_methods._single_wells_plots._plot_inferred_spike_burst_activity import (
    _detect_population_bursts,
    _get_burst_parameters,
    _get_population_spike_data,
)
from ._util import (
    EVK_NON_STIM,
    EVK_STIM,
    MEAN_SUFFIX,
    N_SUFFIX,
    SEM_SUFFIX,
    ROIData,
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_calcium_peaks_events_from_rois,
    _get_spike_synchrony,
    _get_spike_synchrony_matrix,
    _get_spikes_over_threshold,
)

# fmt: off
NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
PERCENTAGE_ACTIVE = "percentage_active"
SPIKE_SYNCHRONY = "spike_synchrony"
CALCIUM_PEAKS_SYNCHRONY = "calcium_peaks_synchrony"
AMP_STIMULATED_PEAKS = "calcium_peaks_amplitudes_stimulated"
AMP_NON_STIMULATED_PEAKS = "calcium_peaks_amplitudes_non_stimulated"
BURST_ACTIVITY = "burst_activity"
CSV_PARAMETERS: dict[str, str] = {
    "calcium_peaks_amplitude": "peaks_amplitudes_dec_dff",
    "calcium_peaks_frequency": "dec_dff_frequency",
    "cell_size": "cell_size",
    "calcium_peaks_iei": "iei",
    "percentage_active": PERCENTAGE_ACTIVE,
    "calcium_peaks_synchrony": CALCIUM_PEAKS_SYNCHRONY,
    "spike_synchrony": SPIKE_SYNCHRONY,
    "burst_activity": BURST_ACTIVITY,
}
CSV_PARAMETERS_EVK = {
    "calcium_peaks_amplitudes_stimulated": AMP_STIMULATED_PEAKS,
    "calcium_peaks_amplitudes_non_stimulated": AMP_NON_STIMULATED_PEAKS
}

PARAMETER_TO_KEY: dict[str, str] = {
    **{v: k for k, v in CSV_PARAMETERS.items()},
    **{v: k for k, v in CSV_PARAMETERS_EVK.items()},
}

SINGLE_VALUES = [
    PERCENTAGE_ACTIVE,
    SPIKE_SYNCHRONY,
    CALCIUM_PEAKS_SYNCHRONY,
    BURST_ACTIVITY,
]
# fmt: on


def save_trace_data_to_csv(
    path: str | Path,
    analysis_data: dict[str, dict[str, ROIData]] | None,
) -> None:
    if not analysis_data:
        return

    LOGGER.info(f"Exporting data to `{path}`...")
    try:
        _export_raw_data(path, analysis_data)
    except Exception as e:
        LOGGER.error(f"Error exporting RAW DATA to CSV: {e}")
    try:
        _export_dff_data(path, analysis_data)
    except Exception as e:
        LOGGER.error(f"Error exporting dFF DATA to CSV: {e}")
    try:
        _export_dec_dff_data(path, analysis_data)
    except Exception as e:
        LOGGER.error(f"Error exporting DEC_DFF DATA to CSV: {e}")
    try:
        _export_inferred_spikes_data(path, analysis_data)
    except Exception as e:
        LOGGER.error(f"Error exporting INFERRED RAW SPIKES DATA to CSV: {e}")
    try:
        _export_inferred_spikes_data(path, analysis_data, raw=False)
    except Exception as e:
        LOGGER.error(f"Error exporting INFERRED THRESHOLDED SPIKES DATA to CSV: {e}")
    LOGGER.info("Exporting data to CSV: DONE!")


def save_analysis_data_to_csv(
    path: str | Path,
    analysis_data: dict[str, dict[str, ROIData]] | None,
) -> None:
    """Save the analysis data as CSV files."""
    if not analysis_data:
        return
    if isinstance(path, str):
        path = Path(path)

    rearrange_cond, rearrange_cond_evk = _rearrange_data(analysis_data)

    msg = f"Exporting data to `{path}`..."
    LOGGER.info(msg)
    try:
        _export_to_csv_mean_values_grouped_by_condition(path, rearrange_cond)
    except Exception as e:
        LOGGER.error(f"Error exporting spontanoous analysis data to CSV: {e}")
    try:
        _export_to_csv_mean_values_evk_parameters(path, rearrange_cond_evk)
    except Exception as e:
        LOGGER.error(f"Error exporting evoked analysis data to CSV: {e}")
    LOGGER.info("Exporting data to CSV: DONE!")


def _rearrange_data(analysis_data: dict[str, dict[str, ROIData]]) -> tuple:
    """Rearrange the analysis data by condition and parameter."""
    # Rearrange the data by condition
    fov_by_condition, evk_conditions = _rearrange_fov_by_conditions(analysis_data)
    # Rearrange fov_by_condition by parameter
    fov_by_condition_by_parameter = {
        parameter: _rearrange_by_parameter(fov_by_condition, parameter)
        for parameter in CSV_PARAMETERS.values()
    }
    # Rearrange fov_by_condition by evoked parameters
    fov_by_condition_by_parameter_evk = {
        parameter: _rearrange_by_parameter_evk(evk_conditions, parameter)
        for parameter in CSV_PARAMETERS_EVK.values()
    }
    return fov_by_condition_by_parameter, fov_by_condition_by_parameter_evk


def _rearrange_fov_by_conditions(
    data: dict[str, dict[str, ROIData]],
) -> tuple[
    dict[str, dict[str, dict[str, ROIData]]], dict[str, dict[str, dict[str, ROIData]]]
]:
    """Rearrange the data by condition.

    Parameters
    ----------
    data: dict[str, dict[str, dict[str, ROIData]]
        The data to rearrange.
    """
    conds: dict[str, dict[str, dict[str, ROIData]]] = {}
    evoked_conds: dict[str, dict[str, dict[str, ROIData]]] = {}
    for well_fov, rois in data.items():  # "key1", "key2", ...
        for roi_key, roi_data in rois.items():  #  ("1", ROIData), ("2", ROIData), ...
            c1 = roi_data.condition_1
            c2 = roi_data.condition_2
            if c1 and c2:
                cond_key = f"{c1}_{c2}"
            elif c1:
                cond_key = c1
            elif c2:
                cond_key = c2
            else:
                cond_key = "NoCondition"
            conds.setdefault(cond_key, {}).setdefault(well_fov, {})[roi_key] = roi_data

            # update the evoked conditions dict
            if roi_data.evoked_experiment:
                from ._util import get_stimulated_amplitudes_from_roi_data

                # Compute amplitudes on-demand (without LED power equation for now)
                amps_stim, amps_non_stim = get_stimulated_amplitudes_from_roi_data(
                    roi_data, led_power_equation=None
                )

                if roi_data.stimulated:
                    amps_dict = amps_stim
                    stim_label = EVK_STIM
                else:
                    amps_dict = amps_non_stim
                    stim_label = EVK_NON_STIM

                if amps_dict:
                    for power_and_pulse in amps_dict:
                        key = f"{cond_key}_{stim_label}_{power_and_pulse}"
                        evoked_conds.setdefault(key, {}).setdefault(well_fov, {})[
                            roi_key
                        ] = roi_data

    return conds, evoked_conds


def _rearrange_by_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
    parameter: str,
) -> dict[str, dict[str, list[Any]]]:
    """Create a dict grouped by the specified parameter per condition."""
    if parameter == PERCENTAGE_ACTIVE:
        try:
            return _get_percentage_active_parameter(data)
        except Exception as e:
            LOGGER.error(f"Error calculating percentage active: {e}")
            return {}
    if parameter == SPIKE_SYNCHRONY:
        try:
            return _get_spike_synchrony_parameter(data)
        except Exception as e:
            LOGGER.error(f"Error calculating spike synchrony: {e}")
            return {}
    if parameter == CALCIUM_PEAKS_SYNCHRONY:
        try:
            return _get_calcium_peaks_event_synchrony_parameter(data)
        except Exception as e:
            LOGGER.error(f"Error calculating peak event synchrony: {e}")
            return {}
    if parameter == BURST_ACTIVITY:
        try:
            return _get_burst_activity_parameter(data)
        except Exception as e:
            LOGGER.error(f"Error calculating burst activity: {e}")
            return {}
    try:
        return _get_parameter(data, parameter)
    except Exception as e:
        LOGGER.error(f"Error calculating parameter '{parameter}': {e}")
        return {}


def _rearrange_by_parameter_evk(
    data: dict[str, dict[str, dict[str, ROIData]]],
    parameter: str,
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Create a dict grouped by the specified parameter per condition."""
    # AMPLITUDE STIMULATED
    if parameter == AMP_STIMULATED_PEAKS:
        try:
            return _get_amplitude_stim_or_non_stim_peaks_parameter(data)
        except Exception as e:
            LOGGER.error(f"Error calculating stimulated peaks: {e}")
            return {}

    # AMPLITUDE NON-STIMULATED
    if parameter == AMP_NON_STIMULATED_PEAKS:
        try:
            return _get_amplitude_stim_or_non_stim_peaks_parameter(
                data, stimulated=False
            )
        except Exception as e:
            LOGGER.error(f"Error calculating non-stimulated peaks: {e}")
            return {}
    return {}


def _export_raw_data(path: Path | str, data: dict[str, dict[str, ROIData]]) -> None:
    """Save the raw data as CSV files.

    Columns are frames and rows are ROIs.
    """
    path = Path(path)
    exp_name = path.stem
    folder = path / "raw_data"
    folder.mkdir(parents=True, exist_ok=True)

    # store traces by well_fov and roi_key
    rows = {}
    for well_fov, rois in data.items():
        for roi_key, roi_data in rois.items():
            if roi_data.raw_trace is None:
                continue
            row_name = f"{well_fov}_{roi_key}"
            rows[row_name] = roi_data.raw_trace

    if not rows:
        return

    # convert to DataFrame (handles unequal lengths by filling with NaN)
    df = pd.DataFrame.from_dict(rows, orient="index")

    # give the columns t0, t1, t2, ... name
    df.columns = [f"t{i}" for i in range(df.shape[1])]

    # save to CSV
    df.to_csv(folder / f"{exp_name}_raw_data.csv", index=True)


def _export_dff_data(path: Path | str, data: dict[str, dict[str, ROIData]]) -> None:
    """Save the dFF data as CSV files.

    Columns are frames and rows are ROIs.
    """
    path = Path(path)
    exp_name = path.stem
    folder = path / "dff_data"
    folder.mkdir(parents=True, exist_ok=True)

    # store traces by well_fov and roi_key
    rows = {}
    for well_fov, rois in data.items():
        for roi_key, roi_data in rois.items():
            if roi_data.dff is None:
                continue
            row_name = f"{well_fov}_{roi_key}"
            rows[row_name] = roi_data.dff

    # convert to DataFrame (handles unequal lengths by filling with NaN)
    df = pd.DataFrame.from_dict(rows, orient="index")

    # give the columns t0, t1, t2, ... name
    df.columns = [f"t{i}" for i in range(df.shape[1])]

    # save to CSV
    df.to_csv(folder / f"{exp_name}_dff_data.csv", index=True)


def _export_dec_dff_data(path: Path | str, data: dict[str, dict[str, ROIData]]) -> None:
    """Save the dec_dFF data as CSV files.

    Columns are frames and rows are ROIs.
    """
    path = Path(path)
    exp_name = path.stem
    folder = path / "dec_dff_data"
    folder.mkdir(parents=True, exist_ok=True)

    # store traces by well_fov and roi_key
    rows = {}
    for well_fov, rois in data.items():
        for roi_key, roi_data in rois.items():
            if roi_data.dec_dff is None:
                continue
            row_name = f"{well_fov}_{roi_key}"
            rows[row_name] = roi_data.dec_dff

    # convert to DataFrame (handles unequal lengths by filling with NaN)
    df = pd.DataFrame.from_dict(rows, orient="index")

    # give the columns t0, t1, t2, ... name
    df.columns = [f"t{i}" for i in range(df.shape[1])]

    # save to CSV
    df.to_csv(folder / f"{exp_name}_dec_dff_data.csv", index=True)


def _export_inferred_spikes_data(
    path: Path | str, data: dict[str, dict[str, ROIData]], raw: bool = True
) -> None:
    """Save the inferred spikes data as CSV files.

    Columns are frames and rows are ROIs.
    """
    path = Path(path)
    exp_name = path.stem
    folder = path / "inferred_spikes_data"
    folder.mkdir(parents=True, exist_ok=True)

    # store traces by well_fov and roi_key
    rows = {}
    for well_fov, rois in data.items():
        for roi_key, roi_data in rois.items():
            if (spikes := _get_spikes_over_threshold(roi_data, raw)) is None:
                continue
            row_name = f"{well_fov}_{roi_key}"
            rows[row_name] = spikes

    # convert to DataFrame (handles unequal lengths by filling with NaN)
    df = pd.DataFrame.from_dict(rows, orient="index")

    # give the columns t0, t1, t2, ... name
    df.columns = [f"t{i}" for i in range(df.shape[1])]

    # save to CSV
    suffix = "inferred_spikes_raw_data" if raw else "inferred_spikes_thresholded_data"
    df.to_csv(folder / f"{exp_name}_{suffix}.csv", index=True)


def _export_to_csv_mean_values_grouped_by_condition(
    path: Path | str, data: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Export mean values grouped by condition to CSV."""
    path = Path(path)
    exp_name = path.stem
    folder = path / "grouped"

    for parameter, condition_dict in data.items():
        if parameter in SINGLE_VALUES:
            folder.mkdir(parents=True, exist_ok=True)
            _export_to_csv_single_values(folder, exp_name, parameter, condition_dict)
            continue

        folder.mkdir(parents=True, exist_ok=True)

        output_rows = []

        # get all FOV names
        fov_names: set[str] = set()
        for cond_data in condition_dict.values():
            fov_names.update(cond_data.keys())

        for fov in sorted(fov_names):
            row: dict[str, Any] = {"FOV": fov}
            for cond in sorted(condition_dict):
                fovs = condition_dict[cond]
                values = fovs.get(fov)

                if values is None:
                    row[f"{cond}{MEAN_SUFFIX}"] = ""
                    row[f"{cond}{SEM_SUFFIX}"] = ""
                    row[f"{cond}{N_SUFFIX}"] = ""
                    continue

                if isinstance(values, list) and any(
                    isinstance(el, list) for el in values
                ):
                    flat_values = [v for roi in values for v in roi]
                else:
                    flat_values = values

                if len(flat_values) == 0:
                    row[f"{cond}{MEAN_SUFFIX}"] = ""
                    row[f"{cond}{SEM_SUFFIX}"] = ""
                    row[f"{cond}{N_SUFFIX}"] = ""
                    continue

                mean_val = np.mean(flat_values)
                n_val = len(flat_values)
                sem_val = (
                    np.std(flat_values, ddof=1) / np.sqrt(n_val) if n_val > 1 else 0
                )
                row[f"{cond}{MEAN_SUFFIX}"] = round(mean_val, 5)
                row[f"{cond}{SEM_SUFFIX}"] = round(sem_val, 5)
                row[f"{cond}{N_SUFFIX}"] = n_val

            output_rows.append(row)

        df = pd.DataFrame(output_rows)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}.csv"
        df.to_csv(csv_path, index=False)


def _export_to_csv_percentage_active_n(
    path: Path, exp_name: str, data: dict[str, dict[str, Any]]
) -> None:
    """Export percentage active data with percentage and n columns per condition."""
    combined_columns = {}

    for condition, fovs in sorted(data.items()):
        percentages = []
        sample_sizes = []

        for _, value in fovs.items():
            for item in value:
                if isinstance(item, tuple) and len(item) == 2:
                    percentage, n = item
                    percentages.append(percentage)
                    sample_sizes.append(n)
                elif isinstance(item, (int, float)):
                    # Backward compatibility: if it's just a percentage
                    percentages.append(float(item))
                    sample_sizes.append(1)  # Default n=1

        # Add percentage and n columns for this condition
        combined_columns[f"{condition}_%"] = percentages
        combined_columns[f"{condition}_n"] = sample_sizes

    # Export CSV with alternating percentage and n columns
    if combined_columns:
        padded_rows = zip_longest(*combined_columns.values(), fillvalue=float("nan"))
        df = pd.DataFrame(padded_rows, columns=list(combined_columns.keys()))
        df = df.round(4)

        csv_path = path / f"{exp_name}_{PARAMETER_TO_KEY[PERCENTAGE_ACTIVE]}.csv"
        df.to_csv(csv_path, index=False)


def _export_to_csv_single_values(
    path: Path, exp_name: str, parameter: str, data: dict[str, dict[str, Any]]
) -> None:
    """Export single-value data to CSV."""
    if parameter == PERCENTAGE_ACTIVE:
        _export_to_csv_percentage_active_n(path, exp_name, data)
        return

    if parameter == BURST_ACTIVITY:
        _export_to_csv_burst_activity(path, exp_name, data)
        return

    columns = {}
    max_len = 0
    for condition, fovs in sorted(data.items()):
        values = []
        for _, value in fovs.items():
            values.extend(iter(value))
        columns[condition] = values
        max_len = max(max_len, len(values))

    # create DataFrame
    # pad with NaNs to make all columns equal length (for pandas DataFrame)
    padded_rows = zip_longest(*columns.values(), fillvalue=float("nan"))
    df = pd.DataFrame(padded_rows, columns=list(columns.keys()))
    df = df.round(4)

    # save to CSV
    csv_path = path / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}.csv"
    df.to_csv(csv_path, index=False)


def numeric_intensity(full_key: str, index: int) -> float:
    """
    Return the stimulus intensity.

    ...encoded in …_###.###mW/cm²_…_Mean or just …_###_…_Mean.
    If no number is found, fall back to +inf so those keys end up last.
    """
    parts = full_key.split("_")[index]
    m = NUMBER_RE.search(parts)
    return float(m.group()) if m else float("inf")


def condition_tag(full_key: str) -> str:
    """Return the condition tag from the full key."""
    condition = full_key.split("_")
    return "_".join(condition[:-2])


def _export_to_csv_mean_values_evk_parameters(
    path: Path | str, data: dict[str, dict[str, dict[str, dict[str, list[Any]]]]]
) -> None:
    """Export mean values of evoked parameters to CSV."""
    path = Path(path)
    exp_name = path.stem
    folder = path / "grouped_evk"

    for parameter, condition_dict in data.items():
        if not condition_dict:
            continue

        folder.mkdir(parents=True, exist_ok=True)

        output_rows = []

        # Collect all unique FOV_stim combinations
        fov_stim_keys = set()
        for cond_data in condition_dict.values():
            for fov, stim_dict in cond_data.items():
                for stim in stim_dict:
                    fov_stim_keys.add((fov, stim))

        # Create rows per (FOV, stimulus)
        sorted_keys = sorted(
            fov_stim_keys, key=lambda t: (t[0], numeric_intensity(t[1], 0))
        )
        for fov, stim in sorted_keys:
            row_key = f"{fov}_{stim}"
            row = {"FOV": row_key}

            sorted_cond_keys = sorted(
                condition_dict,
                key=lambda k: (condition_tag(k), numeric_intensity(k, -2)),
            )

            for cond in sorted_cond_keys:
                fovs = condition_dict[cond]
                if stim_values := fovs.get(fov, {}).get(stim):
                    mean_val = np.mean(stim_values)
                    n_val = len(stim_values)
                    sem_val = (
                        np.std(stim_values, ddof=1) / np.sqrt(n_val) if n_val > 1 else 0
                    )
                    row[f"{cond}{MEAN_SUFFIX}"] = round(mean_val, 5)
                    row[f"{cond}{SEM_SUFFIX}"] = round(sem_val, 5)
                    row[f"{cond}{N_SUFFIX}"] = str(n_val)
                else:
                    row[f"{cond}{MEAN_SUFFIX}"] = ""
                    row[f"{cond}{SEM_SUFFIX}"] = ""
                    row[f"{cond}{N_SUFFIX}"] = ""

            output_rows.append(row)

        df = pd.DataFrame(output_rows)
        csv_path = folder / f"{exp_name}_{parameter}.csv"
        df.to_csv(csv_path, index=False)


def _get_percentage_active_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the percentage of active cells with sample sizes."""
    percentage_active_dict: dict[str, dict[str, list[tuple[float, int]]]] = {}
    for condition, well_fov_dict in sorted(data.items()):
        for well_fov, roi_dict in well_fov_dict.items():
            actives = sum(1 if roi_data.active else 0 for roi_data in roi_dict.values())
            total = len(roi_dict)
            percentage_active = actives / total * 100
            # Store as tuple: (percentage, sample_size)
            percentage_active_dict.setdefault(condition, {}).setdefault(
                well_fov, []
            ).append((percentage_active, total))

    return percentage_active_dict


def _get_spike_synchrony_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by spike synchrony."""
    spike_synchrony_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, key_dict in sorted(data.items()):
        for well_fov, roi_dict in key_dict.items():
            spike_dict: dict[str, list[float]] = {}
            for roi_key, roi_data in roi_dict.items():
                if thresholded_spikes := _get_spikes_over_threshold(roi_data):
                    spike_dict[roi_key] = thresholded_spikes

            # Calculate spike synchrony matrix
            spike_synchrony_matrix = _get_spike_synchrony_matrix(spike_dict)

            # Calculate global spike synchrony
            global_spike_synchrony = _get_spike_synchrony(spike_synchrony_matrix)

            spike_synchrony_dict.setdefault(condition, {}).setdefault(
                well_fov, []
            ).append(global_spike_synchrony)

    return spike_synchrony_dict


def _get_calcium_peaks_event_synchrony_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by peak event synchrony."""
    peak_event_synchrony_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, key_dict in sorted(data.items()):
        for well_fov, roi_dict in key_dict.items():
            # Get peak event trains using the existing function
            peak_trains = _get_calcium_peaks_events_from_rois(roi_dict, rois=None)

            if peak_trains is None or len(peak_trains) < 2:
                continue

            # Convert to the format expected by the synchrony matrix function
            peak_event_data_dict = {
                roi_name: cast(list[float], peak_train.astype(float).tolist())
                for roi_name, peak_train in peak_trains.items()
            }

            # Calculate peak event synchrony matrix
            peak_event_synchrony_matrix = _get_calcium_peaks_event_synchrony_matrix(
                peak_event_data_dict
            )

            # Calculate global peak event synchrony
            global_peak_event_synchrony = _get_calcium_peaks_event_synchrony(
                peak_event_synchrony_matrix
            )

            if global_peak_event_synchrony is not None:
                peak_event_synchrony_dict.setdefault(condition, {}).setdefault(
                    well_fov, []
                ).append(global_peak_event_synchrony)

    return peak_event_synchrony_dict


def _get_burst_activity_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by burst activity metrics.

    For each well/condition, calculates 4 burst metrics:
    - Count: Number of bursts detected
    - Avg Duration: Average burst duration in seconds
    - Avg Interval: Average interval between bursts in seconds
    - Rate: Burst rate (bursts per minute)

    Returns
    -------
    dict[str, dict[str, list[Any]]]
        Dictionary with burst metrics organized by condition and well_fov
    """
    burst_activity_dict: dict[str, dict[str, list[Any]]] = {}

    for condition, well_fov_dict in sorted(data.items()):
        for well_fov, roi_dict in well_fov_dict.items():
            # Get burst parameters from ROI data
            burst_params = _get_burst_parameters(roi_dict)
            if burst_params is None:
                continue

            burst_threshold, min_burst_duration, smoothing_sigma = burst_params

            # Get spike trains and time axis for population analysis
            spike_trains, _, time_axis = _get_population_spike_data(roi_dict)

            if spike_trains is None or len(spike_trains) < 2:
                continue

            # Calculate population activity
            population_activity = np.mean(spike_trains, axis=0)

            # Smooth population activity for burst detection
            smoothed_activity = gaussian_filter1d(
                population_activity, sigma=smoothing_sigma
            )

            # Detect bursts
            bursts = _detect_population_bursts(
                smoothed_activity, burst_threshold / 100, min_burst_duration
            )

            # Calculate burst statistics
            burst_count = len(bursts)

            if burst_count == 0:
                # No bursts detected
                burst_metrics = {
                    "Count": 0,
                    "Avg Duration": 0.0,
                    "Avg Interval": 0.0,
                    "Rate": 0.0,
                }
            else:
                # Calculate durations and intervals
                durations = []
                intervals = []

                for i, (start, end) in enumerate(bursts):
                    # Convert indices to time
                    duration_sec = (end - start) * (time_axis[1] - time_axis[0])
                    durations.append(duration_sec)

                    # Calculate interval to next burst
                    if i < len(bursts) - 1:
                        next_start = bursts[i + 1][0]
                        interval_sec = (next_start - end) * (
                            time_axis[1] - time_axis[0]
                        )
                        intervals.append(interval_sec)

                # Calculate statistics
                avg_duration = np.mean(durations) if durations else 0.0
                avg_interval = np.mean(intervals) if intervals else 0.0

                # Calculate rate (bursts per minute)
                total_time_min = (time_axis[-1] - time_axis[0]) / 60.0
                burst_rate = burst_count / total_time_min if total_time_min > 0 else 0.0

                burst_metrics = {
                    "Count": burst_count,
                    "Avg Duration": avg_duration,
                    "Avg Interval": avg_interval,
                    "Rate": burst_rate,
                }

            # Store the metrics for this well
            burst_activity_dict.setdefault(condition, {}).setdefault(
                well_fov, []
            ).append(burst_metrics)

    return burst_activity_dict


def _get_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]], parameter: str
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the specified parameter."""
    parameter_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, well_fov_dict in sorted(data.items()):
        for well_fov, roi_dict in well_fov_dict.items():
            for roi_data in roi_dict.values():
                if not hasattr(roi_data, parameter):
                    raise ValueError(
                        f"The parameter '{parameter}' is not found in the ROI data."
                    )
                value = getattr(roi_data, parameter)
                if value is None:
                    continue
                parameter_dict.setdefault(condition, {}).setdefault(
                    well_fov, []
                ).append(value)

    return parameter_dict


def _get_amplitude_stim_or_non_stim_peaks_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]], stimulated: bool = True
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Group the data by condition → FOV → power_pulse (matching the condition name)."""
    amps_dict: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for condition, fov_dict in sorted(data.items()):
        if stimulated and f"{EVK_STIM}_" in condition:
            target_power_pulse = condition.split(f"{EVK_STIM}_")[-1]
        elif not stimulated and f"{EVK_NON_STIM}_" in condition:
            target_power_pulse = condition.split(f"{EVK_NON_STIM}_")[-1]
        else:
            continue  # skip unrelated conditions
        for fov, roi_dict in fov_dict.items():
            for roi_data in roi_dict.values():
                from ._util import get_stimulated_amplitudes_from_roi_data

                # Compute amplitudes on-demand
                amps_stim, amps_non_stim = get_stimulated_amplitudes_from_roi_data(
                    roi_data, led_power_equation=None
                )

                amps = amps_stim if stimulated else amps_non_stim
                if not amps:
                    continue
                if target_power_pulse not in amps:
                    continue
                values = amps[target_power_pulse]
                amps_dict.setdefault(condition, {}).setdefault(fov, {}).setdefault(
                    target_power_pulse, []
                ).extend(values)
    return amps_dict


def _export_to_csv_burst_activity(
    path: Path, exp_name: str, data: dict[str, dict[str, Any]]
) -> None:
    """Export burst activity data to CSV with 4 columns per condition."""
    combined_columns = {}

    for condition, fovs in sorted(data.items()):
        counts = []
        durations = []
        intervals = []
        rates = []

        for _, well_data in fovs.items():
            for metrics in well_data:
                if isinstance(metrics, dict):
                    counts.append(metrics.get("Count", 0))
                    durations.append(metrics.get("Avg Duration", 0.0))
                    intervals.append(metrics.get("Avg Interval", 0.0))
                    rates.append(metrics.get("Rate", 0.0))

        # Add 4 columns for each condition
        combined_columns[f"{condition}_Count"] = counts
        combined_columns[f"{condition}_Avg_Duration"] = durations
        combined_columns[f"{condition}_Avg_Interval"] = intervals
        combined_columns[f"{condition}_Rate"] = rates

    # Export CSV with 4 columns per condition
    if combined_columns:
        padded_rows = zip_longest(*combined_columns.values(), fillvalue=float("nan"))
        df = pd.DataFrame(padded_rows, columns=list(combined_columns.keys()))
        df = df.round(4)

        csv_path = path / f"{exp_name}_{PARAMETER_TO_KEY[BURST_ACTIVITY]}.csv"
        df.to_csv(csv_path, index=False)
