from __future__ import annotations

import re
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._logger import LOGGER
from ._util import (
    EVK_NON_STIM,
    EVK_STIM,
    MEAN_SUFFIX,
    N_SUFFIX,
    SEM_SUFFIX,
    ROIData,
    _get_synchrony_matrix,
    get_linear_phase,
    get_synchrony,
)

# fmt: off
NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
PERCENTAGE_ACTIVE = "percentage_active"
SYNCHRONY = "synchrony"
AMP_STIMULATED_PEAKS = "amplitudes_stimulated_peaks"
AMP_NON_STIMULATED_PEAKS = "amplitudes_non_stimulated_peaks"
CSV_PARAMETERS: dict[str, str] = {
    "amplitude": "peaks_amplitudes_dec_dff",
    "frequency": "dec_dff_frequency",
    "cell_size": "cell_size",
    "iei": "iei",
    "percentage_active": PERCENTAGE_ACTIVE,
    "synchrony": SYNCHRONY,
}
CSV_PARAMETERS_EVK = {
    "amplitudes_stimulated_peaks": AMP_STIMULATED_PEAKS,
    "amplitudes_non_stimulated_peaks": AMP_NON_STIMULATED_PEAKS
}

PARAMETER_TO_KEY: dict[str, str] = {
    **{v: k for k, v in CSV_PARAMETERS.items()},
    **{v: k for k, v in CSV_PARAMETERS_EVK.items()},
}

SINGLE_VALUES = [PERCENTAGE_ACTIVE, SYNCHRONY]
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
        LOGGER.error( f"Error exporting dFF DATA to CSV: {e}")
    try:
        _export_dec_dff_data(path, analysis_data)
    except Exception as e:
        LOGGER.error(f"Error exporting DEC_DFF DATA to CSV: {e}")
    try:
        _export_inferred_spikes_data(path, analysis_data)
    except Exception as e:
        LOGGER.error(f"Error exporting INFERRED SPIKES DATA to CSV: {e}")
    LOGGER.info("Exporting data to CSV: DONE!")


def save_analysys_data_to_csv(
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
        LOGGER.error(f"Error exporting spontanoous analysys data to CSV: {e}")
    try:
        _export_to_csv_mean_values_evk_parameters(path, rearrange_cond_evk)
    except Exception as e:
        LOGGER.error(f"Error exporting evoked analysys data to CSV: {e}")
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
            if roi_data.stimulated:
                amps_dict = roi_data.amplitudes_stimulated_peaks
                stim_label = EVK_STIM
            else:
                amps_dict = roi_data.amplitudes_non_stimulated_peaks
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
    if parameter == SYNCHRONY:
        try:
            return _get_synchrony_parameter(data)
        except Exception as e:
            LOGGER.error(f"Error calculating synchrony: {e}")
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
    path: Path | str, data: dict[str, dict[str, ROIData]]
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
            if roi_data.inferred_spikes is None:
                continue
            row_name = f"{well_fov}_{roi_key}"
            rows[row_name] = roi_data.inferred_spikes

    # convert to DataFrame (handles unequal lengths by filling with NaN)
    df = pd.DataFrame.from_dict(rows, orient="index")

    # give the columns t0, t1, t2, ... name
    df.columns = [f"t{i}" for i in range(df.shape[1])]

    # save to CSV
    df.to_csv(folder / f"{exp_name}_inferred_spikes_data.csv", index=True)


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


def _get_synchrony_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the synchrony."""
    synchrony_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, key_dict in sorted(data.items()):
        for well_fov, roi_dict in key_dict.items():
            phase_dict: dict[str, list[float]] = {}
            for roi_key, roi_data in roi_dict.items():
                if (
                    not roi_data.dec_dff
                    or not roi_data.peaks_dec_dff
                    or len(roi_data.peaks_dec_dff) < 1
                ):
                    continue
                frames = len(roi_data.dec_dff)
                peaks = np.array(roi_data.peaks_dec_dff)
                phase_dict[roi_key] = get_linear_phase(frames, peaks)

            synchrony_matrix = _get_synchrony_matrix(phase_dict)

            linear_synchrony = get_synchrony(synchrony_matrix)

            synchrony_dict.setdefault(condition, {}).setdefault(well_fov, []).append(
                linear_synchrony
            )
    return synchrony_dict


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
                amps = (
                    roi_data.amplitudes_stimulated_peaks
                    if stimulated
                    else roi_data.amplitudes_non_stimulated_peaks
                )
                if not amps:
                    continue
                if target_power_pulse not in amps:
                    continue
                values = amps[target_power_pulse]
                amps_dict.setdefault(condition, {}).setdefault(fov, {}).setdefault(
                    target_power_pulse, []
                ).extend(values)
    return amps_dict
