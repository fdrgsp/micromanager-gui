from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._logger import LOGGER
from ._util import (
    ROIData,
    _get_synchrony,
    _get_synchrony_matrix,
)

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
    "amplitudes_non_stimulated_peaks": AMP_NON_STIMULATED_PEAKS,
}

PARAMETER_TO_KEY: dict[str, str] = {
    **{v: k for k, v in CSV_PARAMETERS.items()},
    **{v: k for k, v in CSV_PARAMETERS_EVK.items()},
}
NESTED = ["peaks_amplitudes_dec_dff", "dec_dff_frequency", "iei", "cell_size"]
SINGLE_VALUE = ["percentage_active"]
TUPLES = ["synchrony"]


def _save_to_csv(
    path: str | Path,
    analysis_data: dict[str, dict[str, ROIData]] | None,
) -> None:
    """Save the analysis data as CSV files."""
    if analysis_data is None:
        return
    if isinstance(path, str):
        path = Path(path)

    # Rearrange the data by condition
    fov_by_condition, evk_conditions = _rearrange_fov_by_conditions(analysis_data)

    # Rearrange fov_by_condition by parameter
    fov_by_condition_by_parameter = {
        parameter: _rearrange_by_parameter(fov_by_condition, parameter)
        for parameter in CSV_PARAMETERS.values()
    }

    fov_by_condition_by_parameter_evk = {
        parameter: _rearrange_by_parameter_evk(evk_conditions, parameter)
        for parameter in CSV_PARAMETERS_EVK.values()
    }

    # fmt: off
    # Save the data as CSV files
    try:
        _export_raw_data(path, analysis_data)
        _export_dff_data(path, analysis_data)
        _export_dec_dff_data(path, analysis_data)
        _export_to_csv_mean_values_grouped_by_condition(path, fov_by_condition_by_parameter)  # noqa E501
        _export_to_csv_mean_values_evk_parameters(path, fov_by_condition_by_parameter_evk)  # noqa E501
    except Exception as e:
        LOGGER.error(f"Error exporting data to CSV: {e}")
        return
    LOGGER.info(f"Successfully exported data as CSV files to `{path}`.")
    # fmt: on


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
                stim_label = "evk_stim"
            else:
                amps_dict = roi_data.amplitudes_non_stimulated_peaks
                stim_label = "evk_non_stim"
            if amps_dict:
                for power_and_pulse in amps_dict:
                    key = f"{cond_key}_{stim_label}_{power_and_pulse}"
                    evoked_conds.setdefault(key, {}).setdefault(well_fov, {})[
                        roi_key
                    ] = roi_data

    return conds, evoked_conds


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


def _export_to_csv_mean_values_grouped_by_condition(
    path: Path | str, data: dict[str, dict[str, dict[str, Any]]]
) -> None:
    path = Path(path)
    exp_name = path.stem
    folder = path / "grouped"
    folder.mkdir(parents=True, exist_ok=True)

    for parameter, condition_dict in data.items():
        if parameter not in PARAMETER_TO_KEY:
            continue  # Skip unknown parameters

        output_rows = []

        # get all FOV names
        fov_names: set[str] = set()
        for cond_data in condition_dict.values():
            fov_names.update(cond_data.keys())

        for fov in sorted(fov_names):
            row: dict[str, Any] = {"FOV": fov}
            for cond, fovs in condition_dict.items():
                values = fovs.get(fov)

                if values is None:
                    row[f"{cond}_Mean"] = ""
                    row[f"{cond}_SEM"] = ""
                    row[f"{cond}_N"] = ""
                    continue

                # handle nested structure for multi-ROI data
                if parameter in NESTED:
                    if isinstance(values, list) and any(
                        isinstance(el, list) for el in values
                    ):
                        flat_values = [v for roi in values for v in roi]
                    else:
                        flat_values = values
                    mean_val = np.mean(flat_values)
                    n_val = len(flat_values)
                    sem_val = (
                        np.std(flat_values, ddof=1) / np.sqrt(n_val) if n_val > 1 else 0
                    )
                    row[f"{cond}_Mean"] = round(mean_val, 5)
                    row[f"{cond}_SEM"] = round(sem_val, 5)
                    row[f"{cond}_N"] = n_val

                # handle single-value lists like percentage_active or synchrony
                elif parameter in SINGLE_VALUE:
                    val = values[0] if isinstance(values, list) and values else ""
                    row[f"{cond}_Value"] = (
                        round(val, 5) if isinstance(val, (int, float)) else ""
                    )
                elif parameter in TUPLES:
                    # tuples are sync_value, p_value, z_score
                    val = values[0]
                    row[f"{cond}_Value"] = (
                        round(val, 5) if isinstance(val, (int, float)) else ""
                    )
                else:
                    # catch-all fallback
                    row[f"{cond}_Data"] = str(values)

            output_rows.append(row)

        df = pd.DataFrame(output_rows)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}.csv"
        df.to_csv(csv_path, index=False)


def _export_to_csv_mean_values_evk_parameters(
    path: Path | str, data: dict[str, dict[str, dict[str, dict[str, list[Any]]]]]
) -> None:
    path = Path(path)
    exp_name = path.stem
    folder = path / "grouped_evk"

    for parameter, condition_dict in data.items():
        if parameter not in CSV_PARAMETERS_EVK.keys():
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
        for fov, stim in sorted(fov_stim_keys):
            row_key = f"{fov}_{stim}"
            row = {"FOV": row_key}

            for cond, fovs in condition_dict.items():
                if stim_values := fovs.get(fov, {}).get(stim):
                    mean_val = np.mean(stim_values)
                    n_val = len(stim_values)
                    sem_val = (
                        np.std(stim_values, ddof=1) / np.sqrt(n_val) if n_val > 1 else 0
                    )
                    row[f"{cond}_Mean"] = round(mean_val, 5)
                    row[f"{cond}_SEM"] = round(sem_val, 5)
                    row[f"{cond}_N"] = str(n_val)
                else:
                    row[f"{cond}_Mean"] = ""
                    row[f"{cond}_SEM"] = ""
                    row[f"{cond}_N"] = ""

            output_rows.append(row)

        df = pd.DataFrame(output_rows)
        csv_path = folder / f"{exp_name}_{parameter}.csv"
        df.to_csv(csv_path, index=False)


def _rearrange_by_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
    parameter: str,
) -> dict[str, dict[str, list[Any]]]:
    """Create a dict grouped by the specified parameter per condition."""
    if parameter == PERCENTAGE_ACTIVE:
        try:
            return _get_percentage_active_parameter(data)
        except Exception as e:
            print(f"Error calculating percentage active: {e}")
            return {}
    if parameter == SYNCHRONY:
        try:
            return _get_synchrony_parameter(data)
        except Exception as e:
            print(f"Error calculating synchrony: {e}")
            return {}
    try:
        return _get_parameter(data, parameter)
    except Exception as e:
        print(f"Error calculating parameter '{parameter}': {e}")
        return {}


def _rearrange_by_parameter_evk(
    data: dict[str, dict[str, dict[str, ROIData]]],
    parameter: str,
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Create a dict grouped by the specified parameter per condition."""
    if parameter == AMP_STIMULATED_PEAKS:
        try:
            return _get_amplitude_stim_or_non_stim_peaks_parameter(data)
        except Exception as e:
            print(f"Error calculating stimulated peaks: {e}")
            return {}
    if parameter == AMP_NON_STIMULATED_PEAKS:
        try:
            return _get_amplitude_stim_or_non_stim_peaks_parameter(
                data, stimulated=False
            )
        except Exception as e:
            print(f"Error calculating non-stimulated peaks: {e}")
            return {}
    return {}


def _get_percentage_active_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the percentage of active cells."""
    percentage_active_dict: dict[str, dict[str, list[float]]] = {}
    for condition, well_fov_dict in data.items():
        for well_fov, roi_dict in well_fov_dict.items():
            actives = 0
            for roi_data in roi_dict.values():
                value = 1 if roi_data.active else 0
                actives += value
            percentage_active = actives / len(roi_dict) * 100
            percentage_active_dict.setdefault(condition, {}).setdefault(
                well_fov, []
            ).append(percentage_active)

    return percentage_active_dict


def _get_synchrony_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the synchrony."""
    synchrony_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, key_dict in data.items():
        for well_fov, roi_dict in key_dict.items():
            instantaneous_phase_dict: dict[str, list[float]] = {
                roi_key: roi_data.instantaneous_phase
                for roi_key, roi_data in roi_dict.items()
                if roi_data.instantaneous_phase is not None
            }
            synchrony_matrix = _get_synchrony_matrix(instantaneous_phase_dict)

            linear_synchrony = _get_synchrony(synchrony_matrix)

            # null_scores = compute_null_distribution(
            #     instantaneous_phase_dict, num_shuffles=1000
            # )
            # z_score = compute_z_score(linear_synchrony, null_scores)
            # “Empirical p-values were computed using 1000 surrogate shuffles of the
            # data, comparing the observed synchrony score to the distribution of
            # scores from randomly permuted phase traces (using a +1 correction).”
            # Standard practice in permutation testing (called pseudocount correction):
            # p_value = (
            #     np.sum(null_scores >= linear_synchrony) + 1) / (len(null_scores) + 1
            # )

            synchrony_dict.setdefault(condition, {}).setdefault(well_fov, []).append(
                (linear_synchrony, 0.0, 0.0)
            )
    return synchrony_dict


def _get_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]], parameter: str
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the specified parameter."""
    parameter_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, well_fov_dict in data.items():
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
    for condition, fov_dict in data.items():
        if stimulated and "evk_stim_" in condition:
            target_power_pulse = condition.split("evk_stim_")[-1]
        elif not stimulated and "evk_non_stim_" in condition:
            target_power_pulse = condition.split("evk_non_stim_")[-1]
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
