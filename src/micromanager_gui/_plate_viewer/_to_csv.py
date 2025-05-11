from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._util import ROIData, _get_synchrony, _get_synchrony_matrix

PERCENTAGE_ACTIVE = "percentage_active"
SYNCHRONY = "synchrony"
AMP_STIMULATED_PEAKS = "amplitudes_stimulated_peaks"
AMP_SPONTANEOUS_PEAKS = "amplitudes_spontaneous_peaks"
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
    "amplitudes_spontaneous_peaks": AMP_SPONTANEOUS_PEAKS,
    # TODO: add the other parameters. Means we need to update the methods (e.g.
    # _get_percentage_active_parameter, _get_synchrony_parameter, etc.) to handle the
    # evoked data.
    # **CSV_PARAMETERS,
}

PARAMETER_TO_KEY: dict[str, str] = {
    **{v: k for k, v in CSV_PARAMETERS.items()},
    **{v: k for k, v in CSV_PARAMETERS_EVK.items()},
}


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
    _export_raw_data(path, analysis_data)
    _export_to_csv_grouped_by_conditions_per_fovs(path, fov_by_condition_by_parameter)
    _export_to_csv_grouped_by_conditions(path, fov_by_condition_by_parameter)
    _export_to_csv_grouped_by_conditions_per_fovs_evk(path, fov_by_condition_by_parameter_evk)  # noqa: E501
    _export_to_csv_grouped_by_conditions_evk(path, fov_by_condition_by_parameter_evk)
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
            if roi_data.stimulated and roi_data.amplitudes_stimulated_peaks:
                for power_and_pulse in roi_data.amplitudes_stimulated_peaks:
                    key = f"{cond_key}_evk_stim_{power_and_pulse}"
                    evoked_conds.setdefault(key, {}).setdefault(well_fov, {})[
                        roi_key
                    ] = roi_data
            else:
                key = f"{cond_key}_evk_spont"
                evoked_conds.setdefault(key, {}).setdefault(well_fov, {})[roi_key] = (
                    roi_data
                )

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


def _export_to_csv_grouped_by_conditions_per_fovs(
    path: Path | str, data: dict[str, dict[str, dict[str, list]]]
) -> None:
    """Save each parameter in `data` as a separate CSV with columns as condition_key."""
    path = Path(path)
    exp_name = path.stem
    folder = path / "csv_by_conditions_and_fovs"
    folder.mkdir(parents=True, exist_ok=True)

    for parameter, condition_dict in data.items():
        series_dict = {}
        if condition_dict is None:
            continue
        for condition, keys in condition_dict.items():
            for key, values in keys.items():
                col_name = f"{condition}_{key}"
                flat_values = _flatten_if_list_of_lists(values)
                series_dict[col_name] = pd.Series(flat_values)

        df = pd.DataFrame(series_dict)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}_per_fov.csv"
        df.to_csv(csv_path, index=False)


def _export_to_csv_grouped_by_conditions(
    path: Path | str, data: dict[str, dict[str, dict[str, list]]]
) -> None:
    """Save each parameter as a separate CSV, grouping all values by condition.

    Each column corresponds to a condition and contains all values (from all keys)
    stacked together.
    """
    path = Path(path)
    exp_name = path.stem
    folder = path / "csv_by_conditions"
    folder.mkdir(parents=True, exist_ok=True)

    for parameter, condition_dict in data.items():
        series_dict = {}
        if condition_dict is None:
            continue
        for condition, keys in condition_dict.items():
            all_values = []
            for values in keys.values():
                flat_values = _flatten_if_list_of_lists(values)
                all_values.extend(flat_values)
            series_dict[condition] = pd.Series(all_values)

        df = pd.DataFrame(series_dict)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}.csv"
        df.to_csv(csv_path, index=False)


def _export_to_csv_grouped_by_conditions_per_fovs_evk(
    path: Path | str, data: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Save each parameter in `data` as a separate CSV with columns as condition_key."""
    path = Path(path)
    exp_name = path.stem
    folder = path / "csv_by_conditions_and_fovs"
    folder.mkdir(parents=True, exist_ok=True)

    for parameter, condition_dict in data.items():
        series_dict = {}
        if condition_dict is None:
            continue
        for condition, keys in condition_dict.items():
            for key, values in keys.items():
                col_name = f"{condition}_{key}"
                if isinstance(values, dict):  #  dict (stimulated)
                    for roi_key, values_1 in values.items():
                        if roi_key in col_name:
                            flat_values = _flatten_if_list_of_lists(values_1)
                else:  # list (spontaneous)
                    flat_values = _flatten_if_list_of_lists(values)
                series_dict[col_name] = pd.Series(flat_values)

        df = pd.DataFrame(series_dict)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}_per_fov.csv"
        df.to_csv(csv_path, index=False)


def _export_to_csv_grouped_by_conditions_evk(
    path: Path | str, data: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Save each parameter as a separate CSV, grouping all values by condition.

    Each column corresponds to a condition and contains all values (from all keys)
    stacked together.
    """
    path = Path(path)
    exp_name = path.stem
    folder = path / "csv_by_conditions"
    folder.mkdir(parents=True, exist_ok=True)

    for parameter, condition_dict in data.items():
        series_dict = {}
        if condition_dict is None:
            continue
        for condition, keys in condition_dict.items():
            all_values = []
            for _, values in keys.items():
                if isinstance(values, dict):  #  dict (stimulated)
                    for _, values_1 in values.items():
                        flat_values = _flatten_if_list_of_lists(values_1)
                else:  # list (spontaneous)
                    flat_values = _flatten_if_list_of_lists(values)
                all_values.extend(flat_values)
            series_dict[condition] = pd.Series(all_values)

        df = pd.DataFrame(series_dict)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}.csv"
        df.to_csv(csv_path, index=False)


def _flatten_if_list_of_lists(values: list[Any]) -> list[Any]:
    """Flatten a list of lists if necessary."""
    if values and all(isinstance(v, list) for v in values):
        return [item for sublist in values for item in sublist]
    return values


def _rearrange_by_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
    parameter: str,
    evk: bool = False,
) -> dict[str, dict[str, list[Any]]]:
    """Create a dict grouped by the specified parameter per condition."""
    if parameter == PERCENTAGE_ACTIVE:
        return _get_percentage_active_parameter(data)
    if parameter == SYNCHRONY:
        return _get_synchrony_parameter(data)
    return _get_parameter(data, parameter)


def _rearrange_by_parameter_evk(
    data: dict[str, dict[str, dict[str, ROIData]]],
    parameter: str,
) -> dict[str, dict[str, dict[str, list[Any]]]] | dict[str, dict[str, list[Any]]]:
    """Create a dict grouped by the specified parameter per condition."""
    if parameter == AMP_STIMULATED_PEAKS:
        return _get_amplitude_stimulated_peaks_parameter(data)
    if parameter == AMP_SPONTANEOUS_PEAKS:
        return _get_amplitude_spontaneous_peaks_parameter(data, parameter)
    else:
        raise ValueError(f"The parameter '{parameter}' is not found in the ROI data.")


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
            synchrony_dict.setdefault(condition, {}).setdefault(well_fov, []).append(
                linear_synchrony
            )
    return synchrony_dict


def _get_amplitude_stimulated_peaks_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]],
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Group the data by the amplitude of stimulated peaks."""
    evk_dict: dict[str, dict[str, dict[str, list[Any]]]] = {}
    for condition, well_fov_dict in data.items():
        for well_fov, roi_dict in well_fov_dict.items():
            for _, roi_data in roi_dict.items():
                if roi_data.amplitudes_stimulated_peaks is not None:
                    # power_pulse is f"{power}_{pulse_len}"
                    for (
                        power_pulse,
                        amp_list,
                    ) in roi_data.amplitudes_stimulated_peaks.items():
                        for amp_val in amp_list:
                            evk_dict.setdefault(condition, {}).setdefault(
                                well_fov, {}
                            ).setdefault(power_pulse, []).append(amp_val)
    return evk_dict


def _get_amplitude_spontaneous_peaks_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]], parameter: str
) -> dict[str, dict[str, list[Any]]]:
    """Group the data by the specified parameter."""
    parameter_dict: dict[str, dict[str, list[Any]]] = {}
    for condition, well_fov_dict in data.items():
        # Skip evoked stimulated conditions
        if "evk_stim" in condition:
            continue
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
