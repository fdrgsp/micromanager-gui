from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._util import ROIData, _get_synchrony, _get_synchrony_matrix

PERCENTAGE_ACTIVE = "percentage_active"
SYNCHRONY = "synchrony"
CSV_PARAMETERS: dict[str, str] = {
    "amplitude": "peaks_amplitudes_dec_dff",
    "frequency": "dec_dff_frequency",
    "cell_size": "cell_size",
    "iei": "iei",
    "percentage_active": PERCENTAGE_ACTIVE,
    "synchrony": SYNCHRONY,
}
PARAMETER_TO_KEY: dict[str, str] = {v: k for k, v in CSV_PARAMETERS.items()}


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
    data_by_condition = _rearrange_by_condition(analysis_data)
    # Rearrange the data by condition and parameter
    data_by_condition_by_parameter = {
        parameter: _rearrange_by_condition_by_parameter(data_by_condition, parameter)
        for parameter in CSV_PARAMETERS.values()
    }
    # Save the data as CSV files
    _export_to_csv_by_conditions_and_fovs(path, data_by_condition_by_parameter)
    _export_to_csv_grouped_by_conditions(path, data_by_condition_by_parameter)


def _export_to_csv_by_conditions_and_fovs(
    path: Path | str, data: dict[str, dict[str, dict[str, list]]]
) -> None:
    """Save each parameter in `data` as a separate CSV with columns as condition_key."""
    path = Path(path)
    exp_name = path.stem
    folder = path / "csv_by_conditions_and_fovs"
    folder.mkdir(parents=True, exist_ok=True)

    for parameter, condition_dict in data.items():
        series_dict = {}
        for condition, keys in condition_dict.items():
            for key, values in keys.items():
                col_name = f"{condition}_{key}"
                flat_values = _flatten_if_list_of_lists(values)
                series_dict[col_name] = pd.Series(flat_values)

        df = pd.DataFrame(series_dict)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}_cf.csv"
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
        for condition, keys in condition_dict.items():
            all_values = []
            for values in keys.values():
                flat_values = _flatten_if_list_of_lists(values)
                all_values.extend(flat_values)
            series_dict[condition] = pd.Series(all_values)

        df = pd.DataFrame(series_dict)
        csv_path = folder / f"{exp_name}_{PARAMETER_TO_KEY[parameter]}_c.csv"
        df.to_csv(csv_path, index=False)


def _flatten_if_list_of_lists(values: list[Any]) -> list[Any]:
    """Flatten a list of lists if necessary."""
    if values and all(isinstance(v, list) for v in values):
        return [item for sublist in values for item in sublist]
    return values


def _rearrange_by_condition(
    data: dict[str, dict[str, ROIData]],
) -> dict[str, dict[str, dict[str, ROIData]]]:
    """Rearrange the data by condition.

    Parameters
    ----------
    data: dict[str, dict[str, dict[str, ROIData]]
        The data to rearrange.
    """
    conds: dict[str, dict[str, dict[str, ROIData]]] = {}
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
            if roi_data.stimulated:
                cond_key += "_evk"
            conds.setdefault(cond_key, {}).setdefault(well_fov, {})[roi_key] = roi_data
    return conds


def _rearrange_by_condition_by_parameter(
    data: dict[str, dict[str, dict[str, ROIData]]], parameter: str
) -> dict[str, dict[str, list[Any]]]:
    """Create a dict grouped by the specified parameter per condition."""
    if parameter == PERCENTAGE_ACTIVE:
        return _get_percentage_active_parameter(data)
    if parameter == SYNCHRONY:
        return _get_synchrony_parameter(data)
    else:
        return _get_parameter(data, parameter)


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
