from __future__ import annotations

from pathlib import Path

import numpy as np
import xlsxwriter

from ._util import COND1, COND2, ROIData, _get_synchrony, _get_synchrony_matrix

# ---------------------------Measurement to save data as CSV---------------------------
# Each metric here will be pulled/calculated from the analysis data and compile into
# a CSV at the end of the analysis.
COMPILE_METRICS = [
    "amplitude",
    "frequency",
    "cell_size",
    "synchrony",
    "iei",
    "percentage_active",
]


def data_to_csv(
    analysis_data: dict[str, dict[str, ROIData]],
    plate_map: dict[str, dict[str, str]],
    stimulated_exp: bool,
    save_path: str,
) -> None:
    """Save the analysis data as CSV files."""
    if analysis_data is None:
        return

    cond_ordered = _organize_fov_by_condition(analysis_data, plate_map, stimulated_exp)

    fov_data_by_metric, cell_size_unit = _compile_per_metric(
        analysis_data, plate_map, stimulated_exp
    )

    _save_as_csv(
        fov_data_by_metric,
        cell_size_unit,
        cond_ordered,
        save_path,
    )


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
    for well_fov, rois in data.items():
        for roi_key, roi_data in rois.items():
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
    return conds


def _organize_fov_by_condition(
    analysis_data: dict[str, dict[str, ROIData]],
    plate_map: dict[str, dict[str, str]],
    stimulated_exp: bool,
) -> list[str]:
    """Ordered the conditions."""
    fovs = list(analysis_data.keys())
    cond_1_list: list = []
    cond_2_list: list = []
    conds_ordered: list[str] = []

    for fov in fovs:
        well = fov.split("_")[0]

        if well not in plate_map:
            cond_1, cond_2 = "unspecified", "unspecified"

        else:
            cond_1 = plate_map[well].get(COND1, "unspecified")
            cond_2 = plate_map[well].get(COND2, "unspecified")

        if cond_1 not in cond_1_list:
            cond_1_list.append(cond_1)

        if cond_2 not in cond_2_list:
            cond_2_list.append(cond_2)

    for cond_1 in cond_1_list:
        for cond_2 in cond_2_list:
            conds = f"{cond_1}{"_"}{cond_2}"
            conds_ordered.append(conds)
            if stimulated_exp:
                conds_ordered.append(f"{conds}_evk")

    return conds_ordered


def _compile_data_per_fov(
    fov_dict: dict[str, ROIData], stimulated_exp: bool
) -> tuple[dict[str, list[float]], dict[str, list[float]], str]:
    """Compile FOV data from all ROI data."""
    # compile data for one fov
    data_dict: dict[str, list[float]] = {}
    sti_data_dict: dict[str, list[float]] = {}

    for measurement in COMPILE_METRICS:
        if measurement not in data_dict:
            data_dict[measurement] = []
        if stimulated_exp and measurement not in sti_data_dict:
            sti_data_dict[measurement] = []

    amplitude_list_fov: list[float] = []
    cell_size_list_fov: list[float] = []
    frequency_list_fov: list[float] = []
    iei_list_fov: list[float] = []
    active_cells: int = 0
    cell_size_unit: str = ""
    instantaneous_phase_dict: dict[str, list[float]] = {}

    if stimulated_exp:
        sti_amplitude_list_fov: list[float] = []
        sti_cell_size_list_fov: list[float] = []
        sti_frequency_list_fov: list[float] = []
        sti_iei_list_fov: list[float] = []
        sti_active_cells: int = 0

    for roi_name, roiData in fov_dict.items():
        if not isinstance(roiData, ROIData):
            continue

        # cell size
        if roiData.cell_size:
            if stimulated_exp and roiData.stimulated:
                sti_cell_size_list_fov.append(roiData.cell_size)
            else:
                cell_size_list_fov.append(roiData.cell_size)

        if len(cell_size_unit) < 1:
            cell_size_unit = (
                roiData.cell_size_units
                if isinstance(roiData.cell_size_units, str)
                else ""
            )

        if (
            isinstance(roiData.instantaneous_phase, list)
            and len(roiData.instantaneous_phase) > 0
        ):
            if roi_name not in instantaneous_phase_dict:
                instantaneous_phase_dict[roi_name] = []
            instantaneous_phase_dict[roi_name] = roiData.instantaneous_phase

        # if cells are active (i.e. have at least one peak)
        if roiData.active:
            # amplitude
            if stimulated_exp and roiData.stimulated:
                (
                    sti_amplitude_list_fov.extend(roiData.peaks_amplitudes_dec_dff)
                    if roiData.peaks_amplitudes_dec_dff
                    else []
                )
            else:
                (
                    amplitude_list_fov.extend(roiData.peaks_amplitudes_dec_dff)
                    if roiData.peaks_amplitudes_dec_dff
                    else []
                )

            # frequency
            if roiData.dec_dff_frequency:
                if stimulated_exp and roiData.stimulated:
                    sti_frequency_list_fov.append(roiData.dec_dff_frequency)
                else:
                    frequency_list_fov.append(roiData.dec_dff_frequency)

            # iei
            if stimulated_exp and roiData.stimulated:
                sti_iei_list_fov.extend(roiData.iei) if roiData.iei else []
            else:
                iei_list_fov.extend(roiData.iei) if roiData.iei else []

            # activity
            if stimulated_exp and roiData.stimulated:
                sti_active_cells += 1
            else:
                active_cells += 1

    synchrony_matrix = _get_synchrony_matrix(instantaneous_phase_dict)
    linear_synchrony = _get_synchrony(synchrony_matrix)

    if stimulated_exp:
        sti_percentage_active = (
            float(sti_active_cells / len(sti_cell_size_list_fov) * 100)
            if len(sti_cell_size_list_fov) > 0
            else np.nan
        )
    percentage_active = float(active_cells / len(cell_size_list_fov) * 100)

    # NOTE: if adding more output measurements,
    # make sure to check that the keys are in the COMPILED_METRICS
    data_dict["amplitude"] = amplitude_list_fov
    data_dict["frequency"] = frequency_list_fov
    data_dict["cell_size"] = cell_size_list_fov
    data_dict["iei"] = iei_list_fov
    data_dict["percentage_active"] = [percentage_active]
    data_dict["synchrony"] = [
        linear_synchrony if isinstance(linear_synchrony, float) else np.nan
    ]
    if stimulated_exp:
        sti_data_dict["amplitude"] = sti_amplitude_list_fov
        sti_data_dict["frequency"] = sti_frequency_list_fov
        sti_data_dict["cell_size"] = sti_cell_size_list_fov
        sti_data_dict["iei"] = sti_iei_list_fov
        sti_data_dict["percentage_active"] = [sti_percentage_active]

    return data_dict, sti_data_dict, cell_size_unit


def _compile_per_metric(
    analysis_data: dict[str, dict[str, ROIData]],
    plate_map: dict[str, dict[str, str]],
    stimulated_exp: bool,
) -> tuple[list[dict[str, list[float]]], str]:
    """Group the FOV data based on metrics and platemap."""
    data_by_metrics: list[dict[str, list[float]]] = []

    for output in COMPILE_METRICS:  # noqa: B007
        data_by_metrics.append({})

    for fov_name, fov_dict in analysis_data.items():
        well = fov_name.split("_")[0]

        if well not in plate_map:
            cond_1, cond_2 = "unspecified", "unspecified"
        else:
            cond_1 = plate_map[well].get(COND1, "unspecified")
            cond_2 = plate_map[well].get(COND2, "unspecified")

        conds = f"{cond_1}{"_"}{cond_2}"

        if stimulated_exp:
            conds_evk = f"{conds}_evk"

        for output_dict in data_by_metrics:
            if conds not in output_dict:
                output_dict[conds] = []
            if stimulated_exp and conds_evk not in output_dict:
                output_dict[conds_evk] = []

        data_per_fov_dict, sti_data_dict, cell_size_unit = _compile_data_per_fov(
            fov_dict, stimulated_exp
        )

        for i, output in enumerate(COMPILE_METRICS):
            output_value = data_per_fov_dict[output]
            data_by_metrics[i][conds].extend(output_value)
            if stimulated_exp:
                sti_output_value = sti_data_dict[output]
                data_by_metrics[i][conds_evk].extend(sti_output_value)

    return data_by_metrics, cell_size_unit


def _save_as_csv(
    compiled_data_list: list[dict[str, list[float]]],
    cell_size_unit: str,
    cond_ordered: list[str],
    save_path: str,
) -> None:
    """Save csv files of the data."""
    exp_name = Path(save_path).stem

    if compiled_data_list is None:
        return None

    condition_list: list[str] = []

    for metric, readout_data in zip(COMPILE_METRICS, compiled_data_list):
        if len(condition_list) < 1:
            condition_list = list(readout_data.keys())

        if metric == "cell_size":
            file_path = Path(save_path) / f"{exp_name}_{metric}_{cell_size_unit}.xlsx"
        else:
            file_path = Path(save_path) / f"{exp_name}_{metric}.xlsx"

        with xlsxwriter.Workbook(file_path, {"nan_inf_to_errors": True}) as wkbk:
            wkst = wkbk.add_worksheet(metric)
            num_format = wkbk.add_format({"num_format": "0.00"})
            column = 0

            for condition in cond_ordered:
                wkst.write(0, column, condition)
                single_cell_data = readout_data.get(condition)
                if isinstance(single_cell_data, list):
                    wkst.write_column(
                        1, column, single_cell_data, cell_format=num_format
                    )
                else:
                    wkst.write(1, column, single_cell_data)
                column += 1
