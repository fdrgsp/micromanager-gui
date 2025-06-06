from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from micromanager_gui import PlateViewer
from micromanager_gui._plate_viewer._analysis import EVOKED
from micromanager_gui._plate_viewer._plate_map import PlateMapData
from micromanager_gui._plate_viewer._to_csv import save_to_csv
from micromanager_gui._plate_viewer._util import ROIData

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


# EVOKED TEST DATA
TEST_DATA_PATH = Path(__file__).parent / "data" / "evoked" / "evk.tensorstore.zarr"

TEST_LABELS_PATH = str(Path(__file__).parent / "data" / "evoked" / "evk_labels")
TEST_ANALYSIS_PATH = str(Path(__file__).parent / "data" / "evoked" / "evk_analysis")

G_MAP = [PlateMapData(name="B5", row_col=(1, 4), condition=("c1", "coral"))]
T_MAP = [PlateMapData(name="B5", row_col=(1, 4), condition=("t1", "aquamarine"))]

SAVE_MAP = {
    "raw_data": {"test_analysis_raw_data.csv"},
    "dff_data": {"test_analysis_dff_data.csv"},
    "dec_dff_data": {"test_analysis_dec_dff_data.csv"},
    "grouped": {
        "test_analysis_amplitude.csv",
        "test_analysis_percentage_active.csv",
        "test_analysis_cell_size.csv",
        "test_analysis_iei.csv",
        "test_analysis_frequency.csv",
        "test_analysis_synchrony.csv",
    },
    "grouped_evk": {"test_analysis_amplitudes_stimulated_peaks.csv"},
}


def _round_numeric_values(value, reference_value):
    """Helper function to round numeric values for comparison."""
    if isinstance(value, list):
        return [round(v, 2) for v in value], [round(v, 2) for v in reference_value]
    elif isinstance(value, float):
        return round(value, 2), round(reference_value, 2)
    elif isinstance(value, dict):
        if not value or not isinstance(next(iter(value.values())), list):
            # Dict with scalar values
            return (
                {k: round(v, 2) for k, v in value.items()},
                {k: round(v, 2) for k, v in reference_value.items()},
            )
        # Dict with list values
        rounded_value = {
            k: [round(v, 2) for v in v_list] for k, v_list in value.items()
        }
        rounded_ref = {
            k: [round(v, 2) for v in v_list] for k, v_list in reference_value.items()
        }
        return rounded_value, rounded_ref
    else:
        return value, reference_value


# ignore future warnings and user warnings
@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
@pytest.mark.usefixtures("dummy_data_loader")
def test_analysis_code_evoked(qtbot: QtBot, dummy_data_loader, tmp_path: Path) -> None:
    pv = PlateViewer()
    qtbot.addWidget(pv)
    # create temporary analysis path and initialize widget
    tmp_analysis_path = tmp_path / "test_analysis/"
    tmp_analysis_path.mkdir(parents=True, exist_ok=True)
    pv.initialize_widget(str(TEST_DATA_PATH), TEST_LABELS_PATH, str(TEST_ANALYSIS_PATH))

    # add plate map
    # fmt: off
    genotype_path = Path(TEST_ANALYSIS_PATH) / "genotype_plate_map.json"
    treatment_path = Path(TEST_ANALYSIS_PATH) / "treatment_plate_map.json"
    pv._plate_map_genotype.setValue(genotype_path)
    pv._plate_map_treatment.setValue(treatment_path)
    assert pv._plate_map_genotype.value() == G_MAP
    assert pv._plate_map_treatment.value() == T_MAP
    # fmt: on

    assert pv._analysis_wdg._experiment_type_combo.currentText() == EVOKED
    assert pv._analysis_wdg.stimulation_area_path is not None

    pv._analysis_wdg.analysis_path = str(tmp_analysis_path)

    # autoselect the only 1 position in the plate map
    assert pv._analysis_wdg._prepare_for_running() == [0]

    # save the plate map data
    pv._analysis_wdg._handle_plate_map()

    # trigger analysis code
    pv._analysis_wdg._extract_trace_data_per_position(0)

    # trigger save to csv
    save_to_csv(tmp_analysis_path, pv.pv_analysis_data)

    # assert that the analysis path is created and contains the expected files
    files = [f.name for f in tmp_analysis_path.iterdir() if f.is_file()]

    assert set(files) == {
        "treatment_plate_map.json",
        "genotype_plate_map.json",
        "B5_0000_p0.json",
        "stimulation_mask.tif",
        "settings.json",
    }, f"Expected files not found. Found: {set(files)}"

    # assert that the subfolders are created and contain the expected files
    subfolders = [f.name for f in tmp_analysis_path.iterdir() if f.is_dir()]
    assert set(subfolders) == set(
        SAVE_MAP.keys()
    ), f"Expected subfolders not found. Found: {set(subfolders)}"
    for dir_name in subfolders:
        dir_path = tmp_analysis_path / dir_name
        assert dir_path.iterdir(), f"Directory {dir_name} is empty"
        file_list = {f.name for f in dir_path.iterdir() if f.is_file()}
        assert file_list == SAVE_MAP[dir_name], (
            f"Files in {dir_name} don't match expected. "
            f"Found: {file_list}, Expected: {SAVE_MAP[dir_name]}"
        )

    # assert that the analysis data is saved correctly compared to the reference data
    saved_file = tmp_analysis_path / "B5_0000_p0.json"
    with open(saved_file) as file:
        data = cast(dict, json.load(file))
    reference_file = (
        Path(__file__).parent / "data" / "evoked" / "evk_analysis" / "B5_0000_p0.json"
    )
    with open(reference_file) as file1:
        reference_data = cast(dict, json.load(file1))

    # Compare all ROIs in the data
    assert set(data.keys()) == set(
        reference_data.keys()
    ), f"ROI keys mismatch: {set(data.keys())} != {set(reference_data.keys())}"

    for roi_id in data.keys():
        roi_data = ROIData(**data[roi_id])
        roi_data1 = ROIData(**reference_data[roi_id])

        # loop through the ROIData attributes and compare them
        for attr, value in roi_data.__dict__.items():
            reference_value = roi_data1.__dict__[attr]

            # Round numeric values for comparison
            value_rounded, ref_rounded = _round_numeric_values(value, reference_value)

            assert (
                value_rounded == ref_rounded
            ), f"ROI {roi_id} mismatch in {attr}: {value_rounded} != {ref_rounded}"
