from pathlib import Path

from micromanager_gui._plate_viewer._to_csv import (
    _rearrange_by_parameter,
    _rearrange_fov_by_conditions,
    _save_to_csv,
)
from micromanager_gui._plate_viewer._util import ROIData

# fmt: off
data = {
    "key1": {
        "1": ROIData(well_fov_position="key1", condition_1="c1", condition_2="t1", stimulated=True, peaks_amplitudes_dec_dff=[1, 2], active=True, instantaneous_phase=[0.1, 0.2, 0.3]),  # noqa: E501
        "2": ROIData(well_fov_position="key1", condition_1="c1", condition_2="t1", stimulated=True, peaks_amplitudes_dec_dff=[1, 2], active=True, instantaneous_phase=[0.1, 0.2, 0.3]),  # noqa: E501
        "3": ROIData(well_fov_position="key1", condition_1="c1", condition_2="t1", stimulated=False, peaks_amplitudes_dec_dff=[1, 2], active=True, instantaneous_phase=[0.1, 0.2, 0.3]),  # noqa: E501
    },
    "key2": {
        "1": ROIData(well_fov_position="key2", condition_1="c2", condition_2="t1", peaks_amplitudes_dec_dff=[3, 4], active=True, instantaneous_phase=[0.3, 0.4, 0.5]),  # noqa: E501
        "2": ROIData(well_fov_position="key2", condition_1="c2", condition_2="t1", peaks_amplitudes_dec_dff=[3, 4], instantaneous_phase=[0.3, 0.4, 0.5]),  # noqa: E501
    },
    "key3": {
        "1": ROIData(condition_1="c3", stimulated=True, peaks_amplitudes_dec_dff=[5, 6]),  # noqa: E501
        "2": ROIData(condition_1="c3", stimulated=True, peaks_amplitudes_dec_dff=[5, 6]),  # noqa: E501
    },
    "key4": {
        "1": ROIData(condition_1="c3", peaks_amplitudes_dec_dff=[7, 8]),
        "2": ROIData(condition_1="c3", peaks_amplitudes_dec_dff=[7, 8]),
    },
    "key5": {
        "1": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[9, 10], active=True, instantaneous_phase=[0.3, 0.4, 0.5]),  # noqa: E501
        "2": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[9, 10], active=True, instantaneous_phase=[0.3, 0.4, 0.5]),  # noqa: E501
    },
    "key6": {
        "1": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[11, 12], active=True),
        "2": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[11, 12], instantaneous_phase=[0.3, 0.4, 0.5]),  # noqa: E501
    },
    "key7": {
        "1": ROIData(peaks_amplitudes_dec_dff=[13, 14]),
        "2": ROIData(peaks_amplitudes_dec_dff=[13, 14]),
    },
    "key8": {
        "1": ROIData(stimulated=True, peaks_amplitudes_dec_dff=[15, 16], active=True),
        "2": ROIData(stimulated=True, peaks_amplitudes_dec_dff=[15, 16], active=True),
        "3": ROIData(stimulated=True, peaks_amplitudes_dec_dff=[15, 16]),
    },
}
# fmt: on


def test_rearrange_by_condition():
    """Test the rearrange_by_condition function."""
    fov_by_condition, _ = _rearrange_fov_by_conditions(data)
    assert list(fov_by_condition.keys()) == [
        "c1_t1",
        "c2_t1",
        "c3",
        "t2",
        "NoCondition",
    ]

    assert list(fov_by_condition["c1_t1"].keys()) == ["key1"]
    assert len(fov_by_condition["c1_t1"]["key1"]) == 3
    assert list(fov_by_condition["c1_t1"]["key1"].keys()) == ["1", "2", "3"]

    assert list(fov_by_condition["c2_t1"].keys()) == ["key2"]
    assert len(fov_by_condition["c2_t1"]["key2"]) == 2
    assert list(fov_by_condition["c2_t1"]["key2"].keys()) == ["1", "2"]

    assert list(fov_by_condition["c3"].keys()) == ["key3", "key4"]
    assert len(fov_by_condition["c3"]["key4"]) == 2
    assert list(fov_by_condition["c3"]["key4"].keys()) == ["1", "2"]
    assert len(fov_by_condition["c3"]["key3"]) == 2
    assert list(fov_by_condition["c3"]["key3"].keys()) == ["1", "2"]

    assert list(fov_by_condition["t2"].keys()) == ["key5", "key6"]
    assert len(fov_by_condition["t2"]["key5"]) == 2
    assert list(fov_by_condition["t2"]["key5"].keys()) == ["1", "2"]
    assert len(fov_by_condition["t2"]["key6"]) == 2
    assert list(fov_by_condition["t2"]["key6"].keys()) == ["1", "2"]


def test_rearrange_by_condition_evk():
    _, evk_conditions = _rearrange_fov_by_conditions(data)
    assert list(evk_conditions.keys()) == [
        "c1_t1_evk_spont",
        "c2_t1_evk_spont",
        "c3_evk_spont",
        "t2_evk_spont",
        "NoCondition_evk_spont",
    ]
    # TODO: complete the test for evk conditions


def test_rearrange_by_condition_by_parameter():
    fov_by_condition, _ = _rearrange_fov_by_conditions(data)
    rearranged_by_param = _rearrange_by_parameter(
        fov_by_condition, "peaks_amplitudes_dec_dff"
    )
    # fmt: off
    assert rearranged_by_param == {
        'c1_t1': {'key1': [[1, 2], [1, 2], [1, 2]]},
        'c2_t1': {'key2': [[3, 4], [3, 4]]},
        'c3': {'key3': [[5, 6], [5, 6]], 'key4': [[7, 8], [7, 8]]},
        't2': {'key5': [[9, 10], [9, 10]], 'key6': [[11, 12], [11, 12]]},
        'NoCondition': {'key7': [[13, 14], [13, 14]], 'key8': [[15, 16], [15, 16], [15, 16]]}  # noqa: E501
    }
    # fmt: on

    # rearranged_by_active = _rearrange_by_parameter(
    #     fov_by_condition, PERCENTAGE_ACTIVE
    # )
    # assert rearranged_by_active == {
    #     "c1_t1_evk": {"key1": [100.0]},
    #     "c1_t1": {"key1": [100.0]},
    #     "c2_t1": {"key2": [50.0]},
    #     "c3_evk": {"key3": [0.0]},
    #     "c3": {"key4": [0.0]},
    #     "t2": {"key5": [100.0], "key6": [50.0]},
    #     "NoCondition": {"key7": [0.0]},
    #     "NoCondition_evk": {"key8": [66.66666666666666]},
    # }
    # rearranged_by_synchrony = _rearrange_by_parameter(fov_by_condition, SYNCHRONY)
    # assert rearranged_by_synchrony == {
    #     "c1_t1_evk": {"key1": [1.0]},
    #     "c1_t1": {"key1": [None]},
    #     "c2_t1": {"key2": [1.0]},
    #     "c3_evk": {"key3": [None]},
    #     "c3": {"key4": [None]},
    #     "t2": {"key5": [1.0], "key6": [None]},
    #     "NoCondition": {"key7": [None]},
    #     "NoCondition_evk": {"key8": [None]},
    # }


def test_save_as_csv(tmp_path: Path):
    """Test the save_as_csv function."""
    folder = tmp_path / "test_folder"
    folder.mkdir()
    _save_to_csv(folder, data)

    expected_file_fov = folder / "csv_by_conditions_and_fovs"
    assert expected_file_fov.exists()
    assert expected_file_fov.is_dir()
    assert (expected_file_fov / "test_folder_amplitude_per_fov.csv").exists()

    expected_file_condition = folder / "csv_by_conditions"
    assert expected_file_condition.exists()
    assert expected_file_condition.is_dir()
    assert (expected_file_condition / "test_folder_amplitude.csv").exists()
