from micromanager_gui._plate_viewer._to_csv import (
    _rearrange_by_condition,
    _rearrange_by_condition_by_parameter,
)
from micromanager_gui._plate_viewer._util import ROIData

data = {
    "key1": {
        "1": ROIData(
            condition_1="c1",
            condition_2="t1",
            stimulated=True,
            peaks_amplitudes_dec_dff=[1, 2],
        ),
        "2": ROIData(
            condition_1="c1",
            condition_2="t1",
            stimulated=True,
            peaks_amplitudes_dec_dff=[1, 2],
        ),
    },
    "key2": {
        "1": ROIData(
            condition_1="c2", condition_2="t1", peaks_amplitudes_dec_dff=[3, 4]
        ),
        "2": ROIData(
            condition_1="c2", condition_2="t1", peaks_amplitudes_dec_dff=[3, 4]
        ),
    },
    "key3": {
        "1": ROIData(
            condition_1="c3", stimulated=True, peaks_amplitudes_dec_dff=[5, 6]
        ),
        "2": ROIData(
            condition_1="c3", stimulated=True, peaks_amplitudes_dec_dff=[5, 6]
        ),
    },
    "key4": {
        "1": ROIData(condition_1="c3", peaks_amplitudes_dec_dff=[7, 8]),
        "2": ROIData(condition_1="c3", peaks_amplitudes_dec_dff=[7, 8]),
    },
    "key5": {
        "1": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[9, 10]),
        "2": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[9, 10]),
    },
    "key6": {
        "1": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[11, 12]),
        "2": ROIData(condition_2="t2", peaks_amplitudes_dec_dff=[11, 12]),
    },
    "key7": {
        "1": ROIData(peaks_amplitudes_dec_dff=[13, 14]),
        "2": ROIData(peaks_amplitudes_dec_dff=[13, 14]),
    },
    "key8": {
        "1": ROIData(stimulated=True, peaks_amplitudes_dec_dff=[15, 16]),
        "2": ROIData(stimulated=True, peaks_amplitudes_dec_dff=[15, 16]),
    },
}


def test_rearrange_by_condition():
    """Test the rearrange_by_condition function."""
    rearranged = _rearrange_by_condition(data)
    assert list(rearranged.keys()) == [
        "c1_t1_evk",
        "c2_t1",
        "c3_evk",
        "c3",
        "t2",
        "NoCondition",
        "NoCondition_evk",
    ]

    assert list(rearranged["c1_t1_evk"].keys()) == ["key1"]
    assert len(rearranged["c1_t1_evk"]["key1"]) == 2

    assert list(rearranged["c2_t1"].keys()) == ["key2"]
    assert len(rearranged["c2_t1"]["key2"]) == 2

    assert list(rearranged["c3_evk"].keys()) == ["key3"]
    assert len(rearranged["c3_evk"]["key3"]) == 2

    assert list(rearranged["c3"].keys()) == ["key4"]
    assert len(rearranged["c3"]["key4"]) == 2

    assert list(rearranged["t2"].keys()) == ["key5", "key6"]
    assert len(rearranged["t2"]["key5"]) == 2
    assert len(rearranged["t2"]["key6"]) == 2

    assert list(rearranged["NoCondition"].keys()) == ["key7"]
    assert len(rearranged["NoCondition"]["key7"]) == 2

    assert list(rearranged["NoCondition_evk"].keys()) == ["key8"]
    assert len(rearranged["NoCondition_evk"]["key8"]) == 2


def test_rearrange_by_condition_by_parameter():
    rearranged = _rearrange_by_condition(data)
    rearranged_by_param = _rearrange_by_condition_by_parameter(
        rearranged, "peaks_amplitudes_dec_dff"
    )
    assert rearranged_by_param == {
        "c1_t1_evk": {"key1": [[1, 2], [1, 2]]},
        "c2_t1": {"key2": [[3, 4], [3, 4]]},
        "c3_evk": {"key3": [[5, 6], [5, 6]]},
        "c3": {"key4": [[7, 8], [7, 8]]},
        "t2": {"key5": [[9, 10], [9, 10]], "key6": [[11, 12], [11, 12]]},
        "NoCondition": {"key7": [[13, 14], [13, 14]]},
        "NoCondition_evk": {"key8": [[15, 16], [15, 16]]},
    }
