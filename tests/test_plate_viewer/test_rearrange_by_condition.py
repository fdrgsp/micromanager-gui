from micromanager_gui._plate_viewer._to_csv import _rearrange_by_condition
from micromanager_gui._plate_viewer._util import ROIData

data = {
    "key1": {
        "1": ROIData(condition_1="c1", condition_2="t1", stimulated=True),
        "2": ROIData(condition_1="c1", condition_2="t1", stimulated=True),
    },
    "key2": {
        "1": ROIData(condition_1="c2", condition_2="t1"),
        "2": ROIData(condition_1="c2", condition_2="t1"),
    },
    "key3": {
        "1": ROIData(condition_1="c3", stimulated=True),
        "2": ROIData(condition_1="c3", stimulated=True),
    },
    "key4": {
        "1": ROIData(condition_1="c3"),
        "2": ROIData(condition_1="c3"),
    },
    "key5": {
        "1": ROIData(condition_2="t2"),
        "2": ROIData(condition_2="t2"),
    },
    "key6": {
        "1": ROIData(condition_2="t2"),
        "2": ROIData(condition_2="t2"),
    },
    "key7": {
        "1": ROIData(),
        "2": ROIData(),
    },
    "key8": {
        "1": ROIData(stimulated=True),
        "2": ROIData(stimulated=True),
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
