from micromanager_gui._plate_viewer._to_csv import _rearrange_by_condition
from micromanager_gui._plate_viewer._util import ROIData

data = {
    "A1_0000_p0": {
        "1": ROIData(condition_1="c1", condition_2="t1"),
        "2": ROIData(condition_1="c1", condition_2="t1"),
    },
    "A2_0000_p0": {
        "1": ROIData(condition_1="c2", condition_2="t1"),
        "2": ROIData(condition_1="c2", condition_2="t1"),
    },
    "B2_0000_p0": {
        "1": ROIData(condition_1="c3"),
        "2": ROIData(condition_1="c3"),
    },
    "B2_0001_p1": {
        "1": ROIData(condition_1="c3"),
        "2": ROIData(condition_1="c3"),
    },
    "C3_0000_p0": {
        "1": ROIData(condition_2="t2"),
        "2": ROIData(condition_2="t2"),
    },
    "C3_0001_p1": {
        "1": ROIData(condition_2="t2"),
        "2": ROIData(condition_2="t2"),
    },
    "D4_0000_p0": {
        "1": ROIData(),
        "2": ROIData(),
    },
    "D4_0001_p1": {
        "1": ROIData(),
        "2": ROIData(),
    },
}


def test_rearrange_by_condition():
    """Test the rearrange_by_condition function."""
    rearranged = _rearrange_by_condition(data)
    assert list(rearranged.keys()) == ["c1_t1", "c2_t1", "c3", "t2", "NoCondition"]
    assert list(rearranged["c1_t1"].keys()) == ["A1_0000_p0"]
    assert len(rearranged["c1_t1"]["A1_0000_p0"]) == 2
    assert list(rearranged["c2_t1"].keys()) == ["A2_0000_p0"]
    assert len(rearranged["c2_t1"]["A2_0000_p0"]) == 2
    assert list(rearranged["c3"].keys()) == ["B2_0000_p0", "B2_0001_p1"]
    assert len(rearranged["c3"]["B2_0000_p0"]) == 2
    assert len(rearranged["c3"]["B2_0001_p1"]) == 2
    assert list(rearranged["t2"].keys()) == ["C3_0000_p0", "C3_0001_p1"]
    assert len(rearranged["t2"]["C3_0000_p0"]) == 2
    assert len(rearranged["t2"]["C3_0001_p1"]) == 2
    assert list(rearranged["NoCondition"].keys()) == ["D4_0000_p0", "D4_0001_p1"]
    assert len(rearranged["NoCondition"]["D4_0000_p0"]) == 2
    assert len(rearranged["NoCondition"]["D4_0001_p1"]) == 2
