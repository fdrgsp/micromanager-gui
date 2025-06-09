from rich import print

from micromanager_gui._plate_viewer._to_csv import _keep_power_conditions
from micromanager_gui._plate_viewer._util import ROIData


def test_keep_power_conditions_cell_distribution():
    """Test _keep_power_conditions with specific cell distribution.

    - 10 total cells
    - 6 active cells (3 stimulated, 3 non-stimulated)
    - 4 non-active cells (all in non-stimulated area).
    """
    # Create test data structure
    data = {
        "condition1_evk_stim": {
            "fov1": {
                # 3 active stimulated cells
                "roi1": ROIData(stimulated=True, active=True),
                "roi2": ROIData(stimulated=True, active=True),
                "roi3": ROIData(stimulated=True, active=True),
            }
        },
        "condition1_evk_non_stim": {
            "fov1": {
                # 3 active non-stimulated cells
                "roi4": ROIData(stimulated=False, active=True),
                "roi5": ROIData(stimulated=False, active=True),
                "roi6": ROIData(stimulated=False, active=True),
                # 4 non-active non-stimulated cells
                "roi7": ROIData(stimulated=False, active=False),
                "roi8": ROIData(stimulated=False, active=False),
                "roi9": ROIData(stimulated=False, active=False),
                "roi10": ROIData(stimulated=False, active=False),
            }
        },
    }

    # Test for stimulated cells
    print("\n=== Testing stimulated cells ===")
    result_stimulated = _keep_power_conditions(data, stimulated=True)
    print("Stimulated cells result:", result_stimulated)

    print("\n=== Testing non-stimulated cells ===")
    # Test for non-stimulated cells
    result_non_stimulated = _keep_power_conditions(data, stimulated=False)
    print("Non-stimulated cells result:", result_non_stimulated)

    print("\n=== Verification ===")
    print("Expected: Stimulated cells should be 100% active (3/3)")
    print("Expected: Non-stimulated cells should be ~42.86% active (3/7)")

    # Test structure assertions
    assert len(result_stimulated) > 0, "Should have stimulated results"
    assert len(result_non_stimulated) > 0, "Should have non-stimulated results"

    # Verify correct conditions are returned
    assert (
        "condition1_evk_stim" in result_stimulated
    ), "Should contain stimulated condition"
    assert (
        "condition1_evk_non_stim" in result_non_stimulated
    ), "Should contain non-stimulated condition"

    # Verify no cross-contamination
    assert (
        "condition1_evk_non_stim" not in result_stimulated
    ), "Stimulated result should not contain non-stim condition"
    assert (
        "condition1_evk_stim" not in result_non_stimulated
    ), "Non-stimulated result should not contain stim condition"

    expected_non_stim_percentage = (
        42.857142857142854  # 3 active out of 7 total non-stimulated cells
    )

    # Verify stimulated percentage - actual data structure:
    # {condition: {fov: {cond: [percentages]}}}
    stim_condition = "condition1_evk_stim"
    stim_fov = "fov1"
    assert stim_condition in result_stimulated, f"Missing condition {stim_condition}"
    assert stim_fov in result_stimulated[stim_condition], f"Missing FOV {stim_fov}"

    stim_fov_data = result_stimulated[stim_condition][stim_fov]
    assert isinstance(stim_fov_data, dict), "FOV data should be a dict"
    assert len(stim_fov_data) > 0, "FOV data should not be empty"

    # Get the first (and should be only) condition key and percentage
    stim_cond_key = next(iter(stim_fov_data.keys()))
    stim_percentages = stim_fov_data[stim_cond_key]
    assert isinstance(stim_percentages, list), "Percentages should be in a list"
    assert len(stim_percentages) > 0, "Should have at least one percentage value"

    expected_stim_percentage = 100.0
    actual_stim_percentage = stim_percentages[0]
    assert actual_stim_percentage == expected_stim_percentage, (
        f"Stimulated percentage: expected {expected_stim_percentage}%, "
        f"got {actual_stim_percentage}%"
    )

    # Verify non-stimulated percentage
    non_stim_condition = "condition1_evk_non_stim"
    non_stim_fov_data = result_non_stimulated[non_stim_condition][stim_fov]
    non_stim_cond_key = next(iter(non_stim_fov_data.keys()))
    non_stim_percentages = non_stim_fov_data[non_stim_cond_key]

    actual_non_stim_percentage = non_stim_percentages[0]
    assert abs(actual_non_stim_percentage - expected_non_stim_percentage) < 0.001, (
        f"Non-stimulated percentage: expected {expected_non_stim_percentage:.3f}%, "
        f"got {actual_non_stim_percentage:.3f}%"
    )

    # Verify data structure integrity
    for _condition, fovs in result_stimulated.items():
        for _fov, cond_data in fovs.items():
            assert isinstance(
                cond_data, dict
            ), f"FOV data should be dict, got {type(cond_data)}"
            for _cond, percentages in cond_data.items():
                assert isinstance(
                    percentages, list
                ), f"Percentages should be list, got {type(percentages)}"
                for percentage in percentages:
                    assert isinstance(
                        percentage, (int, float)
                    ), f"percentage should be numeric, got {type(percentage)}"
                    assert (
                        0 <= percentage <= 100
                    ), f"percentage should be 0-100, got {percentage}"

    for _condition, fovs in result_non_stimulated.items():
        for _fov, cond_data in fovs.items():
            assert isinstance(
                cond_data, dict
            ), f"FOV data should be dict, got {type(cond_data)}"
            for _cond, percentages in cond_data.items():
                assert isinstance(
                    percentages, list
                ), f"Percentages should be list, got {type(percentages)}"
                for percentage in percentages:
                    assert isinstance(
                        percentage, (int, float)
                    ), f"percentage should be numeric, got {type(percentage)}"
                    assert (
                        0 <= percentage <= 100
                    ), f"percentage should be 0-100, got {percentage}"

    print("✅ All assertions passed! Test completed successfully!")


if __name__ == "__main__":
    test_keep_power_conditions_cell_distribution()
