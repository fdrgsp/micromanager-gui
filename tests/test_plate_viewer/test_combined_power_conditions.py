from rich import print

from micromanager_gui._plate_viewer._to_csv import _combined_power_conditions
from micromanager_gui._plate_viewer._util import ROIData


def test_combined_power_conditions_cell_distribution():
    """Test _combined_power_conditions with specific cell distribution.

    - Multiple power conditions that should be combined together
    - Tests that percentage values from different power conditions are combined
    - 8 total stimulated cells across power conditions (4+4)
    - 6 active stimulated cells (3+3)
    - 8 total non-stimulated cells across power conditions (4+4)
    - 5 active non-stimulated cells (2+3)
    """
    # Create test data structure with multiple power conditions
    # The function expects EVK_STIM conditions to contain stimulated cells
    # and EVK_NON_STIM conditions to contain non-stimulated cells
    data = {
        "condition1_evk_stim_power1_100ms": {
            "fov1": {
                # 3 active stimulated cells, 1 inactive in power1
                "roi1": ROIData(stimulated=True, active=True),
                "roi2": ROIData(stimulated=True, active=True),
                "roi3": ROIData(stimulated=True, active=True),
                "roi4": ROIData(stimulated=True, active=False),
            }
        },
        "condition1_evk_stim_power2_200ms": {
            "fov1": {
                # 3 active stimulated cells, 1 inactive in power2
                "roi5": ROIData(stimulated=True, active=True),
                "roi6": ROIData(stimulated=True, active=True),
                "roi7": ROIData(stimulated=True, active=True),
                "roi8": ROIData(stimulated=True, active=False),
            }
        },
        "condition1_evk_non_stim_power1_100ms": {
            "fov1": {
                # 2 active non-stimulated cells, 2 inactive in power1
                "roi9": ROIData(stimulated=False, active=True),
                "roi10": ROIData(stimulated=False, active=True),
                "roi11": ROIData(stimulated=False, active=False),
                "roi12": ROIData(stimulated=False, active=False),
            }
        },
        "condition1_evk_non_stim_power2_200ms": {
            "fov1": {
                # 3 active non-stimulated cells, 1 inactive in power2
                "roi13": ROIData(stimulated=False, active=True),
                "roi14": ROIData(stimulated=False, active=True),
                "roi15": ROIData(stimulated=False, active=True),
                "roi16": ROIData(stimulated=False, active=False),
            }
        },
    }

    print("\n=== Testing stimulated cells (combined power conditions) ===")
    # Test for stimulated cells - should combine all power conditions
    result_stimulated = _combined_power_conditions(data, stimulated=True)
    print("Stimulated cells result:", result_stimulated)

    print("\n=== Testing non-stimulated cells (combined power conditions) ===")
    # Test for non-stimulated cells - should combine all power conditions
    result_non_stimulated = _combined_power_conditions(data, stimulated=False)
    print("Non-stimulated cells result:", result_non_stimulated)

    print("\n=== Verification ===")
    print("Expected: Stimulated cells should be 75% active per power condition")
    print("  (3/4 in power1, 3/4 in power2) -> [75.0, 75.0]")
    print("Expected: Non-stimulated cells should be 50% and 75% active per power")
    print("  (2/4 in power1, 3/4 in power2) -> [50.0, 75.0]")

    # Test structure assertions
    assert len(result_stimulated) > 0, "Should have stimulated results"
    assert len(result_non_stimulated) > 0, "Should have non-stimulated results"

    # Verify correct combined conditions are returned
    assert (
        "condition1_evk_stim" in result_stimulated
    ), "Should contain combined stimulated condition"
    assert (
        "condition1_evk_non_stim" in result_non_stimulated
    ), "Should contain combined non-stimulated condition"

    # Verify combining behavior - should have only one condition per stimulation type
    # but should contain multiple percentage values (one per power condition)
    assert (
        len(result_stimulated) == 1
    ), "Should combine all power conditions into one stimulated condition"
    assert (
        len(result_non_stimulated) == 1
    ), "Should combine all power conditions into one non-stimulated condition"

    # Verify stimulated percentages
    # actual data structure: {condition: {fov: {cond: [percentages]}}}
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

    # Check that we have multiple percentage values (one per power condition)
    assert len(stim_percentages) == 2, "Should have percentages from 2 power conditions"

    # Verify no cross-contamination
    assert (
        "condition1_evk_non_stim" not in result_stimulated
    ), "Stimulated result should not contain non-stim condition"
    assert (
        "condition1_evk_stim" not in result_non_stimulated
    ), "Non-stimulated result should not contain stim condition"

    # Expected: power1 = 50% (2/4), power2 = 75% (3/4)
    expected_non_stim_percentages = [50.0, 75.0]

    # Verify non-stimulated percentages
    non_stim_condition = "condition1_evk_non_stim"
    non_stim_fov_data = result_non_stimulated[non_stim_condition][stim_fov]
    non_stim_cond_key = next(iter(non_stim_fov_data.keys()))
    non_stim_percentages = non_stim_fov_data[non_stim_cond_key]

    expected_stim_percentages = [75.0, 75.0]
    # Verify percentages match expected values
    assert len(stim_percentages) == len(expected_stim_percentages), (
        f"Expected {len(expected_stim_percentages)} percentages, "
        f"got {len(stim_percentages)}"
    )
    for i, (actual, expected) in enumerate(
        zip(stim_percentages, expected_stim_percentages)
    ):
        assert (
            abs(actual - expected) < 0.001
        ), f"Stimulated percentage {i}: expected {expected}%, got {actual}%"

    # Verify percentages match expected values
    assert len(non_stim_percentages) == len(expected_non_stim_percentages), (
        f"Expected {len(expected_non_stim_percentages)} percentages, "
        f"got {len(non_stim_percentages)}"
    )
    for i, (actual, expected) in enumerate(
        zip(non_stim_percentages, expected_non_stim_percentages)
    ):
        assert abs(actual - expected) < 0.001, (
            f"Non-stimulated percentage {i}: expected {expected:.3f}%, "
            f"got {actual:.3f}%"
        )

    # Verify data structure integrity
    for _, fovs in result_stimulated.items():
        for _, cond_data in fovs.items():
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

    for _, fovs in result_non_stimulated.items():
        for _, cond_data in fovs.items():
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


def test_combined_power_conditions_multiple_fovs():
    """
    Test _combined_power_conditions with multiple FOVs and power conditions.
    Verifies that power conditions are combined correctly across different FOVs.
    """
    # Create test data structure with multiple FOVs and power conditions
    data = {
        "condition1_evk_stim_power1_100ms": {
            "fov1": {
                "roi1": ROIData(stimulated=True, active=True),
                "roi2": ROIData(stimulated=True, active=False),
            },
            "fov2": {
                "roi3": ROIData(stimulated=True, active=True),
                "roi4": ROIData(stimulated=True, active=True),
            },
        },
        "condition1_evk_stim_power2_200ms": {
            "fov1": {
                "roi5": ROIData(stimulated=True, active=True),
                "roi6": ROIData(stimulated=True, active=True),
            },
            "fov2": {
                "roi7": ROIData(stimulated=True, active=False),
                "roi8": ROIData(stimulated=True, active=False),
            },
        },
        "condition1_evk_non_stim_power1_100ms": {
            "fov1": {
                "roi9": ROIData(stimulated=False, active=True),
                "roi10": ROIData(stimulated=False, active=False),
            },
            "fov2": {
                "roi11": ROIData(stimulated=False, active=False),
                "roi12": ROIData(stimulated=False, active=False),
            },
        },
        "condition1_evk_non_stim_power2_200ms": {
            "fov1": {
                "roi13": ROIData(stimulated=False, active=True),
                "roi14": ROIData(stimulated=False, active=True),
            },
            "fov2": {
                "roi15": ROIData(stimulated=False, active=True),
                "roi16": ROIData(stimulated=False, active=False),
            },
        },
    }

    # Test combining stimulated conditions
    result_stimulated = _combined_power_conditions(data, stimulated=True)

    # Should have one combined condition with multiple FOVs
    assert len(result_stimulated) == 1, "Should combine all power conditions"
    combined_condition = "condition1_evk_stim"
    assert combined_condition in result_stimulated, "Should have combined condition"

    # Should have both FOVs
    assert "fov1" in result_stimulated[combined_condition], "Should have fov1"
    assert "fov2" in result_stimulated[combined_condition], "Should have fov2"

    # Verify calculations for each FOV
    # FOV1: power1 = 1/2 = 50%, power2 = 2/2 = 100%
    fov1_data = result_stimulated[combined_condition]["fov1"]
    fov1_percentages = fov1_data["percentage_active"]
    expected_fov1_percentages = [50.0, 100.0]  # power1, power2
    assert len(fov1_percentages) == 2, "Should have 2 power conditions"
    for i, (actual, expected) in enumerate(
        zip(fov1_percentages, expected_fov1_percentages)
    ):
        assert (
            abs(actual - expected) < 0.001
        ), f"FOV1 power{i+1}: expected {expected}%, got {actual}%"

    # FOV2: power1 = 2/2 = 100%, power2 = 0/2 = 0%
    fov2_data = result_stimulated[combined_condition]["fov2"]
    fov2_percentages = fov2_data["percentage_active"]
    expected_fov2_percentages = [100.0, 0.0]  # power1, power2
    assert len(fov2_percentages) == 2, "Should have 2 power conditions"
    for i, (actual, expected) in enumerate(
        zip(fov2_percentages, expected_fov2_percentages)
    ):
        assert (
            abs(actual - expected) < 0.001
        ), f"FOV2 power{i+1}: expected {expected}%, got {actual}%"

    # Test combining non-stimulated conditions
    result_non_stimulated = _combined_power_conditions(data, stimulated=False)

    # Should have one combined condition with multiple FOVs
    assert len(result_non_stimulated) == 1, "Should combine all power conditions"
    combined_condition = "condition1_evk_non_stim"
    combined_condition_key = combined_condition
    assert (
        combined_condition_key in result_non_stimulated
    ), "Should have combined condition"

    # Verify calculations for each FOV
    # FOV1: power1 = 1/2 = 50%, power2 = 2/2 = 100%
    fov1_data = result_non_stimulated[combined_condition_key]["fov1"]
    fov1_percentages = fov1_data["percentage_active"]
    expected_fov1_percentages = [50.0, 100.0]  # power1, power2
    assert len(fov1_percentages) == 2, "Should have 2 power conditions"
    for i, (actual, expected) in enumerate(
        zip(fov1_percentages, expected_fov1_percentages)
    ):
        assert (
            abs(actual - expected) < 0.001
        ), f"FOV1 power{i+1}: expected {expected}%, got {actual}%"

    # FOV2: power1 = 0/2 = 0%, power2 = 1/2 = 50%
    fov2_data = result_non_stimulated[combined_condition_key]["fov2"]
    fov2_percentages = fov2_data["percentage_active"]
    expected_fov2_percentages = [0.0, 50.0]  # power1, power2
    assert len(fov2_percentages) == 2, "Should have 2 power conditions"
    for i, (actual, expected) in enumerate(
        zip(fov2_percentages, expected_fov2_percentages)
    ):
        assert (
            abs(actual - expected) < 0.001
        ), f"FOV2 power{i+1}: expected {expected}%, got {actual}%"

    print("✅ Multiple FOVs test passed!")


def test_combined_power_conditions_edge_cases():
    """
    Test _combined_power_conditions with edge cases:
    - Empty data
    - No matching conditions
    - Zero cells of target stimulation type
    """
    # Test with empty data
    result_empty = _combined_power_conditions({}, stimulated=True)
    assert result_empty == {}, "Empty data should return empty result"

    # Test with no matching conditions (no evk_stim/evk_non_stim conditions)
    data_no_match = {
        "condition1": {
            "fov1": {
                "roi1": ROIData(stimulated=True, active=True),
            }
        }
    }
    result_no_match = _combined_power_conditions(data_no_match, stimulated=True)
    assert result_no_match == {}, "No matching conditions should return empty result"

    # Test with no cells of target stimulation type
    data_no_target_type = {
        "condition1_evk_stim_power1_100ms": {
            "fov1": {
                # Only non-stimulated cells
                "roi1": ROIData(stimulated=False, active=True),
                "roi2": ROIData(stimulated=False, active=False),
            }
        }
    }
    result_no_target = _combined_power_conditions(data_no_target_type, stimulated=True)
    assert result_no_target == {}, "No cells of target type should return empty result"

    print("✅ Edge cases test passed!")


if __name__ == "__main__":
    test_combined_power_conditions_cell_distribution()
    test_combined_power_conditions_multiple_fovs()
    test_combined_power_conditions_edge_cases()
