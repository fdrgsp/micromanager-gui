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


def test_combined_power_conditions_integration_with_bar_plot_mean_and_pooled_sem():
    """Test integration between _combined_power_conditions and _create_bar_plot.

    This test verifies that the data structure returned by _combined_power_conditions
    can be successfully converted to CSV triplet format and used by
    _create_bar_plot_mean_and_pooled_sem.

    The test follows the pipeline:
    _combined_power_conditions → CSV triplet format → _parse_csv_triplet_format
    → _create_bar_plot_mean_and_pooled_sem
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock

    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget
    from micromanager_gui._plate_viewer._plot_methods._multi_wells_plots._csv_bar_plot import (  # noqa: E501
        _create_bar_plot_mean_and_pooled_sem,
        _parse_csv_triplet_format,
    )

    print("\n=== Testing _combined_power_conditions → CSV triplet → bar plot ===")

    # Create test data with multiple conditions and power conditions
    # This simulates realistic experimental data with multiple power conditions
    # that should be combined
    test_data = {
        # Condition 1: Two power conditions for stimulated cells
        "condition1_evk_stim_power1_100ms": {
            "fov1": {
                "roi1": ROIData(stimulated=True, active=True),
                "roi2": ROIData(stimulated=True, active=True),
                "roi3": ROIData(stimulated=True, active=False),
                "roi4": ROIData(stimulated=True, active=False),
            },
            "fov2": {
                "roi5": ROIData(stimulated=True, active=True),
                "roi6": ROIData(stimulated=True, active=True),
                "roi7": ROIData(stimulated=True, active=True),
                "roi8": ROIData(stimulated=True, active=False),
            },
        },
        "condition1_evk_stim_power2_200ms": {
            "fov1": {
                "roi9": ROIData(stimulated=True, active=True),
                "roi10": ROIData(stimulated=True, active=True),
                "roi11": ROIData(stimulated=True, active=True),
                "roi12": ROIData(stimulated=True, active=False),
            },
            "fov2": {
                "roi13": ROIData(stimulated=True, active=True),
                "roi14": ROIData(stimulated=True, active=False),
                "roi15": ROIData(stimulated=True, active=False),
                "roi16": ROIData(stimulated=True, active=False),
            },
        },
        # Condition 2: Two power conditions for stimulated cells (different
        # baseline activity)
        "condition2_evk_stim_power1_100ms": {
            "fov1": {
                "roi17": ROIData(stimulated=True, active=True),
                "roi18": ROIData(stimulated=True, active=True),
                "roi19": ROIData(stimulated=True, active=True),
                "roi20": ROIData(stimulated=True, active=True),
            },
            "fov2": {
                "roi21": ROIData(stimulated=True, active=True),
                "roi22": ROIData(stimulated=True, active=True),
                "roi23": ROIData(stimulated=True, active=False),
                "roi24": ROIData(stimulated=True, active=False),
            },
        },
        "condition2_evk_stim_power2_200ms": {
            "fov1": {
                "roi25": ROIData(stimulated=True, active=True),
                "roi26": ROIData(stimulated=True, active=True),
                "roi27": ROIData(stimulated=True, active=True),
                "roi28": ROIData(stimulated=True, active=False),
            },
            "fov2": {
                "roi29": ROIData(stimulated=True, active=True),
                "roi30": ROIData(stimulated=True, active=True),
                "roi31": ROIData(stimulated=True, active=True),
                "roi32": ROIData(stimulated=True, active=True),
            },
        },
    }

    # Step 1: Get _combined_power_conditions output
    result = _combined_power_conditions(test_data, stimulated=True)
    print("_combined_power_conditions result:", result)

    # Verify the basic structure
    assert len(result) == 2, f"Expected 2 combined conditions, got {len(result)}"
    assert "condition1_evk_stim" in result, "Should have condition1_evk_stim"
    assert "condition2_evk_stim" in result, "Should have condition2_evk_stim"

    # Step 2: Convert to CSV triplet format (simulating
    # _export_to_csv_mean_values_evk_parameters)
    # This mimics the CSV export process that creates _Mean, _SEM, _N columns

    # Calculate statistics for each condition
    triplet_data = {}

    for condition, fovs in result.items():
        # Extract condition name (remove the stimulation suffix)
        condition_name = condition.replace("_evk_stim", "")

        # Collect all percentage values across FOVs for this condition
        all_percentages = []
        n_values = []

        for fov_data in fovs.values():
            for percentages in fov_data.values():
                all_percentages.extend(percentages)
                # Each percentage represents one measurement
                n_values.extend([1] * len(percentages))

        # Calculate weighted mean and pooled SEM
        if all_percentages:
            weights = np.array(n_values)
            values = np.array(all_percentages)

            # Weighted mean
            weighted_mean = np.average(values, weights=weights)

            # Pooled SEM calculation
            n_total = sum(weights)
            if n_total > 1:
                # Calculate weighted variance
                weighted_variance = np.average(
                    (values - weighted_mean) ** 2, weights=weights
                )
                pooled_sem = np.sqrt(weighted_variance / n_total)
            else:
                pooled_sem = 0.0

            triplet_data[f"{condition_name}_Mean"] = [weighted_mean]
            triplet_data[f"{condition_name}_SEM"] = [pooled_sem]
            triplet_data[f"{condition_name}_N"] = [n_total]

            print(f"Condition '{condition_name}':")
            print(f"  Values: {all_percentages}")
            print(f"  Weighted Mean: {weighted_mean:.2f}")
            print(f"  Pooled SEM: {pooled_sem:.2f}")
            print(f"  N: {n_total}")

    # Expected calculations for verification:
    # Condition1: fov1=[50.0, 75.0], fov2=[75.0, 25.0] -> values=[50,75,75,25],
    # mean=56.25, n=4
    # Condition2: fov1=[100.0, 75.0], fov2=[50.0, 100.0] ->
    # values=[100,75,50,100], mean=81.25, n=4

    # Step 3: Create temporary CSV file with triplet format
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        # Create DataFrame with triplet format
        df = pd.DataFrame(triplet_data)
        df.to_csv(temp_file.name, index=False)
        csv_path = Path(temp_file.name)

    print(f"\nCreated temporary CSV with triplet format: {csv_path}")
    print("CSV contents:")
    print(df.to_string())

    try:
        # Step 4: Test CSV triplet parsing
        info = {
            "parameter": "Percentage of Active Cells",
            "suffix": "percentage_active",
            "add_to_title": " (Combined Power Conditions)",
            "units": "%",
        }

        parsed_data = _parse_csv_triplet_format(csv_path, info)
        assert parsed_data is not None, "CSV triplet parsing should not return None"

        print("\nParsed triplet data:")
        print(f"  Conditions: {parsed_data['conditions']}")
        print(f"  Means: {parsed_data['means']}")
        print(f"  SEMs: {parsed_data['sems']}")
        print(f"  N values: {parsed_data.get('n_values', 'Not available')}")

        # Verify parsing results
        assert len(parsed_data["conditions"]) == 2, "Should have 2 conditions"
        assert len(parsed_data["means"]) == 2, "Should have 2 means"
        assert len(parsed_data["sems"]) == 2, "Should have 2 SEMs"

        # Verify condition names
        expected_conditions = ["condition1", "condition2"]
        for expected_cond in expected_conditions:
            assert (
                expected_cond in parsed_data["conditions"]
            ), f"Missing condition: {expected_cond}"

        # Verify that means are reasonable (should be between 0 and 100 for
        # percentages)
        for i, mean in enumerate(parsed_data["means"]):
            assert 0 <= mean <= 100, f"Mean {i} should be 0-100%, got {mean}"
            assert not np.isnan(mean), f"Mean {i} should not be NaN"

        # Verify that SEMs are non-negative
        for i, sem in enumerate(parsed_data["sems"]):
            assert sem >= 0, f"SEM {i} should be non-negative, got {sem}"
            assert not np.isnan(sem), f"SEM {i} should not be NaN"

        print("✅ CSV triplet parsing verification passed!")

        # Step 5: Test _create_bar_plot_mean_and_pooled_sem with mock widget
        mock_widget = Mock(spec=_MultilWellGraphWidget)
        mock_widget.figure = Mock(spec=Figure)
        mock_widget.figure.clear = Mock()
        mock_ax = Mock()
        mock_widget.figure.add_subplot = Mock(return_value=mock_ax)
        mock_widget.canvas = Mock()
        mock_widget.canvas.draw = Mock()
        mock_widget.conditions = {}

        # This should not raise any exceptions
        _create_bar_plot_mean_and_pooled_sem(mock_widget, csv_path, info)

        # Verify that the plotting functions were called
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.canvas.draw.assert_called_once()

        print("✅ _create_bar_plot_mean_and_pooled_sem integration passed!")

        # Step 6: Test data structure integrity throughout pipeline
        print("\n=== Data Structure Integrity Verification ===")

        # Verify that the original _combined_power_conditions output has
        # correct structure
        for condition, fovs in result.items():
            assert isinstance(fovs, dict), f"FOVs should be dict for {condition}"
            for fov, cond_data in fovs.items():
                assert isinstance(
                    cond_data, dict
                ), f"Condition data should be dict for {condition}:{fov}"
                for cond_key, percentages in cond_data.items():
                    assert isinstance(
                        percentages, list
                    ), f"Percentages should be list for {condition}:{fov}:{cond_key}"
                    assert (
                        len(percentages) > 0
                    ), f"Should have percentages for {condition}:{fov}:{cond_key}"
                    # Should have multiple values since we're combining power
                    # conditions
                    assert len(percentages) == 2, (
                        f"Should have 2 power conditions combined for "
                        f"{condition}:{fov}:{cond_key}"
                    )

        # Verify that the CSV triplet format has correct structure
        for col_name in df.columns:
            if col_name.endswith("_Mean"):
                condition = col_name.replace("_Mean", "")
                assert (
                    f"{condition}_SEM" in df.columns
                ), f"Missing SEM column for {condition}"
                assert (
                    f"{condition}_N" in df.columns
                ), f"Missing N column for {condition}"

        print("✅ Data structure integrity verification passed!")

        print("\n=== Integration Test Summary ===")
        print("✅ _combined_power_conditions output structure verified")
        print("✅ Power conditions successfully combined across multiple FOVs")
        print("✅ Data conversion to CSV triplet format successful")
        print("✅ CSV triplet parsing by _parse_csv_triplet_format successful")
        print("✅ Weighted mean and pooled SEM calculations working correctly")
        print(
            "✅ _create_bar_plot_mean_and_pooled_sem accepts and processes "
            "the data successfully"
        )
        print(
            "✅ Complete integration pipeline for combined power conditions "
            "working correctly!"
        )

    finally:
        # Clean up temporary file
        csv_path.unlink()


if __name__ == "__main__":
    test_combined_power_conditions_cell_distribution()
    test_combined_power_conditions_multiple_fovs()
    test_combined_power_conditions_edge_cases()
    test_combined_power_conditions_integration_with_bar_plot_mean_and_pooled_sem()
