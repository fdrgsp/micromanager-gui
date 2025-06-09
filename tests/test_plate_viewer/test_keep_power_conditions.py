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


def test_keep_power_conditions_integration_with_bar_plot():
    """Test integration between _keep_power_conditions and _create_bar_plot.

    This test verifies that the data structure returned by _keep_power_conditions
    can be successfully converted to CSV format and used by _create_bar_plot.
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock

    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

    from micromanager_gui._plate_viewer._graph_widgets import _MultilWellGraphWidget
    from micromanager_gui._plate_viewer._plot_methods._multi_wells_plots._csv_bar_plot import (  # noqa: E501
        _create_bar_plot,
        _parse_csv_column_format,
    )

    print("\n=== Testing _keep_power_conditions → CSV → _create_bar_plot ===")

    # Create test data with multiple conditions and FOVs for a more realistic scenario
    test_data = {
        "condition1_evk_stim_power1_50ms": {
            "fov1": {
                "roi1": ROIData(stimulated=True, active=True),
                "roi2": ROIData(stimulated=True, active=True),
                "roi3": ROIData(stimulated=True, active=False),
            },
            "fov2": {
                "roi4": ROIData(stimulated=True, active=True),
                "roi5": ROIData(stimulated=True, active=False),
                "roi6": ROIData(stimulated=True, active=False),
            },
        },
        "condition2_evk_stim_power1_50ms": {
            "fov1": {
                "roi7": ROIData(stimulated=True, active=True),
                "roi8": ROIData(stimulated=True, active=True),
                "roi9": ROIData(stimulated=True, active=True),
                "roi10": ROIData(stimulated=True, active=True),
            },
            "fov2": {
                "roi11": ROIData(stimulated=True, active=True),
                "roi12": ROIData(stimulated=True, active=False),
            },
        },
        "condition3_evk_stim_power1_50ms": {
            "fov1": {
                "roi13": ROIData(stimulated=True, active=False),
                "roi14": ROIData(stimulated=True, active=False),
                "roi15": ROIData(stimulated=True, active=False),
            }
        },
    }

    # Step 1: Get _keep_power_conditions output
    result = _keep_power_conditions(test_data, stimulated=True)
    print("_keep_power_conditions result:", result)

    # Verify the basic structure
    assert len(result) == 3, f"Expected 3 conditions, got {len(result)}"
    assert all(
        isinstance(fovs, dict) for fovs in result.values()
    ), "All condition values should be dicts"

    # Step 2: Convert to CSV format (simulating the export process)
    # This mimics what _export_to_csv_evk_single_values does
    columns = {}
    condition_names = []

    for condition, fovs in result.items():
        # Extract condition name (remove the stimulation suffix)
        if "_evk_stim_" in condition:
            condition_name = condition.replace("_evk_stim_power1_50ms", "")
        else:
            condition_name = condition

        condition_names.append(condition_name)

        # Collect all percentage values for this condition
        values = []
        for cond_data in fovs.values():
            for percentages in cond_data.values():
                values.extend(percentages)

        columns[condition_name] = values
        print(f"Condition '{condition_name}': {len(values)} values = {values}")

    # Expected values:
    # condition1: fov1 = 66.67% (2/3), fov2 = 33.33% (1/3) -> [66.67, 33.33]
    # condition2: fov1 = 100% (4/4), fov2 = 50% (1/2) -> [100.0, 50.0]
    # condition3: fov1 = 0% (0/3) -> [0.0]

    # Verify expected values
    expected_values = {
        "condition1": [66.66666666666667, 33.333333333333336],  # 2/3, 1/3
        "condition2": [100.0, 50.0],  # 4/4, 1/2
        "condition3": [0.0],  # 0/3
    }

    for condition_name, expected in expected_values.items():
        actual = columns[condition_name]
        assert len(actual) == len(expected), (
            f"Condition {condition_name}: expected {len(expected)} values, "
            f"got {len(actual)}"
        )
        for i, (act, exp) in enumerate(zip(actual, expected)):
            assert (
                abs(act - exp) < 0.001
            ), f"Condition {condition_name}[{i}]: expected {exp}, got {act}"

    print("✅ Data conversion verification passed!")

    # Step 3: Create temporary CSV file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        # Find max length for DataFrame creation
        max_len = max(len(values) for values in columns.values())

        # Pad shorter columns with NaN
        padded_columns = {
            col_name: values + [np.nan] * (max_len - len(values))
            for col_name, values in columns.items()
        }

        # Create and save DataFrame
        df = pd.DataFrame(padded_columns)
        df.to_csv(temp_file.name, index=False)
        csv_path = Path(temp_file.name)

    print(f"Created temporary CSV: {csv_path}")
    print("CSV contents:")
    print(df.to_string())

    try:
        # Step 4: Test CSV parsing
        parsed_data = _parse_csv_column_format(csv_path)
        assert parsed_data is not None, "CSV parsing should not return None"

        print("\nParsed data:")
        print(f"  Conditions: {parsed_data['conditions']}")
        print(f"  Means: {parsed_data['means']}")
        print(f"  SEMs: {parsed_data['sems']}")
        print(f"  FOV values: {[len(vals) for vals in parsed_data['fov_values_list']]}")

        # Verify parsing results
        assert len(parsed_data["conditions"]) == 3, "Should have 3 conditions"
        assert len(parsed_data["means"]) == 3, "Should have 3 means"
        assert len(parsed_data["sems"]) == 3, "Should have 3 SEMs"
        assert len(parsed_data["fov_values_list"]) == 3, "Should have 3 FOV value lists"

        # Verify computed means (manually calculated from our test data)
        expected_means = [
            50.0,  # condition1: (66.67 + 33.33) / 2 = 50.0
            75.0,  # condition2: (100.0 + 50.0) / 2 = 75.0
            0.0,  # condition3: 0.0 / 1 = 0.0
        ]

        for i, (actual_mean, expected_mean) in enumerate(
            zip(parsed_data["means"], expected_means)
        ):
            assert (
                abs(actual_mean - expected_mean) < 0.1
            ), f"Mean {i}: expected {expected_mean}, got {actual_mean}"

        print("✅ CSV parsing verification passed!")

        # Step 5: Test _create_bar_plot with mock widget
        mock_widget = Mock(spec=_MultilWellGraphWidget)
        mock_widget.figure = Mock(spec=Figure)
        mock_widget.figure.clear = Mock()
        mock_ax = Mock()
        mock_widget.figure.add_subplot = Mock(return_value=mock_ax)
        mock_widget.canvas = Mock()
        mock_widget.canvas.draw = Mock()
        mock_widget.conditions = {}

        info = {
            "parameter": "Percentage of Active Cells",
            "suffix": "percentage_active",
            "add_to_title": " (Stimulated)",
            "units": "%",
        }

        # This should not raise any exceptions
        _create_bar_plot(mock_widget, csv_path, info)

        # Verify that the plotting functions were called
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.canvas.draw.assert_called_once()

        print("✅ _create_bar_plot integration passed!")

        print("\n=== Integration Test Summary ===")
        print("✅ _keep_power_conditions output structure verified")
        print("✅ Data conversion to CSV format successful")
        print("✅ CSV parsing by _parse_csv_column_format successful")
        print("✅ _create_bar_plot accepts and processes the data successfully")
        print("✅ Complete integration pipeline working correctly!")

    finally:
        # Clean up temporary file
        csv_path.unlink()


if __name__ == "__main__":
    test_keep_power_conditions_cell_distribution()
    test_keep_power_conditions_integration_with_bar_plot()
