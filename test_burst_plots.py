#!/usr/bin/env python3
"""Test script to verify burst activity plotting functionality."""

import tempfile
from pathlib import Path

import pandas as pd


# Create a sample burst activity CSV file
def create_test_burst_csv():
    """Create a test burst activity CSV file with sample data."""
    # Sample data for 2 conditions with burst metrics
    data = {
        "condition1_count": [3, 5, 2, 4, 3],
        "condition1_avg_duration_sec": [2.1, 1.8, 2.5, 1.9, 2.2],
        "condition1_avg_interval_sec": [15.3, 12.7, 18.1, 14.2, 16.8],
        "condition1_rate_burst_per_min": [0.8, 1.2, 0.6, 1.0, 0.9],
        "condition2_count": [7, 6, 8, 5, 6],
        "condition2_avg_duration_sec": [1.5, 1.7, 1.3, 1.9, 1.6],
        "condition2_avg_interval_sec": [8.2, 9.1, 7.5, 10.3, 8.8],
        "condition2_rate_burst_per_min": [2.1, 1.8, 2.4, 1.6, 2.0],
    }

    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)

    # Use temporary file
    temp_dir = Path(tempfile.mkdtemp())
    csv_path = temp_dir / "test_burst_activity.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_burst_parsing():
    """Test the burst activity CSV parsing function."""
    try:
        # Import the parsing function
        from src.micromanager_gui._plate_viewer._plot_methods._multi_wells_plots._csv_bar_plot import (  # noqa: E501
            _parse_csv_burst_activity_format,
        )

        # Create test CSV
        csv_path = create_test_burst_csv()
        print(f"Created test CSV at: {csv_path}")

        # Test parsing for each metric
        metrics = [
            "count",
            "avg_duration_sec",
            "avg_interval_sec",
            "rate_burst_per_min",
        ]

        for metric in metrics:
            print(f"\nTesting metric: {metric}")
            result = _parse_csv_burst_activity_format(csv_path, metric)

            if result:
                print(f"  Conditions: {result['conditions']}")
                print(f"  Means: {result['means']}")
                print(f"  SEMs: {result['sems']}")
                print("  Success!")
            else:
                print(f"  Failed to parse metric {metric}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"Error during testing: {e}")


def test_main_plot_constants():
    """Test that the main plot constants include burst activity plots."""
    try:
        from src.micromanager_gui._plate_viewer._plot_methods._main_plot import (
            CSV_BAR_PLOT_BURST_COUNT,
            CSV_BAR_PLOT_BURST_DURATION,
            CSV_BAR_PLOT_BURST_INTERVAL,
            CSV_BAR_PLOT_BURST_RATE,
            MW_GENERAL_GROUP,
        )

        print("Burst activity plot constants:")
        print(f"  Count: {CSV_BAR_PLOT_BURST_COUNT}")
        print(f"  Duration: {CSV_BAR_PLOT_BURST_DURATION}")
        print(f"  Interval: {CSV_BAR_PLOT_BURST_INTERVAL}")
        print(f"  Rate: {CSV_BAR_PLOT_BURST_RATE}")

        print("\nBurst plots in MW_GENERAL_GROUP:")
        burst_keys = [k for k in MW_GENERAL_GROUP.keys() if "Burst" in k]
        for key in burst_keys:
            print(f"  {key}: {MW_GENERAL_GROUP[key]}")

    except ImportError as e:
        print(f"Import error: {e}")


if __name__ == "__main__":
    print("Testing burst activity plotting implementation...")
    print("=" * 50)

    print("\n1. Testing CSV parsing:")
    test_burst_parsing()

    print("\n2. Testing plot constants:")
    test_main_plot_constants()

    print("\n" + "=" * 50)
    print("Test completed!")
