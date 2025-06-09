#!/usr/bin/env python3
"""Test the new percentage active calculation with sample sizes."""

import numpy as np


# Test the new percentage calculation
def test_percentage_with_sample_sizes():
    """Test that we get both percentage and sample size."""

    # Create mock ROI data
    class MockROIData:
        def __init__(self, active: bool):
            self.active = active

    # Mock data: 3 ROIs, 2 active (67% with n=3)
    mock_data = {
        "condition1": {
            "fov1": {
                "roi1": MockROIData(True),
                "roi2": MockROIData(True),
                "roi3": MockROIData(False),
            }
        }
    }

    # Import and test the function
    import sys

    sys.path.insert(0, "/Users/fdrgsp/Documents/git/micromanager-gui/src")

    from micromanager_gui._plate_viewer._to_csv import _get_percentage_active_parameter

    result = _get_percentage_active_parameter(mock_data)

    print("Result:", result)

    # Check structure
    assert "condition1" in result
    assert "fov1" in result["condition1"]

    # Get the percentage and sample size
    value = result["condition1"]["fov1"][0]
    print(f"Value: {value}")

    if isinstance(value, tuple):
        percentage, n = value
        print(f"Percentage: {percentage}%, Sample size: {n}")

        # Check values
        expected_percentage = 2 / 3 * 100  # 66.67%
        assert abs(percentage - expected_percentage) < 0.01
        assert n == 3

        print("✅ Test passed! We now get both percentage and sample size.")

        # Test weighted mean calculation
        print("\n--- Testing Weighted Mean Calculation ---")

        # Simulate multiple FOVs with different sample sizes
        percentages = [66.67, 100.0, 50.0]  # From different FOVs
        sample_sizes = [3, 2, 4]  # Different sample sizes

        # Calculate weighted mean
        total_active = sum(p * n / 100 for p, n in zip(percentages, sample_sizes))
        total_n = sum(sample_sizes)
        weighted_mean = total_active / total_n * 100

        print(f"Individual percentages: {percentages}")
        print(f"Sample sizes: {sample_sizes}")
        print(f"Weighted mean: {weighted_mean:.2f}%")

        # Calculate proper binomial SEM
        p_prop = total_active / total_n  # proportion (0-1)
        binomial_sem = (p_prop * (1 - p_prop) / total_n) ** 0.5 * 100

        print(f"Proper binomial SEM: {binomial_sem:.2f}%")

        # Compare with naive mean
        naive_mean = np.mean(percentages)
        naive_sem = np.std(percentages, ddof=1) / np.sqrt(len(percentages))

        print(f"Naive mean: {naive_mean:.2f}%")
        print(f"Naive SEM: {naive_sem:.2f}%")

        print(f"\nDifference in means: {abs(weighted_mean - naive_mean):.2f}%")
        print(f"Difference in SEMs: {abs(binomial_sem - naive_sem):.2f}%")

        print("✅ Weighted statistics calculation working!")

    else:
        print("❌ Expected tuple (percentage, n) but got:", type(value))
        raise AssertionError(f"Expected tuple (percentage, n) but got: {type(value)}")


if __name__ == "__main__":
    test_percentage_with_sample_sizes()
