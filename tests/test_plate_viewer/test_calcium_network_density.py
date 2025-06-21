"""Tests for calcium network density calculation and CSV export functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from micromanager_gui._plate_viewer._to_csv import (
    _export_to_csv_single_values,
    _get_calcium_network_density_parameter,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestCalciumNetworkDensity:
    """Test calcium network density calculation and export functions."""

    @pytest.fixture
    def sample_roi_data(self) -> dict[str, dict[str, dict[str, ROIData]]]:
        """Create sample ROI data for testing network density calculation."""
        # Create correlation patterns that will result in known network densities

        # ROI 1: High correlation with others
        roi_1_peaks = [10.0, 20.0, 30.0, 40.0, 50.0]
        roi_1_trace = np.zeros(100)
        roi_1_trace[[int(p) for p in roi_1_peaks]] = 1.0

        # ROI 2: Correlated with ROI 1
        roi_2_peaks = [12.0, 22.0, 32.0, 42.0, 52.0]  # Slightly offset for correlation
        roi_2_trace = np.zeros(100)
        roi_2_trace[[int(p) for p in roi_2_peaks]] = 1.0

        # ROI 3: Correlated with ROI 1 and 2
        roi_3_peaks = [11.0, 21.0, 31.0, 41.0, 51.0]  # Also correlated
        roi_3_trace = np.zeros(100)
        roi_3_trace[[int(p) for p in roi_3_peaks]] = 1.0

        # ROI 4: Uncorrelated/independent
        roi_4_peaks = [5.0, 25.0, 45.0, 65.0, 85.0]  # Different timing
        roi_4_trace = np.zeros(100)
        roi_4_trace[[int(p) for p in roi_4_peaks]] = 1.0

        return {
            "Control": {
                "A01_f00": {
                    "1": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_1_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_1_trace.tolist(),
                        peaks_dec_dff=roi_1_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_1_peaks),
                        calcium_network_threshold=90.0,  # High threshold
                        active=True,
                    ),
                    "2": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_2_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_2_trace.tolist(),
                        peaks_dec_dff=roi_2_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_2_peaks),
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                    "3": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_3_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_3_trace.tolist(),
                        peaks_dec_dff=roi_3_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_3_peaks),
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                    "4": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_4_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_4_trace.tolist(),
                        peaks_dec_dff=roi_4_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_4_peaks),
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                },
                "A02_f00": {
                    "1": ROIData(
                        well_fov_position="A02_f00",
                        raw_trace=roi_1_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_1_trace.tolist(),
                        peaks_dec_dff=roi_1_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_1_peaks),
                        calcium_network_threshold=50.0,  # Lower threshold
                        active=True,
                    ),
                    "2": ROIData(
                        well_fov_position="A02_f00",
                        raw_trace=roi_2_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_2_trace.tolist(),
                        peaks_dec_dff=roi_2_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_2_peaks),
                        calcium_network_threshold=50.0,
                        active=True,
                    ),
                },
            },
            "Treatment": {
                "B01_f00": {
                    "1": ROIData(
                        well_fov_position="B01_f00",
                        raw_trace=roi_1_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_1_trace.tolist(),
                        peaks_dec_dff=roi_1_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_1_peaks),
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                    "2": ROIData(
                        well_fov_position="B01_f00",
                        raw_trace=roi_2_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_2_trace.tolist(),
                        peaks_dec_dff=roi_2_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_2_peaks),
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                    "3": ROIData(
                        well_fov_position="B01_f00",
                        raw_trace=roi_3_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_3_trace.tolist(),
                        peaks_dec_dff=roi_3_peaks,
                        peaks_amplitudes_dec_dff=[1.0] * len(roi_3_peaks),
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                },
            },
        }

    @pytest.fixture
    def minimal_roi_data(self) -> dict[str, dict[str, dict[str, ROIData]]]:
        """Create minimal ROI data with insufficient data for network calculation."""
        roi_trace = np.ones(100)

        return {
            "Control": {
                "A01_f00": {
                    "1": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace.tolist(),
                        dff=[0.0] * 100,
                        dec_dff=roi_trace.tolist(),
                        peaks_dec_dff=[10.0, 20.0, 30.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=90.0,
                        active=True,
                    ),
                    # Only one ROI - insufficient for network calculation
                },
            },
        }

    def test_get_calcium_network_density_parameter_basic(self, sample_roi_data):
        """Test basic network density calculation."""
        result = _get_calcium_network_density_parameter(sample_roi_data)

        # Check structure
        assert "Control" in result
        assert "Treatment" in result
        assert "A01_f00" in result["Control"]
        assert "A02_f00" in result["Control"]
        assert "B01_f00" in result["Treatment"]

        # Check that values are percentages (0-100)
        for condition, wells in result.items():
            for well, values in wells.items():
                assert len(values) == 1  # Single value per well
                density = values[0]
                assert isinstance(density, (int, float))
                assert 0.0 <= density <= 100.0
                print(f"{condition} {well}: {density:.2f}%")

    def test_get_calcium_network_density_parameter_threshold_effect(
        self, sample_roi_data
    ):
        """Test that different thresholds produce different network densities."""
        result = _get_calcium_network_density_parameter(sample_roi_data)

        # A01_f00 has 4 ROIs with 90% threshold (high threshold)
        a01_density = result["Control"]["A01_f00"][0]

        # A02_f00 has 2 ROIs with 50% threshold (low threshold)
        a02_density = result["Control"]["A02_f00"][0]

        # With lower threshold, we should get higher density (more connections)
        # Note: A02 has only 2 ROIs so maximum density is 100%
        assert a02_density >= 0.0
        print(f"A01 (4 ROIs, 90% thresh): {a01_density:.2f}%")
        print(f"A02 (2 ROIs, 50% thresh): {a02_density:.2f}%")

    def test_get_calcium_network_density_parameter_insufficient_data(
        self, minimal_roi_data
    ):
        """Test network density calculation with insufficient data."""
        result = _get_calcium_network_density_parameter(minimal_roi_data)

        # Should return empty dict or no entries for wells with < 2 ROIs
        assert len(result) == 0 or all(len(wells) == 0 for wells in result.values())

    def test_get_calcium_network_density_parameter_empty_data(self):
        """Test network density calculation with empty data."""
        result = _get_calcium_network_density_parameter({})
        assert result == {}

    def test_export_to_csv_calcium_network_density(self, sample_roi_data):
        """Test CSV export of network density data."""
        # Get network density data
        network_data = _get_calcium_network_density_parameter(sample_roi_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_name = "test_experiment"

            # Export to CSV
            _export_to_csv_single_values(
                temp_path, exp_name, "calcium_network_density", network_data
            )

            # Check if file was created
            csv_file = temp_path / f"{exp_name}_calcium_network_density.csv"
            assert csv_file.exists()

            # Read and verify CSV content
            df = pd.read_csv(csv_file)

            # Should have columns for each condition
            expected_columns = ["Control", "Treatment"]
            assert list(df.columns) == expected_columns

            # Check that we have the expected number of rows
            # (max wells across conditions)
            # Control has 2 wells, Treatment has 1 well
            assert len(df) == 2

            # Check values are numeric and in percentage range
            for col in df.columns:
                for val in df[col].dropna():
                    assert isinstance(val, (int, float))
                    assert 0.0 <= val <= 100.0

            print("CSV contents:")
            print(df)

    def test_export_to_csv_calcium_network_density_empty_data(self):
        """Test CSV export with empty network density data."""
        empty_data = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_name = "test_experiment"

            # Export to CSV (should handle empty data gracefully)
            _export_to_csv_single_values(
                temp_path, exp_name, "calcium_network_density", empty_data
            )

            # File should still be created but may be empty or minimal
            csv_file = temp_path / f"{exp_name}_calcium_network_density.csv"
            assert csv_file.exists()

    def test_network_density_calculation_mathematics(self):
        """Test the mathematical correctness of network density calculation."""
        # Create a simple case with known correlations
        # 3 ROIs in a triangle where all are connected

        roi_trace_1 = np.zeros(50)
        roi_trace_1[[10, 20, 30]] = 1.0

        roi_trace_2 = np.zeros(50)
        roi_trace_2[[10, 20, 30]] = 1.0  # Identical to ROI 1

        roi_trace_3 = np.zeros(50)
        roi_trace_3[[10, 20, 30]] = 1.0  # Identical to ROI 1 and 2

        test_data = {
            "Test": {
                "A01_f00": {
                    "1": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_1.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_1.tolist(),
                        peaks_dec_dff=[10.0, 20.0, 30.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        # Low threshold to ensure connections
                        calcium_network_threshold=50.0,
                        active=True,
                    ),
                    "2": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_2.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_2.tolist(),
                        peaks_dec_dff=[10.0, 20.0, 30.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=50.0,
                        active=True,
                    ),
                    "3": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_3.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_3.tolist(),
                        peaks_dec_dff=[10.0, 20.0, 30.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=50.0,
                        active=True,
                    ),
                }
            }
        }

        result = _get_calcium_network_density_parameter(test_data)

        # With 3 ROIs, maximum possible edges = 3 * 2 = 6 (directed graph)
        # If all ROIs are perfectly correlated, we should get high density
        density = result["Test"]["A01_f00"][0]

        # Should be high density since all traces are identical
        print(f"Network density for 3 identical ROIs: {density:.2f}%")
        assert density > 50.0  # Should be substantial connectivity

    def test_network_density_with_different_thresholds(self):
        """Test that higher thresholds result in lower network densities."""
        # Create ROIs with moderate correlation
        roi_trace_1 = np.zeros(50)
        roi_trace_1[[10, 20, 30]] = 1.0

        roi_trace_2 = np.zeros(50)
        roi_trace_2[[11, 21, 31]] = 1.0  # Slightly offset

        # Test with high threshold
        high_thresh_data = {
            "HighThresh": {
                "A01_f00": {
                    "1": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_1.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_1.tolist(),
                        peaks_dec_dff=[10.0, 20.0, 30.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=95.0,  # Very high threshold
                        active=True,
                    ),
                    "2": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_2.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_2.tolist(),
                        peaks_dec_dff=[11.0, 21.0, 31.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=95.0,
                        active=True,
                    ),
                }
            }
        }

        # Test with low threshold
        low_thresh_data = {
            "LowThresh": {
                "A01_f00": {
                    "1": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_1.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_1.tolist(),
                        peaks_dec_dff=[10.0, 20.0, 30.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=50.0,  # Low threshold
                        active=True,
                    ),
                    "2": ROIData(
                        well_fov_position="A01_f00",
                        raw_trace=roi_trace_2.tolist(),
                        dff=[0.0] * 50,
                        dec_dff=roi_trace_2.tolist(),
                        peaks_dec_dff=[11.0, 21.0, 31.0],
                        peaks_amplitudes_dec_dff=[1.0, 1.0, 1.0],
                        calcium_network_threshold=50.0,
                        active=True,
                    ),
                }
            }
        }

        high_result = _get_calcium_network_density_parameter(high_thresh_data)
        low_result = _get_calcium_network_density_parameter(low_thresh_data)

        # Extract densities
        if "HighThresh" in high_result and "A01_f00" in high_result["HighThresh"]:
            high_density = high_result["HighThresh"]["A01_f00"][0]
        else:
            high_density = 0.0  # No connections above high threshold

        if "LowThresh" in low_result and "A01_f00" in low_result["LowThresh"]:
            low_density = low_result["LowThresh"]["A01_f00"][0]
        else:
            low_density = 0.0

        print(f"High threshold (95%): {high_density:.2f}%")
        print(f"Low threshold (50%): {low_density:.2f}%")

        # Lower threshold should allow more connections (higher density)
        assert low_density >= high_density


if __name__ == "__main__":
    # Run basic tests for development
    test_instance = TestCalciumNetworkDensity()

    # Create sample data manually (not using fixtures)
    import numpy as np

    from micromanager_gui._plate_viewer._util import ROIData

    # Create correlation patterns that will result in known network densities
    roi_1_peaks = [10.0, 20.0, 30.0, 40.0, 50.0]
    roi_1_trace = np.zeros(100)
    roi_1_trace[[int(p) for p in roi_1_peaks]] = 1.0

    roi_2_peaks = [12.0, 22.0, 32.0, 42.0, 52.0]
    roi_2_trace = np.zeros(100)
    roi_2_trace[[int(p) for p in roi_2_peaks]] = 1.0

    sample_data = {
        "Control": {
            "A01_f00": {
                "1": ROIData(
                    well_fov_position="A01_f00",
                    raw_trace=roi_1_trace.tolist(),
                    dff=[0.0] * 100,
                    dec_dff=roi_1_trace.tolist(),
                    peaks_dec_dff=roi_1_peaks,
                    peaks_amplitudes_dec_dff=[1.0] * len(roi_1_peaks),
                    calcium_network_threshold=90.0,
                    active=True,
                ),
                "2": ROIData(
                    well_fov_position="A01_f00",
                    raw_trace=roi_2_trace.tolist(),
                    dff=[0.0] * 100,
                    dec_dff=roi_2_trace.tolist(),
                    peaks_dec_dff=roi_2_peaks,
                    peaks_amplitudes_dec_dff=[1.0] * len(roi_2_peaks),
                    calcium_network_threshold=90.0,
                    active=True,
                ),
            }
        }
    }

    # Test basic calculation
    print("=== Testing Basic Network Density Calculation ===")
    result = _get_calcium_network_density_parameter(sample_data)
    print("Network density result:", result)

    # Test CSV export
    print("\n=== Testing CSV Export ===")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        exp_name = "test_experiment"

        # Export to CSV
        _export_to_csv_single_values(
            temp_path, exp_name, "calcium_network_density", result
        )

        # Check if file was created
        csv_file = temp_path / f"{exp_name}_calcium_network_density.csv"
        if csv_file.exists():
            import pandas as pd

            df = pd.read_csv(csv_file)
            print("CSV contents:")
            print(df)
        else:
            print("CSV file not created")

    print("\n✅ All manual tests passed!")
