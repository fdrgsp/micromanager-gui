"""Tests for specific plot method functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from micromanager_gui._plate_viewer._graph_widgets import (
    _MultilWellGraphWidget,
    _SingleWellGraphWidget,
)
from micromanager_gui._plate_viewer._plot_methods._multi_wells_plots._csv_bar_plot import (  # noqa: E501
    _create_bar_plot_burst_activity,
    plot_csv_bar_plot,
)
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_traces_data import (  # noqa: E501
    _plot_traces_data,
)
from micromanager_gui._plate_viewer._util import ROIData

# Test data paths
TEST_DATA_SPONTANEOUS = Path(__file__).parent / "data" / "spontaneous"


class TestTracesPlotsSpecific:
    """Test specific trace plotting functions."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        mock_ax = Mock()
        widget.figure.add_subplot = Mock(return_value=mock_ax)
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    @pytest.fixture
    def sample_roi_data(self) -> dict[str, ROIData]:
        """Create sample ROI data for testing."""
        return {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0, 105.0, 110.0, 115.0, 120.0] * 20,
                dff=[0.0, 0.05, 0.10, 0.15, 0.20] * 20,
                dec_dff=[0.0, 0.04, 0.08, 0.12, 0.16] * 20,
                peaks_dec_dff=[10, 30, 50, 70, 90],
                active=True,
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[95.0, 100.0, 105.0, 110.0, 115.0] * 20,
                dff=[0.0, 0.053, 0.105, 0.158, 0.211] * 20,
                dec_dff=[0.0, 0.042, 0.084, 0.126, 0.168] * 20,
                peaks_dec_dff=[15, 35, 55, 75, 95],
                active=True,
            ),
        }

    def test_plot_raw_traces(self, mock_widget, sample_roi_data):
        """Test plotting raw traces specifically."""
        _plot_traces_data(mock_widget, sample_roi_data, dff=False, dec=False)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_dff_traces(self, mock_widget, sample_roi_data):
        """Test plotting delta F/F traces specifically."""
        _plot_traces_data(mock_widget, sample_roi_data, dff=True, dec=False)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_deconvolved_traces(self, mock_widget, sample_roi_data):
        """Test plotting deconvolved traces specifically."""
        _plot_traces_data(mock_widget, sample_roi_data, dff=False, dec=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_normalized_traces(self, mock_widget, sample_roi_data):
        """Test plotting normalized traces."""
        _plot_traces_data(
            mock_widget, sample_roi_data, dff=False, dec=False, normalize=True
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_with_peaks(self, mock_widget, sample_roi_data):
        """Test plotting traces with peaks overlay."""
        _plot_traces_data(
            mock_widget, sample_roi_data, dff=False, dec=True, with_peaks=True
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_active_only(self, mock_widget, sample_roi_data):
        """Test plotting only active ROI traces."""
        _plot_traces_data(
            mock_widget, sample_roi_data, dff=False, dec=False, active_only=True
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_with_roi_filter(self, mock_widget, sample_roi_data):
        """Test plotting traces with ROI filter."""
        _plot_traces_data(mock_widget, sample_roi_data, rois=[1])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_with_empty_data(self, mock_widget):
        """Test plotting traces with empty data."""
        _plot_traces_data(mock_widget, {})

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)


class TestCSVBarPlotSpecific:
    """Test specific CSV bar plot functions."""

    @pytest.fixture
    def mock_widget(self) -> _MultilWellGraphWidget:
        """Create a mock multi well graph widget."""
        widget = Mock(spec=_MultilWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        mock_ax = Mock()
        widget.figure.add_subplot = Mock(return_value=mock_ax)
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        # Initialize conditions as an empty dict that can be used by the bar plot
        widget.conditions = {}
        return widget

    @pytest.fixture
    def temp_csv_file(self, tmp_path) -> Path:
        """Create a temporary CSV file for testing."""
        csv_content = """FOV,c1_t1_Mean,c1_t1_SEM,c1_t1_N,c2_t2_Mean,c2_t2_SEM,c2_t2_N
B5_0000_p0,0.07943,0.01571,22,0.08234,0.01823,18
B6_0000_p0,0.06421,0.01234,25,0.07892,0.01654,20
"""
        csv_file = tmp_path / "test_amplitude.csv"
        csv_file.write_text(csv_content)
        return csv_file

    @pytest.fixture
    def temp_simple_csv_file(self, tmp_path) -> Path:
        """Create a simple CSV file for testing."""
        csv_content = """condition1,condition2
0.07943,0.08234
0.06421,0.07892
0.05123,0.06543
"""
        csv_file = tmp_path / "test_simple.csv"
        csv_file.write_text(csv_content)
        return csv_file

    def test_plot_csv_bar_plot_with_mean_sem(self, mock_widget, temp_csv_file):
        """Test CSV bar plot with mean and SEM data."""
        info = {
            "parameter": "Amplitude",
            "suffix": "amplitude",
            "add_to_title": " (Deconvolved ΔF/F)",
        }

        plot_csv_bar_plot(mock_widget, temp_csv_file, info, mean_n_sem=True)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_csv_bar_plot_simple(self, mock_widget, temp_simple_csv_file):
        """Test CSV bar plot with simple column data."""
        info = {
            "parameter": "Amplitude",
            "suffix": "amplitude",
            "add_to_title": " (Deconvolved ΔF/F)",
        }

        plot_csv_bar_plot(mock_widget, temp_simple_csv_file, info, mean_n_sem=False)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_csv_bar_plot_nonexistent_file(self, mock_widget):
        """Test CSV bar plot with nonexistent file."""
        info = {"parameter": "Amplitude"}

        # Should not raise an exception
        plot_csv_bar_plot(mock_widget, "/nonexistent/file.csv", info, mean_n_sem=True)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_csv_bar_plot_invalid_csv(self, mock_widget, tmp_path):
        """Test CSV bar plot with invalid CSV content."""
        invalid_csv = tmp_path / "invalid.csv"
        invalid_csv.write_text("not,a,valid,csv,structure\n1,2,3")

        info = {"parameter": "Amplitude"}

        # Should handle gracefully
        plot_csv_bar_plot(mock_widget, invalid_csv, info, mean_n_sem=True)

        mock_widget.figure.clear.assert_called_once()

    @pytest.fixture
    def temp_burst_csv_file(self, tmp_path) -> Path:
        """Create a temporary burst activity CSV file for testing."""
        csv_content = (
            "c1_t1_count,c1_t1_avg_duration_sec,c1_t1_avg_interval_sec,"
            "c1_t1_rate_burst_per_min,c2_t2_count,c2_t2_avg_duration_sec,"
            "c2_t2_avg_interval_sec,c2_t2_rate_burst_per_min\n"
            "5,2.5,15.3,0.8,3,3.2,20.1,0.6\n"
            "7,1.8,12.4,1.2,4,2.9,18.5,0.7\n"
            "6,2.1,14.7,0.9,5,3.5,22.3,0.5\n"
        )
        csv_file = tmp_path / "test_burst_activity.csv"
        csv_file.write_text(csv_content)
        return csv_file

    @pytest.mark.parametrize(
        "burst_metric,expected_conditions",
        [
            ("count", ["c1_t1", "c2_t2"]),
            ("avg_duration_sec", ["c1_t1", "c2_t2"]),
            ("avg_interval_sec", ["c1_t1", "c2_t2"]),
            ("rate_burst_per_min", ["c1_t1", "c2_t2"]),
        ],
    )
    def test_create_bar_plot_burst_activity(
        self, mock_widget, temp_burst_csv_file, burst_metric, expected_conditions
    ):
        """Test burst activity CSV parsing for different metrics."""
        info = {
            "parameter": "Burst Activity",
            "suffix": "burst_activity",
            "burst_metric": burst_metric,
            "add_to_title": "(Inferred Spikes)",
            "units": "Count",
        }

        # Mock necessary attributes
        mock_widget.conditions = {}

        # Call the function directly to test CSV parsing
        # Note: _create_bar_plot_burst_activity doesn't call figure.clear itself
        # That's handled by the main plot_csv_bar_plot function
        try:
            _create_bar_plot_burst_activity(
                mock_widget, temp_burst_csv_file, info, burst_metric
            )
            # If we reach here, the function executed without error
            test_passed = True
        except Exception as e:
            print(f"Function failed with error: {e}")
            test_passed = False

        # Verify the function executed successfully
        assert test_passed, f"Function failed for burst metric: {burst_metric}"

    def test_plot_csv_bar_plot_with_burst_metric(
        self, mock_widget, temp_burst_csv_file
    ):
        """Test plot_csv_bar_plot with burst_metric parameter."""
        info = {
            "parameter": "Burst Count",
            "suffix": "burst_activity",
            "burst_metric": "count",
            "add_to_title": "(Inferred Spikes)",
            "units": "Count",
        }

        plot_csv_bar_plot(mock_widget, temp_burst_csv_file, info, mean_n_sem=False)

        mock_widget.figure.clear.assert_called_once()


class TestPlotMethodsIntegration:
    """Integration tests using real test data if available."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        mock_ax = Mock()
        widget.figure.add_subplot = Mock(return_value=mock_ax)
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_plot_with_real_csv_data(self):
        """Test plotting with real CSV data if available."""
        csv_file = (
            TEST_DATA_SPONTANEOUS
            / "spont_analysis"
            / "grouped"
            / "spont_analysis_amplitude.csv"
        )

        if not csv_file.exists():
            pytest.skip("Real CSV test data not available")

        # Create mock widget
        mock_widget = Mock(spec=_MultilWellGraphWidget)
        mock_widget.figure = Mock(spec=Figure)
        mock_widget.figure.clear = Mock()
        mock_ax = Mock()
        mock_widget.figure.add_subplot = Mock(return_value=mock_ax)
        mock_widget.canvas = Mock()
        mock_widget.canvas.draw = Mock()
        # Initialize conditions as an empty dict that can be used by the bar plot
        mock_widget.conditions = {}

        info = {
            "parameter": "Amplitude",
            "suffix": "amplitude",
            "add_to_title": " (Deconvolved ΔF/F)",
        }

        # Should not raise an exception
        plot_csv_bar_plot(mock_widget, csv_file, info, mean_n_sem=True)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_traces_with_matplotlib_interaction(self, mock_widget):
        """Test that plot functions properly interact with matplotlib."""
        sample_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=list(np.random.random(100)),
                dff=list(np.random.random(100) * 0.1),
                dec_dff=list(np.random.random(100) * 0.08),
                active=True,
            )
        }

        with patch("matplotlib.pyplot.plot"):
            _plot_traces_data(mock_widget, sample_data, dff=False, dec=False)

            mock_widget.figure.clear.assert_called_once()
            mock_widget.figure.add_subplot.assert_called_once()


class TestPlotMethodsParameterValidation:
    """Test parameter validation and edge cases."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        mock_ax = Mock()
        widget.figure.add_subplot = Mock(return_value=mock_ax)
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_plot_traces_with_invalid_roi_ids(self, mock_widget):
        """Test plotting traces with invalid ROI IDs."""
        data_with_invalid_ids = {
            "not_a_number": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0, 105.0, 110.0],
                active=True,
            ),
            "123abc": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[95.0, 100.0, 105.0],
                active=True,
            ),
        }

        # Should handle gracefully
        _plot_traces_data(mock_widget, data_with_invalid_ids, rois=[1, 2])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_traces_with_mismatched_trace_lengths(self, mock_widget):
        """Test plotting traces with different lengths."""
        data_mismatched = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 50,  # 50 points
                dff=[0.1] * 100,  # 100 points
                dec_dff=[0.08] * 25,  # 25 points
                active=True,
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[95.0] * 75,  # 75 points
                dff=[0.12] * 50,  # 50 points
                dec_dff=[0.09] * 150,  # 150 points
                active=True,
            ),
        }

        # Should handle gracefully
        _plot_traces_data(mock_widget, data_mismatched, dff=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_traces_with_extreme_values(self, mock_widget):
        """Test plotting traces with extreme values."""
        data_extreme = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[1e10, 1e-10, float("inf"), -float("inf"), 0],
                dff=[1000, -1000, 0.001, -0.001, 0],
                active=True,
            )
        }

        # Should handle gracefully without crashing
        _plot_traces_data(mock_widget, data_extreme, dff=False)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
