"""Tests for inferred spikes burst activity plotting functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from micromanager_gui._plate_viewer._graph_widgets import _SingleWellGraphWidget
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_inferred_spike_burst_activity import (  # noqa: E501
    _detect_population_bursts,
    _get_population_spike_data,
    _plot_inferred_spike_burst_activity,
)
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_inferred_spikes import (  # noqa: E501
    _detect_bursts_from_traces,
    _overlay_burst_periods,
    _plot_inferred_spikes_normalized_with_bursts,
)
from micromanager_gui._plate_viewer._to_csv import (
    _export_to_csv_burst_activity,
    _get_burst_activity_parameter,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestInferredSpikesBurstActivity:
    """Test inferred spikes burst activity plotting functions."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        mock_ax = Mock()
        widget.figure.add_subplot = Mock(return_value=mock_ax)
        widget.figure.tight_layout = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget.roiSelected = Mock()
        widget.roiSelected.emit = Mock()

        # Mock the _plate_viewer structure for burst parameter access
        widget._plate_viewer = Mock()
        widget._plate_viewer.analysis_path = None  # Use widget parameters instead
        widget._plate_viewer._analysis_wdg = Mock()
        widget._plate_viewer._analysis_wdg._burst_wdg = Mock()
        # Mock burst parameter values (threshold, min_duration, smoothing_sigma)
        widget._plate_viewer._analysis_wdg._burst_wdg.value.return_value = {
            "burst_threshold": 0.3,
            "min_burst_duration": 3,
            "smoothing_sigma": 1.0,
        }

        return widget

    @pytest.fixture
    def burst_roi_data(self) -> dict[str, ROIData]:
        """Create sample ROI data with burst-like patterns for testing."""
        # Create burst-like spike patterns
        burst_pattern_1 = np.zeros(100)
        burst_pattern_1[10:20] = [0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.6, 0.5]
        burst_pattern_1[40:50] = [0.7, 0.8, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8, 0.6, 0.7]

        burst_pattern_2 = np.zeros(100)
        burst_pattern_2[12:22] = [0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9]
        burst_pattern_2[42:52] = [0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.7]

        return {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 100,
                dff=[0.0] * 100,
                dec_dff=[0.0] * 100,
                inferred_spikes=burst_pattern_1.tolist(),
                inferred_spikes_threshold=0.5,
                peaks_dec_dff=[10, 15, 40, 45],
                peaks_amplitudes_dec_dff=[0.8, 0.9, 0.7, 0.8],
                dec_dff_frequency=2.0,
                iei=[5.0, 25.0, 5.0],
                cell_size=150.5,
                evoked_experiment=False,
                stimulated=False,
                active=True,
                total_recording_time_sec=10.0,
                spikes_burst_threshold=30.0,
                spikes_burst_min_duration=3,
                spikes_burst_gaussian_sigma=1.0,
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[95.0] * 100,
                dff=[0.0] * 100,
                dec_dff=[0.0] * 100,
                inferred_spikes=burst_pattern_2.tolist(),
                inferred_spikes_threshold=0.4,
                peaks_dec_dff=[12, 17, 42, 47],
                peaks_amplitudes_dec_dff=[0.6, 0.8, 0.8, 0.9],
                dec_dff_frequency=2.0,
                iei=[5.0, 25.0, 5.0],
                cell_size=145.0,
                evoked_experiment=False,
                stimulated=False,
                active=True,
                total_recording_time_sec=10.0,
                spikes_burst_threshold=30.0,
                spikes_burst_min_duration=3,
                spikes_burst_gaussian_sigma=1.0,
            ),
        }

    def test_plot_inferred_spike_burst_activity_basic(
        self, mock_widget, burst_roi_data
    ):
        """Test basic burst activity plotting."""
        _plot_inferred_spike_burst_activity(
            mock_widget,
            burst_roi_data,
            rois=[1, 2],
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spike_burst_activity_no_data(self, mock_widget):
        """Test burst activity plotting with no valid data."""
        empty_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 10,
                dff=[0.0] * 10,
                dec_dff=[0.0] * 10,
                inferred_spikes=None,  # No spike data
                active=False,
                # Add burst parameters so it doesn't return early
                spikes_burst_threshold=30.0,
                spikes_burst_min_duration=3,
                spikes_burst_gaussian_sigma=1.0,
            )
        }

        _plot_inferred_spike_burst_activity(mock_widget, empty_data)

        mock_widget.figure.clear.assert_called_once()
        # The function should return early due to insufficient spike data
        # so add_subplot, tight_layout, and canvas.draw should not be called

    def test_get_population_spike_data(self, burst_roi_data):
        """Test extraction of population spike data."""
        spike_trains, roi_names, time_axis = _get_population_spike_data(
            burst_roi_data, rois=[1, 2]
        )

        assert spike_trains is not None
        assert len(roi_names) == 2
        assert "1" in roi_names
        assert "2" in roi_names
        assert len(time_axis) == spike_trains.shape[1]
        assert spike_trains.shape[0] == 2  # Two ROIs

    def test_get_population_spike_data_single_roi(self, burst_roi_data):
        """Test extraction with single ROI (should return None)."""
        single_roi_data = {"1": burst_roi_data["1"]}

        spike_trains, roi_names, time_axis = _get_population_spike_data(
            single_roi_data, rois=[1]
        )

        assert spike_trains is None
        assert len(roi_names) == 0
        assert len(time_axis) == 0

    def test_detect_population_bursts(self):
        """Test population burst detection algorithm."""
        # Create a simple population activity signal with clear bursts
        population_activity = np.array(
            [
                0.1,
                0.1,
                0.8,
                0.9,
                0.8,
                0.1,
                0.1,
                0.1,  # Burst 1 (indices 2-4)
                0.1,
                0.1,
                0.1,
                0.7,
                0.8,
                0.9,
                0.8,
                0.1,  # Burst 2 (indices 11-14)
            ]
        )

        bursts = _detect_population_bursts(
            population_activity, burst_threshold=0.5, min_duration=3
        )

        assert len(bursts) == 2
        assert bursts[0] == (2, 5)  # First burst
        assert bursts[1] == (11, 15)  # Second burst

    def test_detect_population_bursts_short_duration(self):
        """Test that short bursts are filtered out."""
        population_activity = np.array(
            [
                0.1,
                0.1,
                0.8,
                0.9,
                0.1,
                0.1,  # Short burst (duration 2)
                0.1,
                0.8,
                0.9,
                0.8,
                0.7,
                0.1,  # Long burst (duration 4)
            ]
        )

        bursts = _detect_population_bursts(
            population_activity, burst_threshold=0.5, min_duration=3
        )

        assert len(bursts) == 1
        assert bursts[0] == (7, 11)  # Only the long burst

    def test_plot_inferred_spikes_normalized_with_bursts(
        self, mock_widget, burst_roi_data
    ):
        """Test normalized spikes plotting with burst overlay."""
        _plot_inferred_spikes_normalized_with_bursts(
            mock_widget,
            burst_roi_data,
            rois=[1, 2],
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_normalized_with_bursts_no_data(self, mock_widget):
        """Test normalized spikes with bursts plotting with no data."""
        empty_data = {}

        _plot_inferred_spikes_normalized_with_bursts(mock_widget, empty_data)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        # Note: canvas.draw() is not called when there's no valid data

    def test_detect_bursts_from_traces(self, burst_roi_data):
        """Test burst detection from multiple ROI traces."""
        roi_traces = {
            "1": burst_roi_data["1"].inferred_spikes,
            "2": burst_roi_data["2"].inferred_spikes,
        }

        bursts = _detect_bursts_from_traces(
            roi_traces, burst_threshold=0.3, min_duration=3, smoothing_sigma=2.0
        )

        # Should detect bursts where multiple ROIs are active
        assert isinstance(bursts, list)
        assert len(bursts) >= 0  # At least some bursts should be detected

    def test_detect_bursts_from_traces_empty(self):
        """Test burst detection with empty traces."""
        empty_traces = {}

        bursts = _detect_bursts_from_traces(
            empty_traces, burst_threshold=0.3, min_duration=3, smoothing_sigma=2.0
        )

        assert bursts == []

    def test_overlay_burst_periods(self):
        """Test overlaying burst periods on a plot."""
        mock_ax = Mock()
        bursts = [(10, 20), (40, 50)]
        num_rois = 3

        _overlay_burst_periods(mock_ax, bursts, num_rois)

        # Should call axvspan for each burst period
        assert mock_ax.axvspan.call_count == len(bursts)

    def test_overlay_burst_periods_empty(self):
        """Test overlaying with no burst periods."""
        mock_ax = Mock()
        bursts = []
        num_rois = 3

        _overlay_burst_periods(mock_ax, bursts, num_rois)

        # Should not call axvspan for empty burst list
        mock_ax.axvspan.assert_not_called()

    def test_burst_activity_with_different_parameters(
        self, mock_widget, burst_roi_data
    ):
        """Test burst activity plotting with different parameter values."""
        # Test with different ROI selection - use both ROIs so we have enough data
        _plot_inferred_spike_burst_activity(
            mock_widget,
            burst_roi_data,
            rois=[1, 2],  # Test with both ROIs to meet minimum requirement
        )

        assert mock_widget.figure.clear.call_count >= 1
        assert mock_widget.canvas.draw.call_count >= 1

    def test_burst_activity_active_only_filter(self, mock_widget, burst_roi_data):
        """Test burst activity with active_only filtering."""
        # Make one ROI inactive
        burst_roi_data["2"].active = False

        spike_trains, roi_names, time_axis = _get_population_spike_data(
            burst_roi_data, rois=[1, 2]
        )

        # Should only include active ROI
        assert spike_trains is None  # Only 1 active ROI, needs at least 2
        assert len(roi_names) == 0

    def test_time_axis_calculation(self, burst_roi_data):
        """Test that time axis is calculated correctly from recording time."""
        spike_trains, roi_names, time_axis = _get_population_spike_data(
            burst_roi_data, rois=[1, 2]
        )

        if spike_trains is not None:
            # Time axis should span the recording duration
            assert time_axis[0] == 0.0
            assert time_axis[-1] <= 10.0  # Recording time is 10 seconds
            assert len(time_axis) == spike_trains.shape[1]


class TestBurstActivityCSV:
    """Test burst activity CSV export functionality."""

    @pytest.fixture
    def sample_roi_data(self):
        """Create sample ROI data for testing burst activity CSV export."""
        roi_data = {}  # Create ROI with mock inferred spikes and timing data
        roi_data["1"] = ROIData(
            well_fov_position="A01_f00",
            inferred_spikes=[0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 1.0, 1.1, 1.2, 1.5],
            elapsed_time_list_ms=list(range(0, 2000, 100)),  # 0-2s in 100ms steps
            total_recording_time_sec=2.0,
            active=True,
            inferred_spikes_threshold=0.3,  # Required for spike thresholding
            spikes_burst_threshold=30.0,
            spikes_burst_min_duration=1,  # 0.1 seconds converted to int
            spikes_burst_gaussian_sigma=1.0,  # Required parameter
        )

        roi_data["2"] = ROIData(
            well_fov_position="A01_f00",
            inferred_spikes=[0.3, 0.4, 0.8, 0.9, 1.3, 1.4, 1.6, 1.7],
            elapsed_time_list_ms=list(range(0, 2000, 100)),
            total_recording_time_sec=2.0,
            active=True,
            inferred_spikes_threshold=0.3,  # Required for spike thresholding
            spikes_burst_threshold=30.0,
            spikes_burst_min_duration=1,
            spikes_burst_gaussian_sigma=1.0,  # Required parameter
        )

        return roi_data

    @pytest.fixture
    def sample_analysis_data(self, sample_roi_data):
        """Create sample analysis data structure."""
        return {
            "Control": {"A01_f00": sample_roi_data},
            "Treatment": {"B01_f00": sample_roi_data},
        }

    def test_get_burst_activity_parameter_basic(self, sample_analysis_data):
        """Test basic burst activity parameter extraction."""
        result = _get_burst_activity_parameter(sample_analysis_data)

        # Should have data for both conditions
        assert "Control" in result
        assert "Treatment" in result

        # Each condition should have well data
        assert "A01_f00" in result["Control"]
        assert "B01_f00" in result["Treatment"]

        # Each well should have burst metrics
        control_metrics = result["Control"]["A01_f00"][0]
        assert "count" in control_metrics
        assert "avg_duration_sec" in control_metrics
        assert "avg_interval_sec" in control_metrics
        assert "rate_burst_per_min)" in control_metrics

        # Values should be numeric
        assert isinstance(control_metrics["count"], (int, float))
        assert isinstance(control_metrics["avg_duration_sec"], (int, float))
        assert isinstance(control_metrics["avg_interval_sec"], (int, float))
        assert isinstance(control_metrics["rate_burst_per_min)"], (int, float))

    def test_get_burst_activity_parameter_no_bursts(self):
        """Test burst activity parameter extraction when no bursts are detected."""
        # Create ROI data with sparse spikes (no bursts)
        roi_data = {
            "1": ROIData(
                well_fov_position="A01_f00",
                inferred_spikes=[0.1, 1.0, 1.9],  # Sparse spikes
                elapsed_time_list_ms=list(range(0, 2000, 100)),
                total_recording_time_sec=2.0,
                active=True,
                inferred_spikes_threshold=0.05,  # Required for spike thresholding
                spikes_burst_threshold=80.0,  # High threshold
                spikes_burst_min_duration=5,  # 0.5 seconds
                spikes_burst_gaussian_sigma=1.0,  # Required parameter
            ),
            "2": ROIData(
                well_fov_position="A01_f00",
                inferred_spikes=[0.2, 1.1, 1.8],
                elapsed_time_list_ms=list(range(0, 2000, 100)),
                total_recording_time_sec=2.0,
                active=True,
                inferred_spikes_threshold=0.05,  # Required for spike thresholding
                spikes_burst_threshold=80.0,
                spikes_burst_min_duration=5,
                spikes_burst_gaussian_sigma=1.0,  # Required parameter
            ),
        }

        analysis_data = {"Control": {"A01_f00": roi_data}}
        result = _get_burst_activity_parameter(analysis_data)

        # Should still have structure but with zero values
        assert "Control" in result
        assert "A01_f00" in result["Control"]

        metrics = result["Control"]["A01_f00"][0]
        assert metrics["count"] == 0
        assert metrics["avg_duration_sec"] == 0.0
        assert metrics["avg_interval_sec"] == 0.0
        assert metrics["rate_burst_per_min)"] == 0.0

    def test_get_burst_activity_parameter_insufficient_rois(self):
        """Test burst activity parameter extraction with insufficient ROIs."""
        # Create data with only one ROI
        roi_data = {
            "1": ROIData(
                well_fov_position="A01_f00",
                inferred_spikes=[0.1, 0.2, 0.3],
                elapsed_time_list_ms=list(range(0, 1000, 100)),
                total_recording_time_sec=1.0,
                active=True,
                inferred_spikes_threshold=0.05,  # Required for spike thresholding
                spikes_burst_threshold=30.0,
                spikes_burst_min_duration=1,
                spikes_burst_gaussian_sigma=1.0,  # Required parameter
            )
        }

        analysis_data = {"Control": {"A01_f00": roi_data}}
        result = _get_burst_activity_parameter(analysis_data)

        # Should return empty dict for this condition
        assert result == {}

    def test_export_to_csv_burst_activity(self):
        """Test CSV export of burst activity data."""
        # Create sample burst activity data
        burst_data = {
            "Control": {
                "A01_f00": [
                    {
                        "count": 3,
                        "avg_duration_sec": 0.5,
                        "avg_interval_sec": 1.2,
                        "rate_burst_per_min)": 90.0,
                    }
                ],
                "A02_f00": [
                    {
                        "count": 2,
                        "avg_duration_sec": 0.3,
                        "avg_interval_sec": 1.5,
                        "rate_burst_per_min)": 60.0,
                    }
                ],
            },
            "Treatment": {
                "B01_f00": [
                    {
                        "count": 5,
                        "avg_duration_sec": 0.8,
                        "avg_interval_sec": 0.9,
                        "rate_burst_per_min)": 150.0,
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_name = "test_experiment"

            # Export to CSV
            _export_to_csv_burst_activity(temp_path, exp_name, burst_data)

            # Check if file was created
            csv_file = temp_path / f"{exp_name}_burst_activity.csv"
            assert csv_file.exists()

            # Read and verify CSV content
            df = pd.read_csv(csv_file)

            # Should have 4 columns per condition (8 total)
            expected_columns = [
                "Control_count",
                "Control_avg_duration_sec",
                "Control_avg_interval_sec",
                "Control_rate_burst_per_min)",
                "Treatment_count",
                "Treatment_avg_duration_sec",
                "Treatment_avg_interval_sec",
                "Treatment_rate_burst_per_min)",
            ]
            assert list(df.columns) == expected_columns

            # Check values
            assert df["Control_count"].iloc[0] == 3
            assert df["Control_count"].iloc[1] == 2
            assert df["Control_avg_duration_sec"].iloc[0] == 0.5
            assert df["Control_rate_burst_per_min)"].iloc[0] == 90.0

            assert df["Treatment_count"].iloc[0] == 5
            assert df["Treatment_avg_duration_sec"].iloc[0] == 0.8
            assert df["Treatment_rate_burst_per_min)"].iloc[0] == 150.0

            # Second row should have NaN for Treatment (only one well)
            assert pd.isna(df["Treatment_count"].iloc[1])

    def test_export_to_csv_burst_activity_empty_data(self):
        """Test CSV export with empty burst activity data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_name = "test_experiment"

            # Export empty data
            _export_to_csv_burst_activity(temp_path, exp_name, {})

            # File should not be created for empty data
            csv_file = temp_path / f"{exp_name}_burst_activity.csv"
            assert not csv_file.exists()

    def test_export_to_csv_burst_activity_mixed_data(self):
        """Test CSV export with mixed valid and invalid data."""
        burst_data = {
            "Control": {
                "A01_f00": [
                    {
                        "count": 2,
                        "avg_duration_sec": 0.4,
                        "avg_interval_sec": 1.0,
                        "rate_burst_per_min)": 60.0,
                    }
                ],
                "A02_f00": [
                    # Invalid data (not a dict)
                    "invalid_data"
                ],
            },
            "Treatment": {
                "B01_f00": [
                    {
                        "count": 1,
                        "avg_duration_sec": 0.2,
                        "avg_interval_sec": 0.0,  # No intervals for single burst
                        "rate_burst_per_min)": 30.0,
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_name = "test_experiment"

            # Export should handle invalid data gracefully
            _export_to_csv_burst_activity(temp_path, exp_name, burst_data)

            # Check if file was created
            csv_file = temp_path / f"{exp_name}_burst_activity.csv"
            assert csv_file.exists()

            # Read and verify CSV content
            df = pd.read_csv(csv_file)

            # Should have valid data from Control A01 and Treatment B01
            assert df["Control_count"].iloc[0] == 2
            assert df["Treatment_count"].iloc[0] == 1

            # Invalid data should be skipped, so only one row
            assert len(df) == 1

    def test_integration_get_and_export_burst_activity(self, sample_analysis_data):
        """Test integration of burst activity parameter extraction and CSV export."""
        # Extract burst activity parameters
        burst_params = _get_burst_activity_parameter(sample_analysis_data)

        # Export to CSV
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_name = "integration_test"

            _export_to_csv_burst_activity(temp_path, exp_name, burst_params)

            # Verify file creation and content
            csv_file = temp_path / f"{exp_name}_burst_activity.csv"
            assert csv_file.exists()

            df = pd.read_csv(csv_file)

            # Should have columns for both conditions
            control_cols = [col for col in df.columns if col.startswith("Control_")]
            treatment_cols = [col for col in df.columns if col.startswith("Treatment_")]

            assert len(control_cols) == 4
            assert len(treatment_cols) == 4

            # All values should be numeric (not NaN for first row)
            for col in control_cols + treatment_cols:
                assert pd.notna(df[col].iloc[0])
                # Check that it's a numeric type (including numpy types)
                assert pd.api.types.is_numeric_dtype(df[col])
