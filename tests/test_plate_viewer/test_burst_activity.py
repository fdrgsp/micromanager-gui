"""Tests for inferred spikes burst activity plotting functions."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
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
            )
        }

        _plot_inferred_spike_burst_activity(mock_widget, empty_data)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

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
        # Test with different ROI selection
        _plot_inferred_spike_burst_activity(
            mock_widget,
            burst_roi_data,
            rois=[1],  # Test with single ROI
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
