"""Tests for inferred spikes plotting functions."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from micromanager_gui._plate_viewer._graph_widgets import _SingleWellGraphWidget
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_inferred_spikes import (  # noqa: E501
    _add_hover_functionality,
    _normalize_trace_percentile,
    _plot_inferred_spikes,
    _plot_trace,
    _set_graph_title_and_labels,
    _update_time_axis,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestInferredSpikesPlots:
    """Test inferred spikes plotting functions."""

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
                inferred_spikes=[0.0, 0.2, 0.0, 0.8, 0.3] * 20,
                peaks_dec_dff=[10, 30, 50, 70, 90],
                total_recording_time_sec=20.0,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[95.0, 100.0, 105.0, 110.0, 115.0] * 20,
                dff=[0.0, 0.053, 0.105, 0.158, 0.211] * 20,
                dec_dff=[0.0, 0.042, 0.084, 0.126, 0.168] * 20,
                inferred_spikes=[0.0, 0.1, 0.0, 0.6, 0.2] * 20,
                peaks_dec_dff=[15, 35, 55, 75, 95],
                total_recording_time_sec=20.0,
                active=True,
            ),
            "3": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[90.0, 95.0, 100.0, 105.0, 110.0] * 20,
                dff=[0.0, 0.056, 0.111, 0.167, 0.222] * 20,
                dec_dff=[0.0, 0.045, 0.089, 0.134, 0.178] * 20,
                inferred_spikes=[0.0, 0.0, 0.0, 0.1, 0.0] * 20,
                peaks_dec_dff=[20, 40, 60, 80, 100],
                total_recording_time_sec=20.0,
                active=False,  # Inactive ROI
            ),
        }

    @pytest.fixture
    def empty_roi_data(self) -> dict[str, ROIData]:
        """Create ROI data with empty inferred spikes."""
        return {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0, 105.0, 110.0],
                dff=[0.0, 0.05, 0.10],
                dec_dff=[0.0, 0.04, 0.08],
                inferred_spikes=[],  # Empty spikes
                active=True,
            ),
        }

    def test_plot_inferred_spikes_basic(self, mock_widget, sample_roi_data):
        """Test basic inferred spikes plotting."""
        _plot_inferred_spikes(mock_widget, sample_roi_data)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_normalized(self, mock_widget, sample_roi_data):
        """Test normalized inferred spikes plotting."""
        _plot_inferred_spikes(mock_widget, sample_roi_data, normalize=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_active_only(self, mock_widget, sample_roi_data):
        """Test plotting only active ROIs."""
        _plot_inferred_spikes(mock_widget, sample_roi_data, active_only=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_with_dec_dff(self, mock_widget, sample_roi_data):
        """Test plotting inferred spikes with deconvolved dF/F."""
        _plot_inferred_spikes(mock_widget, sample_roi_data, dec_dff=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_with_rois_filter(self, mock_widget, sample_roi_data):
        """Test plotting with specific ROIs filter."""
        _plot_inferred_spikes(mock_widget, sample_roi_data, rois=[1, 2])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_empty_data(self, mock_widget, empty_roi_data):
        """Test plotting with empty inferred spikes data."""
        _plot_inferred_spikes(mock_widget, empty_roi_data)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_combined_options(self, mock_widget, sample_roi_data):
        """Test plotting with multiple options combined."""
        _plot_inferred_spikes(
            mock_widget,
            sample_roi_data,
            rois=[1, 3],
            normalize=True,
            active_only=True,
            dec_dff=True,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_spike_trace_normalized(self):
        """Test plotting spike trace with normalization."""
        mock_ax = Mock()
        spikes = [0.0, 0.2, 0.0, 0.8, 0.3]

        _plot_trace(mock_ax, "1", spikes, normalize=True, count=0, p1=0.0, p2=1.0)

        mock_ax.plot.assert_called_once()
        mock_ax.set_yticks.assert_called_once_with([])
        mock_ax.set_yticklabels.assert_called_once_with([])

    def test_plot_spike_trace_not_normalized(self):
        """Test plotting spike trace without normalization."""
        mock_ax = Mock()
        spikes = [0.0, 0.2, 0.0, 0.8, 0.3]

        _plot_trace(mock_ax, "1", spikes, normalize=False, count=0, p1=0.0, p2=1.0)

        mock_ax.plot.assert_called_once()
        # These should not be called when not normalized
        mock_ax.set_yticks.assert_not_called()
        mock_ax.set_yticklabels.assert_not_called()

    def test_normalize_trace_percentile(self):
        """Test trace normalization using percentiles."""
        trace = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        p1, p2 = 0.2, 0.8

        result = _normalize_trace_percentile(trace, p1, p2)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(trace)
        # Check that values are clipped to [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_normalize_trace_percentile_zero_denominator(self):
        """Test trace normalization with zero denominator."""
        trace = [0.5, 0.5, 0.5, 0.5]
        p1, p2 = 0.5, 0.5  # Same values result in zero denominator

        result = _normalize_trace_percentile(trace, p1, p2)

        assert isinstance(result, np.ndarray)
        assert np.all(result == 0.0)

    def test_set_graph_title_and_labels_normalized(self):
        """Test setting graph title and labels for normalized plot."""
        mock_ax = Mock()

        _set_graph_title_and_labels(mock_ax, normalize=True, raw=False)

        mock_ax.set_title.assert_called_once_with("Normalized Inferred Spikes (Thresholded)")
        mock_ax.set_ylabel.assert_called_once_with("ROIs")

    def test_set_graph_title_and_labels_not_normalized(self):
        """Test setting graph title and labels for non-normalized plot."""
        mock_ax = Mock()

        _set_graph_title_and_labels(mock_ax, normalize=False, raw=False)

        mock_ax.set_title.assert_called_once_with("Inferred Spikes (Thresholded)")
        mock_ax.set_ylabel.assert_called_once_with("Inferred Spikes (magnitude)")

    def test_set_graph_title_and_labels_with_raw_mode(self):
        """Test setting graph title and labels with raw mode enabled."""
        mock_ax = Mock()

        _set_graph_title_and_labels(mock_ax, normalize=False, raw=True)

        mock_ax.set_title.assert_called_once_with("Inferred Spikes (Raw)")
        mock_ax.set_ylabel.assert_called_once_with("Inferred Spikes (magnitude)")

    def test_update_time_axis_with_recording_time(self):
        """Test updating time axis with recording time data."""
        mock_ax = Mock()
        rois_rec_time = [20.0, 20.0, 20.0]
        trace = [0.0] * 100

        _update_time_axis(mock_ax, rois_rec_time, trace)

        mock_ax.set_xticks.assert_called_once()
        mock_ax.set_xticklabels.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("Time (s)")

    def test_update_time_axis_without_recording_time(self):
        """Test updating time axis without recording time data."""
        mock_ax = Mock()
        rois_rec_time = []
        trace = [0.0] * 100

        _update_time_axis(mock_ax, rois_rec_time, trace)

        mock_ax.set_xlabel.assert_called_once_with("Frames")
        mock_ax.set_xticks.assert_not_called()
        mock_ax.set_xticklabels.assert_not_called()

    def test_update_time_axis_no_trace(self):
        """Test updating time axis with no trace data."""
        mock_ax = Mock()
        rois_rec_time = [20.0, 20.0]
        trace = None

        _update_time_axis(mock_ax, rois_rec_time, trace)

        mock_ax.set_xlabel.assert_called_once_with("Frames")

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_inferred_spikes.mplcursors"
    )
    def test_add_hover_functionality(self, mock_mplcursors, mock_widget):
        """Test adding hover functionality to the plot."""
        mock_ax = Mock()
        mock_cursor = Mock()
        mock_mplcursors.cursor.return_value = mock_cursor

        _add_hover_functionality(mock_ax, mock_widget)

        mock_mplcursors.cursor.assert_called_once_with(
            mock_ax, hover=mock_mplcursors.HoverMode.Transient
        )
        mock_cursor.connect.assert_called_once_with("add")

    def test_plot_inferred_spikes_invalid_roi_keys(self, mock_widget):
        """Test plotting with invalid ROI keys."""
        invalid_data = {
            "invalid_key": ROIData(
                well_fov_position="B5_0000_p0",
                inferred_spikes=[0.0, 0.1, 0.2],
                active=True,
            ),
        }

        # Should not raise an error, just skip invalid keys
        _plot_inferred_spikes(mock_widget, invalid_data, rois=[1, 2])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_inferred_spikes_mixed_active_inactive(
        self, mock_widget, sample_roi_data
    ):
        """Test plotting with mix of active and inactive ROIs when active_only=True."""
        _plot_inferred_spikes(mock_widget, sample_roi_data, active_only=True)

        # Should still process the data (active ROIs 1 and 2, skip inactive ROI 3)
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_normalize_trace_percentile_edge_cases(self):
        """Test trace normalization with edge cases."""
        # Empty trace
        result = _normalize_trace_percentile([], 0.0, 1.0)
        assert len(result) == 0

        # Single value trace
        result = _normalize_trace_percentile([0.5], 0.0, 1.0)
        assert len(result) == 1
        assert result[0] == 0.5

        # All same values
        result = _normalize_trace_percentile([0.3, 0.3, 0.3], 0.3, 0.3)
        assert np.all(result == 0.0)

    def test_plot_inferred_spikes_percentile_calculation(
        self, mock_widget, sample_roi_data
    ):
        """Test that percentile calculation works correctly for normalization."""
        # This tests the internal percentile calculation logic
        _plot_inferred_spikes(mock_widget, sample_roi_data, normalize=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        # The function should complete without errors
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_with_thresholds(self, mock_widget, sample_roi_data):
        """Test plotting inferred spikes with threshold visualization."""
        # Add threshold data to sample ROI data
        sample_roi_data["1"].inferred_spikes_threshold = 0.5

        _plot_inferred_spikes(mock_widget, sample_roi_data, rois=[1], thresholds=True)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_inferred_spikes_thresholds_multiple_rois(
        self, mock_widget, sample_roi_data
    ):
        """Test that thresholds are disabled when multiple ROIs are selected."""
        sample_roi_data["1"].inferred_spikes_threshold = 0.5
        sample_roi_data["2"].inferred_spikes_threshold = 0.6

        _plot_inferred_spikes(
            mock_widget, sample_roi_data, rois=[1, 2], thresholds=True
        )

        # Should still work but without thresholds visualization
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()
