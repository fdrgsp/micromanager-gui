from unittest.mock import Mock

import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._spike_raster_plots import (  # noqa: E501
    _generate_spike_raster_plot,
    _plot_stimulated_vs_non_stimulated_spike_traces,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestSpikeRasterPlots:
    """Test class for spike raster plots functionality."""

    @pytest.fixture
    def mock_widget(self):
        """Create a mock widget for testing."""
        widget = Mock()
        widget.figure = Mock()
        widget.canvas = Mock()
        widget.roiSelected = Mock()

        # Mock the subplot
        mock_ax = Mock()
        widget.figure.add_subplot.return_value = mock_ax
        widget.figure.clear = Mock()
        widget.figure.tight_layout = Mock()
        widget.figure.colorbar = Mock()

        return widget

    @pytest.fixture
    def sample_spike_data(self):
        """Create sample spike data for testing."""
        # ROI 1 with spike data
        roi1 = ROIData(
            raw_trace=[1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 1.0, 3.0, 2.0, 1.0],
            inferred_spikes=[0.1, 0.8, 0.2, 0.1, 0.0, 1.2, 0.0, 0.9, 0.3, 0.0],
            inferred_spikes_threshold=0.5,
            total_recording_time_sec=10.0,
        )

        # ROI 2 with spike data
        roi2 = ROIData(
            raw_trace=[2.0, 1.0, 3.0, 4.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0],
            inferred_spikes=[0.2, 0.0, 0.7, 1.1, 0.3, 0.0, 0.8, 0.2, 1.3, 0.0],
            inferred_spikes_threshold=0.5,
            total_recording_time_sec=10.0,
        )

        return {"1": roi1, "2": roi2}

    def test_generate_spike_raster_plot_basic(self, mock_widget, sample_spike_data):
        """Test basic spike raster plot generation."""
        _generate_spike_raster_plot(
            mock_widget,
            sample_spike_data,
            rois=None,
            amplitude_colors=False,
            colorbar=False,
        )

        # Verify the widget methods were called
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_generate_spike_raster_plot_with_amplitude_colors(
        self, mock_widget, sample_spike_data
    ):
        """Test spike raster plot with amplitude-based coloring."""
        _generate_spike_raster_plot(
            mock_widget,
            sample_spike_data,
            rois=None,
            amplitude_colors=True,
            colorbar=True,
        )

        # Verify the widget methods were called
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()
        mock_widget.figure.colorbar.assert_called_once()

    def test_generate_spike_raster_plot_no_spikes(self, mock_widget):
        """Test spike raster plot when no spike data is available."""
        # Create data without spike information
        data = {"1": ROIData(raw_trace=[1.0, 2.0, 3.0])}

        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _generate_spike_raster_plot(
            mock_widget, data, rois=None, amplitude_colors=False, colorbar=False
        )

        # Verify that the function completes without displaying text
        mock_ax.text.assert_not_called()
        # Verify canvas is still updated
        mock_widget.canvas.draw.assert_called_once()

    def test_generate_spike_raster_plot_empty_spikes(self, mock_widget):
        """Test spike raster plot when no spikes are above threshold."""
        data = {
            "1": ROIData(
                raw_trace=[1.0, 2.0, 3.0, 2.0, 1.0],
                inferred_spikes=[0.1, 0.2, 0.1, 0.3, 0.2],  # All below threshold
                inferred_spikes_threshold=0.5,
                total_recording_time_sec=5.0,
            )
        }

        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _generate_spike_raster_plot(
            mock_widget, data, rois=None, amplitude_colors=False, colorbar=False
        )

        # Verify that the function completes without displaying text
        mock_ax.text.assert_not_called()
        # Verify canvas is still updated
        mock_widget.canvas.draw.assert_called_once()

    def test_generate_spike_raster_plot_with_roi_filter(
        self, mock_widget, sample_spike_data
    ):
        """Test spike raster plot with ROI filtering."""
        _generate_spike_raster_plot(
            mock_widget,
            sample_spike_data,
            rois=[1],  # Only ROI 1
            amplitude_colors=False,
            colorbar=False,
        )

        # Verify the widget methods were called
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_stimulated_vs_non_stimulated_spike_traces(self, mock_widget):
        """Test stimulated vs non-stimulated spike traces plot."""
        # Create data with stimulation timing
        data = {
            "1": ROIData(
                raw_trace=[1.0, 2.0, 3.0, 2.0, 1.0],
                inferred_spikes=[0.1, 0.8, 0.2, 0.1, 0.0],
                inferred_spikes_threshold=0.5,
                total_recording_time_sec=5.0,
                stimulations_frames_and_powers={"1": 50, "3": 80},
                stimulated=True,
                active=True,
            ),
            "2": ROIData(
                raw_trace=[2.0, 1.0, 3.0, 4.0, 2.0],
                inferred_spikes=[0.2, 0.0, 0.7, 1.1, 0.3],
                inferred_spikes_threshold=0.5,
                total_recording_time_sec=5.0,
                stimulations_frames_and_powers={"1": 50, "3": 80},
                stimulated=False,
                active=True,
            ),
        }

        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_stimulated_vs_non_stimulated_spike_traces(mock_widget, data, rois=None)

        # Verify the widget methods were called
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_stimulated_vs_non_stimulated_spike_traces_no_stimulation(
        self, mock_widget
    ):
        """Test stimulated vs non-stimulated spike traces plot with no stimulation."""
        data = {
            "1": ROIData(
                raw_trace=[1.0, 2.0, 3.0, 2.0, 1.0],
                inferred_spikes=[0.1, 0.8, 0.2, 0.1, 0.0],
                inferred_spikes_threshold=0.5,
                total_recording_time_sec=5.0,
                active=True,
                # No stimulations_frames_and_powers attribute
            )
        }

        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_stimulated_vs_non_stimulated_spike_traces(mock_widget, data, rois=None)

        # Should still work, just no green coloring
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()
