from unittest.mock import Mock

import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._raster_plots import (  # noqa: E501
    _generate_raster_plot,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestRasterPlots:
    @pytest.fixture
    def mock_widget(self):
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.tight_layout = Mock()
        widget.figure.colorbar = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    @pytest.fixture
    def sample_roi_data(self):
        return {
            "1": ROIData(
                well_fov_position="A1",
                raw_trace=[1.0, 2.0, 3.0, 4.0, 5.0],
                peaks_dec_dff=[1, 3],
                peaks_amplitudes_dec_dff=[0.5, 0.8],
                total_recording_time_in_sec=10.0,
            ),
            "2": ROIData(
                well_fov_position="A2",
                raw_trace=[2.0, 3.0, 4.0, 5.0, 6.0],
                peaks_dec_dff=[0, 2, 4],
                peaks_amplitudes_dec_dff=[0.3, 0.6, 0.9],
                total_recording_time_in_sec=10.0,
            ),
        }

    def test_generate_raster_plot_basic(self, mock_widget, sample_roi_data):
        """Test basic raster plot generation."""
        _generate_raster_plot(
            widget=mock_widget,
            data=sample_roi_data,
            rois=[1, 2],
            amplitude_colors=False,
            colorbar=False,
        )
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_widget.figure.tight_layout.assert_called_once()

    def test_generate_raster_plot_with_amplitude_colors(
        self, mock_widget, sample_roi_data
    ):
        """Test raster plot generation with amplitude colors."""
        _generate_raster_plot(
            widget=mock_widget,
            data=sample_roi_data,
            rois=[1, 2],
            amplitude_colors=True,
            colorbar=True,
        )
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_widget.figure.colorbar.assert_called_once()
        mock_widget.figure.tight_layout.assert_called_once()

    def test_generate_raster_plot_no_peaks(self, mock_widget):
        """Test raster plot with ROI data that has no peaks."""
        data_no_peaks = {
            "1": ROIData(
                well_fov_position="A1",
                raw_trace=[1.0, 2.0, 3.0],
                peaks_dec_dff=None,
                peaks_amplitudes_dec_dff=None,
            ),
        }
        _generate_raster_plot(
            widget=mock_widget,
            data=data_no_peaks,
            rois=[1],
            amplitude_colors=False,
            colorbar=False,
        )
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_widget.figure.tight_layout.assert_called_once()

    def test_generate_raster_plot_empty_peaks(self, mock_widget):
        """Test raster plot with ROI data that has empty peaks."""
        data_empty_peaks = {
            "1": ROIData(
                well_fov_position="A1",
                raw_trace=[1.0, 2.0, 3.0],
                peaks_dec_dff=[],
                peaks_amplitudes_dec_dff=[],
            ),
        }
        _generate_raster_plot(
            widget=mock_widget,
            data=data_empty_peaks,
            rois=[1],
            amplitude_colors=False,
            colorbar=False,
        )
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_widget.figure.tight_layout.assert_called_once()
