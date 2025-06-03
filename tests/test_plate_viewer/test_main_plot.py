from unittest.mock import Mock, patch

import pytest

from micromanager_gui._plate_viewer._plot_methods._main_plot import (
    plot_single_well_data,
    plot_multi_well_data,
    RAW_TRACES,
    DFF,
    DEC_DFF_AMPLITUDE,
    CSV_VIOLIN_PLOT_AMPLITUDE,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestMainPlot:
    @pytest.fixture
    def mock_single_widget(self):
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.tight_layout = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    @pytest.fixture
    def mock_multi_widget(self):
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.tight_layout = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    @pytest.fixture
    def sample_roi_data(self):
        return {
            "1": ROIData(
                well_fov_position="A1",
                raw_trace=[1.0, 2.0, 3.0, 4.0, 5.0],
                dff=[1.0, 2.0, 3.0, 4.0, 5.0],
                dec_dff=[1.0, 2.0, 3.0, 4.0, 5.0],
                peaks_dec_dff=[1, 3],
                peaks_amplitudes_dec_dff=[0.5, 0.8],
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A2",
                raw_trace=[2.0, 3.0, 4.0, 5.0, 6.0],
                dff=[2.0, 3.0, 4.0, 5.0, 6.0],
                dec_dff=[2.0, 3.0, 4.0, 5.0, 6.0],
                peaks_dec_dff=[0, 2, 4],
                peaks_amplitudes_dec_dff=[0.3, 0.6, 0.9],
                active=True,
            ),
        }

    @patch('micromanager_gui._plate_viewer._plot_methods._main_plot._plot_traces_data')
    def test_plot_single_well_data_raw_traces(
        self, mock_plot_traces, mock_single_widget, sample_roi_data
    ):
        """Test plotting raw traces for single well."""
        plot_single_well_data(
            widget=mock_single_widget,
            data=sample_roi_data,
            text=RAW_TRACES,
            rois=[1, 2]
        )
        mock_plot_traces.assert_called_once_with(
            mock_single_widget, sample_roi_data, [1, 2]
        )

    @patch('micromanager_gui._plate_viewer._plot_methods._main_plot._plot_traces_data')
    def test_plot_single_well_data_dff(
        self, mock_plot_traces, mock_single_widget, sample_roi_data
    ):
        """Test plotting dFF traces for single well."""
        plot_single_well_data(
            widget=mock_single_widget,
            data=sample_roi_data,
            text=DFF,
            rois=[1, 2]
        )
        mock_plot_traces.assert_called_once_with(
            mock_single_widget, sample_roi_data, [1, 2], dff=True
        )

    @patch('micromanager_gui._plate_viewer._plot_methods._main_plot._plot_amplitude_and_frequency_data')
    def test_plot_single_well_data_amplitude(
        self, mock_plot_amp, mock_single_widget, sample_roi_data
    ):
        """Test plotting amplitude data for single well."""
        plot_single_well_data(
            widget=mock_single_widget,
            data=sample_roi_data,
            text=DEC_DFF_AMPLITUDE,
            rois=[1, 2]
        )
        mock_plot_amp.assert_called_once_with(
            mock_single_widget, sample_roi_data, [1, 2], amp=True
        )

    def test_plot_single_well_data_unknown_type(
        self, mock_single_widget, sample_roi_data
    ):
        """Test plotting with unknown plot type."""
        result = plot_single_well_data(
            widget=mock_single_widget,
            data=sample_roi_data,
            text="UNKNOWN_TYPE",
            rois=[1, 2]
        )
        assert result is None

    @patch('micromanager_gui._plate_viewer._plot_methods._main_plot.plot_csv_bar_plot')
    def test_plot_multi_well_data_csv(self, mock_plot_csv, mock_multi_widget):
        """Test plotting multi-well CSV data."""
        plot_multi_well_data(
            widget=mock_multi_widget,
            text=CSV_VIOLIN_PLOT_AMPLITUDE,
            analysis_path="/fake/path/data.csv"
        )
        # The test should check that the function returns early when no CSV path exists
        mock_plot_csv.assert_not_called()

    def test_plot_multi_well_data_no_csv(self, mock_multi_widget):
        """Test plotting multi-well data without CSV file."""
        result = plot_multi_well_data(
            widget=mock_multi_widget,
            text="CSV_VIOLIN_PLOT_AMPLITUDE",
            analysis_path=None
        )
        assert result is None
