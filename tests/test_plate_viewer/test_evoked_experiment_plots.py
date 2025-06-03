from unittest.mock import Mock

import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plolt_evoked_evperiment_data_plots import (  # noqa: E501
    _plot_evoked_experiment_data,
    _plot_stim_or_not_stim_peaks_amplitude,
    extract_leading_number,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestEvokedExperimentPlots:
    @pytest.fixture
    def mock_widget(self):
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.tight_layout = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget._plate_viewer = Mock()
        widget._plate_viewer.pv_analysis_path = None
        return widget

    @pytest.fixture
    def sample_roi_data(self):
        return {
            "1": ROIData(
                well_fov_position="A1",
                dff=[1.0, 2.0, 3.0],
                dec_dff=[1.0, 2.0, 3.0],
                peaks_dec_dff=[0, 2],
                active=True,
                stimulated=False,
                total_recording_time_in_sec=10.0,
                stmulations_frames_and_powers={"100": 5},
            ),
            "2": ROIData(
                well_fov_position="A2",
                dff=[2.0, 3.0, 4.0],
                dec_dff=[2.0, 3.0, 4.0],
                peaks_dec_dff=[1],
                active=True,
                stimulated=True,
                total_recording_time_in_sec=10.0,
                stmulations_frames_and_powers={"100": 5},
            ),
        }

    def test_plot_evoked_experiment_data_no_stimulated_area(
        self, mock_widget, sample_roi_data
    ):
        """Test plotting evoked experiment data without stimulated area."""
        _plot_evoked_experiment_data(
            widget=mock_widget,
            data=sample_roi_data,
            rois=[1, 2],
            stimulated_area=False,
            with_rois=False,
            stimulated=False,
            with_peaks=False,
        )
        # Should call the basic plotting function
        mock_widget.figure.clear.assert_called()

    def test_plot_evoked_experiment_data_with_peaks(self, mock_widget, sample_roi_data):
        """Test plotting evoked experiment data with peaks."""
        _plot_evoked_experiment_data(
            widget=mock_widget,
            data=sample_roi_data,
            rois=[1, 2],
            stimulated_area=False,
            with_rois=False,
            with_peaks=True,
        )
        # Should call plotting functions
        mock_widget.figure.clear.assert_called()

    def test_plot_stim_or_not_stim_peaks_amplitude(self, mock_widget, sample_roi_data):
        """Test plotting stimulated vs non-stimulated peaks amplitude."""
        # Set up analysis path to avoid early return
        mock_widget._plate_viewer.pv_analysis_path = "/fake/path"

        # Create data with required amplitude data
        test_data = {
            "1": ROIData(
                well_fov_position="A1",
                active=True,
                amplitudes_stimulated_peaks={"10_100": [0.5, 0.6]},
                amplitudes_non_stimulated_peaks={"10_100": [0.3, 0.4]},
            )
        }

        _plot_stim_or_not_stim_peaks_amplitude(
            widget=mock_widget,
            data=test_data,
            rois=[1],
            stimulated=True,
            std=False,
            sem=False,
        )
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_extract_leading_number(self):
        """Test extracting leading numbers from strings."""
        assert extract_leading_number("123.45abc") == 123.45
        assert extract_leading_number("0.5test") == 0.5
        # Test case where no number is found - should raise ValueError
        with pytest.raises(ValueError):
            extract_leading_number("no_number")
        assert extract_leading_number("999") == 999.0
