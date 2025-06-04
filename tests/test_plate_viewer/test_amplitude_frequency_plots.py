from unittest.mock import Mock

import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_amplitudes_and_frequencies_data import (  # noqa: E501
    _plot_amplitude_and_frequency_data,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestAmplitudeFrequencyPlots:
    @pytest.fixture
    def mock_widget(self):
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
            "1": ROIData(well_fov_position="A1", dff=[1.0, 2.0, 3.0]),
            "2": ROIData(well_fov_position="A2", dff=[2.0, 3.0, 4.0]),
        }

    def test_plot_amplitude_and_frequency_data(self, mock_widget, sample_roi_data):
        _plot_amplitude_and_frequency_data(
            widget=mock_widget,
            data=sample_roi_data,
            rois=[1, 2],
            amp=True,
            freq=True,
        )
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()
