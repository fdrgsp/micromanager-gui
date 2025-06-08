from unittest.mock import Mock, patch

import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_synchrony import (  # noqa: E501
    _plot_synchrony_data,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestSynchronyPlots:
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
                instantaneous_phase=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            "2": ROIData(
                well_fov_position="A2",
                instantaneous_phase=[0.2, 0.3, 0.4, 0.5, 0.6],
            ),
        }

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._synchrony_plots._get_phase_dict_from_rois"
    )
    @patch("micromanager_gui._plate_viewer._util._get_synchrony_matrix")
    @patch("micromanager_gui._plate_viewer._util.get_synchrony")
    def test_plot_synchrony_data_success(
        self,
        mock_get_synchrony,
        mock_get_matrix,
        mock_get_phase,
        mock_widget,
        sample_roi_data,
    ):
        """Test successful synchrony plotting."""
        # Mock the dependencies
        mock_get_phase.return_value = {"1": [0.1, 0.2], "2": [0.2, 0.3]}
        mock_get_matrix.return_value = [[1.0, 0.8], [0.8, 1.0]]
        mock_get_synchrony.return_value = 0.9

        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_synchrony_data(mock_widget, sample_roi_data, rois=[1, 2])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_ax.imshow.assert_called_once()
        mock_widget.figure.colorbar.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._synchrony_plots._get_phase_dict_from_rois"
    )
    def test_plot_synchrony_data_no_phase(
        self, mock_get_phase, mock_widget, sample_roi_data
    ):
        """Test synchrony plotting when no phase data is available."""
        mock_get_phase.return_value = None

        result = _plot_synchrony_data(mock_widget, sample_roi_data, rois=[1, 2])

        assert result is None
        mock_widget.figure.clear.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._synchrony_plots._get_phase_dict_from_rois"
    )
    @patch("micromanager_gui._plate_viewer._util._get_synchrony_matrix")
    def test_plot_synchrony_data_no_matrix(
        self, mock_get_matrix, mock_get_phase, mock_widget, sample_roi_data
    ):
        """Test synchrony plotting when synchrony matrix cannot be calculated."""
        mock_get_phase.return_value = {"1": [0.1, 0.2]}
        mock_get_matrix.return_value = None

        result = _plot_synchrony_data(mock_widget, sample_roi_data, rois=[1])

        assert result is None
        mock_widget.figure.clear.assert_called_once()
