"""Tests for cell size plotting functionality."""

from unittest.mock import Mock, patch

import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_cell_size import (  # noqa: E501
    _add_hover_functionality,
    _plot_cell_size_data,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestCellSizePlots:
    """Test cell size plotting functions."""

    @pytest.fixture
    def mock_widget(self):
        """Create a mock widget with all necessary matplotlib components."""
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.tight_layout = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget.roiSelected = Mock()
        widget.roiSelected.emit = Mock()
        return widget

    @pytest.fixture
    def mock_ax(self):
        """Create a mock matplotlib axis."""
        ax = Mock()
        ax.scatter = Mock()
        ax.set_title = Mock()
        ax.set_ylabel = Mock()
        ax.set_xlabel = Mock()
        ax.set_xticks = Mock()
        ax.set_xticklabels = Mock()
        ax.get_ylabel = Mock(return_value="Cell Size (μm²)")
        return ax

    @pytest.fixture
    def sample_roi_data_with_cell_size(self):
        """Create sample ROI data with cell size values."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                cell_size=25.5,
                cell_size_units="μm²",
            ),
            "2": ROIData(
                well_fov_position="A2",
                cell_size=30.2,
                cell_size_units="μm²",
            ),
            "3": ROIData(
                well_fov_position="A3",
                cell_size=22.8,
                cell_size_units="μm²",
            ),
        }

    @pytest.fixture
    def roi_data_no_cell_size(self):
        """Create ROI data without cell size values."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                cell_size=None,
                cell_size_units=None,
            ),
            "2": ROIData(
                well_fov_position="A2",
                cell_size=0,
                cell_size_units="μm²",
            ),
        }

    @pytest.fixture
    def mixed_roi_data(self):
        """Create ROI data with mixed cell size availability."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                cell_size=25.5,
                cell_size_units="μm²",
            ),
            "2": ROIData(
                well_fov_position="A2",
                cell_size=None,
                cell_size_units=None,
            ),
            "3": ROIData(
                well_fov_position="A3",
                cell_size=30.2,
                cell_size_units="μm²",
            ),
        }

    @pytest.fixture
    def roi_data_different_units(self):
        """Create ROI data with different units."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                cell_size=25.5,
                cell_size_units="px²",
            ),
            "2": ROIData(
                well_fov_position="A2",
                cell_size=30.2,
                cell_size_units="μm²",
            ),
        }

    @pytest.fixture
    def roi_data_no_units(self):
        """Create ROI data without units."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                cell_size=25.5,
                cell_size_units=None,
            ),
            "2": ROIData(
                well_fov_position="A2",
                cell_size=30.2,
                cell_size_units="",
            ),
        }

    def test_plot_cell_size_data_basic(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test basic cell size data plotting."""
        _plot_cell_size_data(
            widget=mock_widget,
            data=sample_roi_data_with_cell_size,
            rois=None,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_cell_size_data_with_roi_filter(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test cell size plotting with ROI filter."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=sample_roi_data_with_cell_size,
            rois=[1, 3],  # Only plot ROIs 1 and 3
        )

        # Should have been called twice (for ROIs 1 and 3)
        assert mock_ax.scatter.call_count == 2

        # Check that the correct ROI numbers were used
        call_args = mock_ax.scatter.call_args_list
        # First positional argument is the x value (ROI number)
        roi_values = [call[0][0] for call in call_args]
        assert 1 in roi_values
        assert 3 in roi_values

    def test_plot_cell_size_data_empty_data(self, mock_widget):
        """Test plotting with empty data."""
        _plot_cell_size_data(
            widget=mock_widget,
            data={},
            rois=None,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_cell_size_data_no_cell_sizes(
        self, mock_widget, roi_data_no_cell_size
    ):
        """Test plotting with data that has no cell sizes."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=roi_data_no_cell_size,
            rois=None,
        )

        # Should not call scatter since no ROIs have cell size data
        mock_ax.scatter.assert_not_called()

    def test_plot_cell_size_data_mixed_availability(self, mock_widget, mixed_roi_data):
        """Test plotting with mixed cell size data availability."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=mixed_roi_data,
            rois=None,
        )

        # Should have been called twice (for ROIs 1 and 3, not 2)
        assert mock_ax.scatter.call_count == 2

    def test_plot_cell_size_data_units_handling(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test that units are correctly handled in the y-axis label."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=sample_roi_data_with_cell_size,
            rois=None,
        )

        # Check that the y-axis label includes the units
        mock_ax.set_ylabel.assert_called_with("Cell Size (μm²)")

    def test_plot_cell_size_data_no_units(self, mock_widget, roi_data_no_units):
        """Test plotting when no units are available."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=roi_data_no_units,
            rois=None,
        )

        # Check that the y-axis label has empty units
        mock_ax.set_ylabel.assert_called_with("Cell Size ()")

    def test_plot_cell_size_data_first_unit_used(
        self, mock_widget, roi_data_different_units
    ):
        """Test that the first encountered unit is used for the y-axis label."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=roi_data_different_units,
            rois=None,
        )

        # Should use the first unit encountered (px²)
        mock_ax.set_ylabel.assert_called_with("Cell Size (px²)")

    def test_plot_cell_size_data_axis_properties(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test that axis properties are set correctly."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=sample_roi_data_with_cell_size,
            rois=None,
        )

        # Check axis labels and title
        mock_ax.set_xlabel.assert_called_with("ROI")
        mock_ax.set_ylabel.assert_called_with("Cell Size (μm²)")
        mock_ax.set_title.assert_called_with("Cell Size per ROI")

        # Check that x-axis ticks are cleared
        mock_ax.set_xticks.assert_called_with([])
        mock_ax.set_xticklabels.assert_called_with([])

    def test_plot_cell_size_data_roi_labels(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test that ROI labels are correctly set for scatter points."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=sample_roi_data_with_cell_size,
            rois=None,
        )

        # Check that scatter was called with correct labels
        call_args = mock_ax.scatter.call_args_list
        # Extract 'label' keyword argument
        labels = [call[1]["label"] for call in call_args]
        expected_labels = ["ROI 1", "ROI 2", "ROI 3"]
        assert all(label in expected_labels for label in labels)

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_cell_size.mplcursors.cursor"
    )
    def test_add_hover_functionality_called(self, mock_cursor, mock_ax, mock_widget):
        """Test that hover functionality is properly initialized."""
        mock_cursor_instance = Mock()
        mock_cursor.return_value = mock_cursor_instance

        _add_hover_functionality(mock_ax, mock_widget)

        # Check that mplcursors.cursor was called with correct parameters
        mock_cursor.assert_called_once()
        # The cursor should be called with the axis and hover mode
        args, kwargs = mock_cursor.call_args
        assert mock_ax in args

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_cell_size.mplcursors.cursor"
    )
    def test_add_hover_functionality_cursor_connect(
        self, mock_cursor, mock_ax, mock_widget
    ):
        """Test that cursor connect is called for hover functionality."""
        mock_cursor_instance = Mock()
        mock_cursor.return_value = mock_cursor_instance

        _add_hover_functionality(mock_ax, mock_widget)

        # Check that connect was called on the cursor instance
        mock_cursor_instance.connect.assert_called_once_with("add")

    def test_plot_cell_size_data_with_hover(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test that hover functionality is added during plotting."""
        with patch(
            "micromanager_gui._plate_viewer._plot_methods."
            "_single_wells_plots._plot_cell_size._add_hover_functionality"
        ) as mock_hover:
            _plot_cell_size_data(
                widget=mock_widget,
                data=sample_roi_data_with_cell_size,
                rois=None,
            )

            # Check that hover functionality was added
            mock_hover.assert_called_once()

    def test_plot_cell_size_data_roi_out_of_range(
        self, mock_widget, sample_roi_data_with_cell_size
    ):
        """Test plotting with ROI filter that includes non-existent ROIs."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=sample_roi_data_with_cell_size,
            rois=[1, 5, 7],  # ROIs 5 and 7 don't exist in the data
        )

        # Should only plot ROI 1 (the only one that exists)
        assert mock_ax.scatter.call_count == 1

    def test_plot_cell_size_data_zero_cell_size(self, mock_widget):
        """Test plotting with zero cell size values."""
        data_with_zero = {
            "1": ROIData(
                well_fov_position="A1",
                cell_size=0.0,
                cell_size_units="μm²",
            ),
            "2": ROIData(
                well_fov_position="A2",
                cell_size=25.5,
                cell_size_units="μm²",
            ),
        }

        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cell_size_data(
            widget=mock_widget,
            data=data_with_zero,
            rois=None,
        )

        # Should only plot ROI 2 (ROI 1 has zero cell size)
        assert mock_ax.scatter.call_count == 1
