"""Tests for IEI plotting functionality."""

from unittest.mock import Mock

import numpy as np
import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_iei_data import (  # noqa: E501
    _add_hover_functionality,
    _plot_iei_data,
    _plot_metrics,
    _set_graph_title_and_labels,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestIEIPlots:
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
        ax.errorbar = Mock()
        ax.plot = Mock()
        ax.set_title = Mock()
        ax.set_ylabel = Mock()
        ax.set_xlabel = Mock()
        ax.set_xticks = Mock()
        ax.set_xticklabels = Mock()
        return ax

    @pytest.fixture
    def sample_roi_data_with_iei(self):
        """Create sample ROI data with IEI values."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                iei=[1.5, 2.0, 1.8, 2.2, 1.9],
            ),
            "2": ROIData(
                well_fov_position="A2",
                iei=[1.0, 1.5, 1.2, 1.8],
            ),
            "3": ROIData(
                well_fov_position="A3",
                iei=[2.5, 3.0, 2.8],
            ),
        }

    @pytest.fixture
    def roi_data_no_iei(self):
        """Create ROI data without IEI values."""
        return {
            "1": ROIData(
                well_fov_position="A1",
                iei=None,
            ),
            "2": ROIData(
                well_fov_position="A2",
                iei=[],
            ),
        }

    def test_plot_iei_data_basic(self, mock_widget, sample_roi_data_with_iei):
        """Test basic IEI data plotting."""
        _plot_iei_data(
            widget=mock_widget,
            data=sample_roi_data_with_iei,
            rois=None,
            std=False,
            sem=False,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_iei_data_with_std(self, mock_widget, sample_roi_data_with_iei):
        """Test IEI data plotting with standard deviation."""
        _plot_iei_data(
            widget=mock_widget,
            data=sample_roi_data_with_iei,
            rois=None,
            std=True,
            sem=False,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_iei_data_with_sem(self, mock_widget, sample_roi_data_with_iei):
        """Test IEI data plotting with standard error of the mean."""
        _plot_iei_data(
            widget=mock_widget,
            data=sample_roi_data_with_iei,
            rois=None,
            std=False,
            sem=True,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_iei_data_with_roi_filter(self, mock_widget, sample_roi_data_with_iei):
        """Test IEI data plotting with specific ROI filter."""
        _plot_iei_data(
            widget=mock_widget,
            data=sample_roi_data_with_iei,
            rois=[1, 3],
            std=False,
            sem=False,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_iei_data_no_iei(self, mock_widget, roi_data_no_iei):
        """Test IEI data plotting with ROIs that have no IEI data."""
        _plot_iei_data(
            widget=mock_widget,
            data=roi_data_no_iei,
            rois=None,
            std=False,
            sem=False,
        )

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_metrics_basic_plot(self, mock_ax):
        """Test basic plotting without error bars."""
        roi_data = ROIData(
            well_fov_position="A1",
            iei=[1.5, 2.0, 1.8, 2.2, 1.9],
        )

        _plot_metrics(mock_ax, "1", roi_data, std=False, sem=False)

        mock_ax.plot.assert_called_once()
        args, kwargs = mock_ax.plot.call_args
        assert args[0] == [1, 1, 1, 1, 1]  # x-coordinates
        assert args[1] == [1.5, 2.0, 1.8, 2.2, 1.9]  # y-coordinates
        assert kwargs["label"] == "ROI 1"

    def test_plot_metrics_with_std(self, mock_ax):
        """Test plotting with standard deviation error bars."""
        roi_data = ROIData(
            well_fov_position="A1",
            iei=[1.5, 2.0, 1.8, 2.2, 1.9],
        )

        _plot_metrics(mock_ax, "1", roi_data, std=True, sem=False)

        mock_ax.errorbar.assert_called_once()
        args, kwargs = mock_ax.errorbar.call_args

        expected_mean = np.mean([1.5, 2.0, 1.8, 2.2, 1.9])
        expected_std = np.std([1.5, 2.0, 1.8, 2.2, 1.9])

        assert args[0] == [1]
        assert args[1] == expected_mean
        assert kwargs["yerr"] == expected_std
        assert kwargs["fmt"] == "o"
        assert kwargs["label"] == "ROI 1"

    def test_plot_metrics_with_sem(self, mock_ax):
        """Test plotting with standard error of the mean error bars."""
        roi_data = ROIData(
            well_fov_position="A1",
            iei=[1.5, 2.0, 1.8, 2.2, 1.9],
        )

        _plot_metrics(mock_ax, "1", roi_data, std=False, sem=True)

        mock_ax.errorbar.assert_called_once()
        args, kwargs = mock_ax.errorbar.call_args

        expected_mean = np.mean([1.5, 2.0, 1.8, 2.2, 1.9])
        expected_sem = expected_mean / np.sqrt(len([1.5, 2.0, 1.8, 2.2, 1.9]))

        assert args[0] == [1]
        assert args[1] == expected_mean
        assert kwargs["yerr"] == expected_sem
        assert kwargs["fmt"] == "o"
        assert kwargs["label"] == "ROI 1"

    def test_plot_metrics_no_iei(self, mock_ax):
        """Test plotting when ROI has no IEI data."""
        roi_data = ROIData(
            well_fov_position="A1",
            iei=None,
        )

        _plot_metrics(mock_ax, "1", roi_data, std=False, sem=False)

        mock_ax.plot.assert_not_called()
        mock_ax.errorbar.assert_not_called()

    def test_plot_metrics_empty_iei(self, mock_ax):
        """Test plotting when ROI has empty IEI list."""
        roi_data = ROIData(
            well_fov_position="A1",
            iei=[],
        )

        _plot_metrics(mock_ax, "1", roi_data, std=False, sem=False)

        mock_ax.plot.assert_not_called()
        mock_ax.errorbar.assert_not_called()

    def test_set_graph_title_and_labels_basic(self, mock_ax):
        """Test setting title and labels for basic plot."""
        _set_graph_title_and_labels(mock_ax, std=False, sem=False)

        mock_ax.set_title.assert_called_once_with(
            "Inter-event intervals (Sec - Deconvolved ΔF/F)"
        )
        mock_ax.set_ylabel.assert_called_once_with("Inter-event intervals (Sec)")
        mock_ax.set_xlabel.assert_called_once_with("ROIs")

    def test_set_graph_title_and_labels_with_std(self, mock_ax):
        """Test setting title and labels for std plot."""
        _set_graph_title_and_labels(mock_ax, std=True, sem=False)

        mock_ax.set_title.assert_called_once_with(
            "Inter-event intervals (Sec - Mean ± StD - Deconvolved ΔF/F)"
        )

    def test_set_graph_title_and_labels_with_sem(self, mock_ax):
        """Test setting title and labels for sem plot."""
        _set_graph_title_and_labels(mock_ax, std=False, sem=True)

        mock_ax.set_title.assert_called_once_with(
            "Inter-event intervals (Sec - Mean ± SEM - Deconvolved ΔF/F)"
        )

    def test_add_hover_functionality_basic(self, mock_ax, mock_widget):
        """Test adding hover functionality to the plot (basic test)."""
        # This just tests that the function can be called without errors
        try:
            _add_hover_functionality(mock_ax, mock_widget)
        except ImportError:
            # mplcursors not available, skip
            pytest.skip("mplcursors not available")

    def test_edge_cases_single_iei_value(self, mock_ax):
        """Test behavior with single IEI value."""
        roi_data = ROIData(
            well_fov_position="A1",
            iei=[2.5],
        )

        _plot_metrics(mock_ax, "1", roi_data, std=True, sem=False)
        args, kwargs = mock_ax.errorbar.call_args
        assert args[1] == 2.5  # Mean should be the single value
        assert kwargs["yerr"] == 0.0  # Std of single value is 0

    def test_integration_full_workflow(self, mock_widget):
        """Test the complete workflow from start to finish."""
        test_data = {
            "1": ROIData(
                well_fov_position="A1",
                iei=[1.5, 2.0, 1.8, 2.2],
            ),
            "2": ROIData(
                well_fov_position="A2",
                iei=None,
            ),
            "3": ROIData(
                well_fov_position="A3",
                iei=[3.0, 3.5, 2.8],
            ),
        }

        # Test with std
        _plot_iei_data(
            widget=mock_widget,
            data=test_data,
            rois=[1, 3],
            std=True,
            sem=False,
        )

        mock_widget.figure.clear.assert_called()
        mock_widget.figure.add_subplot.assert_called_with(111)
        mock_widget.figure.tight_layout.assert_called()
        mock_widget.canvas.draw.assert_called()
