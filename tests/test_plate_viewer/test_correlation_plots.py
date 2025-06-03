"""Tests for correlation plotting methods."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots import (
    _get_correlation_matrix,
    _plot_correlation_data,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestCorrelationPlots:
    """Test correlation plotting functions."""

    @pytest.fixture
    def mock_widget(self):
        """Create a mock widget for testing."""
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.colorbar = Mock()
        widget.figure.tight_layout = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    @pytest.fixture
    def sample_roi_data(self) -> dict[str, ROIData]:
        """Create sample ROI data with traces."""
        return {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[1.0, 2.0, 3.0, 4.0, 5.0]
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[2.0, 3.0, 4.0, 5.0, 6.0]
            ),
            "3": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[3.0, 4.0, 5.0, 6.0, 7.0]
            ),
        }

    def test_get_correlation_matrix_valid_data(self, sample_roi_data):
        """Test correlation matrix calculation with valid data."""
        result = _get_correlation_matrix(sample_roi_data, rois=[1, 2, 3])
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        # Check that diagonal elements are 1 (self-correlation)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_get_correlation_matrix_no_rois(self, sample_roi_data):
        """Test correlation matrix with no ROIs specified."""
        result = _get_correlation_matrix(sample_roi_data, rois=None)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    def test_get_correlation_matrix_insufficient_rois(self, sample_roi_data):
        """Test correlation matrix with insufficient ROIs."""
        result = _get_correlation_matrix(sample_roi_data, rois=[1])
        
        assert result is None

    def test_get_correlation_matrix_no_dff_data(self):
        """Test correlation matrix with no dFF data."""
        roi_data = {
            "1": ROIData(well_fov_position="B5_0000_p0"),
            "2": ROIData(well_fov_position="B5_0000_p0"),
        }
        
        result = _get_correlation_matrix(roi_data, rois=[1, 2])
        
        assert result is None

    def test_get_correlation_matrix_mixed_data(self):
        """Test correlation matrix with mixed data availability."""
        roi_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[1.0, 2.0, 3.0]
            ),
            "2": ROIData(well_fov_position="B5_0000_p0"),  # No dFF data
            "3": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[4.0, 5.0, 6.0]
            ),
        }
        
        result = _get_correlation_matrix(roi_data, rois=[1, 2, 3])
        
        # Should return None or handle gracefully since ROI 2 has no data
        # Implementation may vary, but should not crash
        assert result is None or isinstance(result, np.ndarray)

    def test_get_correlation_matrix_different_lengths(self):
        """Test correlation matrix with different trace lengths."""
        roi_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[1.0, 2.0, 3.0]
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[4.0, 5.0, 6.0, 7.0, 8.0]  # Different length
            ),
        }
        
        result = _get_correlation_matrix(roi_data, rois=[1, 2])
        
        # Should handle gracefully - may truncate or return None
        assert result is None or isinstance(result, np.ndarray)

    @patch('micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots._get_correlation_matrix')
    def test_plot_correlation_data_success(self, mock_get_matrix, mock_widget, sample_roi_data):
        """Test successful correlation plotting."""
        # Mock the correlation matrix
        mock_matrix = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        mock_get_matrix.return_value = mock_matrix
        
        # Mock the subplot
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax
        
        _plot_correlation_data(mock_widget, sample_roi_data, rois=[1, 2, 3])
        
        # Verify the plot was created
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_ax.imshow.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    @patch('micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots._get_correlation_matrix')
    def test_plot_correlation_data_no_matrix(self, mock_get_matrix, mock_widget, sample_roi_data):
        """Test handling when correlation matrix cannot be calculated."""
        mock_get_matrix.return_value = None
        
        result = _plot_correlation_data(mock_widget, sample_roi_data, rois=[1, 2])
        
        assert result is None
        mock_widget.figure.clear.assert_called_once()

    def test_plot_correlation_data_title_content(self, mock_widget, sample_roi_data):
        """Test that the plot title contains expected content."""
        with patch('micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots._get_correlation_matrix') as mock_matrix:
            mock_matrix.return_value = np.eye(3)
            
            mock_ax = Mock()
            mock_widget.figure.add_subplot.return_value = mock_ax
            
            _plot_correlation_data(mock_widget, sample_roi_data, rois=[1, 2, 3])
            
            # Check that title is set
            mock_ax.set_title.assert_called_once()
            title_call = mock_ax.set_title.call_args[0][0]
            assert "Correlation" in title_call

    def test_plot_correlation_data_colorbar(self, mock_widget, sample_roi_data):
        """Test that colorbar is properly configured."""
        with patch('micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots._get_correlation_matrix') as mock_matrix:
            mock_matrix.return_value = np.eye(3)
            
            mock_ax = Mock()
            mock_widget.figure.add_subplot.return_value = mock_ax
            
            _plot_correlation_data(mock_widget, sample_roi_data, rois=[1, 2, 3])
            
            # Verify colorbar was created
            mock_widget.figure.colorbar.assert_called_once()

    def test_plot_correlation_data_axis_labels(self, mock_widget, sample_roi_data):
        """Test that axis labels are set correctly."""
        with patch('micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots._get_correlation_matrix') as mock_matrix:
            mock_matrix.return_value = np.eye(3)
            
            mock_ax = Mock()
            mock_widget.figure.add_subplot.return_value = mock_ax
            
            _plot_correlation_data(mock_widget, sample_roi_data, rois=[1, 2, 3])
            
            # Verify axis labels
            mock_ax.set_xlabel.assert_called_once_with("ROI")
            mock_ax.set_ylabel.assert_called_once_with("ROI")

    def test_plot_correlation_data_colormap_range(self, mock_widget, sample_roi_data):
        """Test that colormap range is appropriate for correlation data."""
        with patch('micromanager_gui._plate_viewer._plot_methods._single_wells_plots._correlation_plots._get_correlation_matrix') as mock_matrix:
            mock_matrix.return_value = np.array([[1.0, -0.5, 0.8], [-0.5, 1.0, 0.3], [0.8, 0.3, 1.0]])
            
            mock_ax = Mock()
            mock_widget.figure.add_subplot.return_value = mock_ax
            
            _plot_correlation_data(mock_widget, sample_roi_data, rois=[1, 2, 3])
            
            # Check that imshow was called with appropriate parameters
            mock_ax.imshow.assert_called_once()
            call_args = mock_ax.imshow.call_args
            # Should include vmin and vmax for correlation range
            assert 'vmin' in call_args[1] or 'vmax' in call_args[1]

    def test_get_correlation_matrix_edge_cases(self):
        """Test edge cases for correlation matrix calculation."""
        # Test with constant traces (zero variance)
        roi_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[1.0, 1.0, 1.0, 1.0, 1.0]  # Constant
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[2.0, 2.0, 2.0, 2.0, 2.0]  # Constant
            ),
        }
        
        result = _get_correlation_matrix(roi_data, rois=[1, 2])
        
        # Should handle constant traces gracefully
        # Implementation may return NaN or handle differently
        assert result is None or isinstance(result, np.ndarray)

    def test_get_correlation_matrix_empty_traces(self):
        """Test correlation matrix with empty traces."""
        roi_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[]
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                dff=[]
            ),
        }
        
        result = _get_correlation_matrix(roi_data, rois=[1, 2])
        
        assert result is None
