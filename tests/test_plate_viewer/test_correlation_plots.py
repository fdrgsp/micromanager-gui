"""Tests for correlation plots functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation import (  # noqa: E501
    _add_hover_functionality_clustering,
    _add_hover_functionality_cross_corr,
    _calculate_cross_correlation,
    _plot_cross_correlation_data,
    _plot_hierarchical_clustering_data,
    _plot_hierarchical_clustering_dendrogram,
    _plot_hierarchical_clustering_map,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestCorrelationPlots:
    """Test suite for correlation plot functionality."""

    @pytest.fixture
    def mock_widget(self):
        """Create a mock widget for testing."""
        widget = Mock()
        widget.figure = Mock()
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.figure.tight_layout = Mock()
        widget.figure.colorbar = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget.roiSelected = Mock()
        widget.roiSelected.emit = Mock()
        return widget

    @pytest.fixture
    def sample_roi_data_active(self):
        """Create sample ROI data with active traces for correlation testing."""
        # Create synthetic traces with some correlation patterns
        # ROI 1: sine wave
        trace1 = [np.sin(i * 0.1) for i in range(100)]
        # ROI 2: shifted sine wave (correlated)
        trace2 = [np.sin(i * 0.1 + 0.5) for i in range(100)]
        # ROI 3: cosine wave (different pattern)
        trace3 = [np.cos(i * 0.1) for i in range(100)]

        return {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                raw_trace=[100.0] * 100,
                dec_dff=trace1,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                raw_trace=[100.0] * 100,
                dec_dff=trace2,
                active=True,
            ),
            "3": ROIData(
                well_fov_position="A1_0000_p0",
                raw_trace=[100.0] * 100,
                dec_dff=trace3,
                active=True,
            ),
        }

    @pytest.fixture
    def sample_roi_data_inactive(self):
        """Create sample ROI data with inactive ROIs."""
        return {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                raw_trace=[100.0] * 50,
                dec_dff=[0.1] * 50,
                active=False,  # Inactive ROI
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                raw_trace=[100.0] * 50,
                dec_dff=None,  # No dec_dff data
                active=True,
            ),
        }

    @pytest.fixture
    def single_roi_data(self):
        """Create sample data with only one active ROI."""
        return {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                raw_trace=[100.0] * 50,
                dec_dff=[0.1] * 50,
                active=True,
            ),
        }

    def test_calculate_cross_correlation_basic(self, sample_roi_data_active):
        """Test basic cross-correlation calculation."""
        correlation_matrix, rois_idxs = _calculate_cross_correlation(
            sample_roi_data_active
        )

        assert correlation_matrix is not None
        assert rois_idxs is not None
        assert len(rois_idxs) == 3
        assert correlation_matrix.shape == (3, 3)

        # Check that diagonal elements are 1 (self-correlation)
        np.testing.assert_array_almost_equal(
            np.diag(correlation_matrix), [1.0, 1.0, 1.0], decimal=2
        )

        # Check that matrix is symmetric
        np.testing.assert_array_almost_equal(
            correlation_matrix, correlation_matrix.T, decimal=10
        )

    def test_calculate_cross_correlation_with_roi_filter(self, sample_roi_data_active):
        """Test cross-correlation calculation with specific ROI selection."""
        correlation_matrix, rois_idxs = _calculate_cross_correlation(
            sample_roi_data_active, rois=[1, 2]
        )

        assert correlation_matrix is not None
        assert rois_idxs is not None
        assert len(rois_idxs) == 2
        assert correlation_matrix.shape == (2, 2)
        assert set(rois_idxs) == {1, 2}

    def test_calculate_cross_correlation_inactive_rois(self, sample_roi_data_inactive):
        """Test cross-correlation with inactive ROIs."""
        correlation_matrix, rois_idxs = _calculate_cross_correlation(
            sample_roi_data_inactive
        )

        # Should return None for insufficient active ROIs
        assert correlation_matrix is None
        assert rois_idxs is None

    def test_calculate_cross_correlation_single_roi(self, single_roi_data):
        """Test cross-correlation with only one active ROI."""
        correlation_matrix, rois_idxs = _calculate_cross_correlation(single_roi_data)

        # Should return None for insufficient ROIs
        assert correlation_matrix is None
        assert rois_idxs is None

    def test_calculate_cross_correlation_empty_data(self):
        """Test cross-correlation with empty data."""
        correlation_matrix, rois_idxs = _calculate_cross_correlation({})

        assert correlation_matrix is None
        assert rois_idxs is None

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation._add_hover_functionality_cross_corr"
    )
    def test_plot_cross_correlation_data_success(
        self, mock_hover, mock_widget, sample_roi_data_active
    ):
        """Test successful cross-correlation plotting."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_cross_correlation_data(mock_widget, sample_roi_data_active)

        # Verify widget methods were called
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

        # Verify axis configuration
        mock_ax.set_title.assert_called_once_with(
            "Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)"
        )
        mock_ax.set_xlabel.assert_called_once_with("ROI")
        mock_ax.set_ylabel.assert_called_once_with("ROI")
        mock_ax.set_box_aspect.assert_called_once_with(1)

        # Verify imshow was called
        mock_ax.imshow.assert_called_once()

        # Verify hover functionality was added
        mock_hover.assert_called_once()

    def test_plot_cross_correlation_data_insufficient_rois(
        self, mock_widget, single_roi_data
    ):
        """Test cross-correlation plotting with insufficient ROIs."""
        _plot_cross_correlation_data(mock_widget, single_roi_data)

        # Should clear figure but not proceed with plotting
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

        # Should not call drawing methods since no data to plot
        mock_widget.canvas.draw.assert_not_called()

    def test_plot_cross_correlation_data_with_roi_filter(
        self, mock_widget, sample_roi_data_active
    ):
        """Test cross-correlation plotting with ROI filtering."""
        _plot_cross_correlation_data(mock_widget, sample_roi_data_active, rois=[1, 2])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    @patch("mplcursors.cursor")
    def test_add_hover_functionality_cross_corr(self, mock_cursor, mock_widget):
        """Test hover functionality for cross-correlation plots."""
        mock_image = Mock()
        rois = [1, 2, 3]
        values = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])

        # Mock cursor object
        mock_cursor_obj = Mock()
        mock_cursor.return_value = mock_cursor_obj

        _add_hover_functionality_cross_corr(mock_image, mock_widget, rois, values)

        # Verify cursor was created
        mock_cursor.assert_called_once()

        # Verify connect method was called
        mock_cursor_obj.connect.assert_called_once_with("add")

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation._plot_hierarchical_clustering_map"
    )
    def test_plot_hierarchical_clustering_data_map_mode(
        self, mock_plot_map, mock_widget, sample_roi_data_active
    ):
        """Test hierarchical clustering plotting in map mode."""
        _plot_hierarchical_clustering_data(
            mock_widget, sample_roi_data_active, use_dendrogram=False
        )

        mock_widget.figure.clear.assert_called_once()
        mock_plot_map.assert_called_once()
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation._plot_hierarchical_clustering_dendrogram"
    )
    def test_plot_hierarchical_clustering_data_dendrogram_mode(
        self, mock_plot_dendro, mock_widget, sample_roi_data_active
    ):
        """Test hierarchical clustering plotting in dendrogram mode."""
        _plot_hierarchical_clustering_data(
            mock_widget, sample_roi_data_active, use_dendrogram=True
        )

        mock_widget.figure.clear.assert_called_once()
        mock_plot_dendro.assert_called_once()
        mock_widget.figure.tight_layout.assert_called_once()
        mock_widget.canvas.draw.assert_called_once()

    def test_plot_hierarchical_clustering_data_insufficient_rois(
        self, mock_widget, single_roi_data
    ):
        """Test hierarchical clustering with insufficient ROIs."""
        _plot_hierarchical_clustering_data(mock_widget, single_roi_data)

        # Should clear figure but not proceed with plotting
        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

        # Should not call drawing methods since no data to plot
        mock_widget.canvas.draw.assert_not_called()

    def test_plot_hierarchical_clustering_dendrogram(self):
        """Test hierarchical clustering dendrogram plotting."""
        # For this test, we'll just verify that the function can be called
        # without errors when using actual matplotlib objects
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.add_subplot(111)
        correlation_matrix = np.array(
            [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]
        )
        rois_idxs = [1, 2, 3]

        # This should not raise an exception
        _plot_hierarchical_clustering_dendrogram(ax, correlation_matrix, rois_idxs)

        # Verify the basic axis properties were set
        expected_title = (
            "Pairwise Cross-Correlation - Hierarchical Clustering Dendrogram\n"
            "(Calcium Peaks Events)"
        )
        assert ax.get_title() == expected_title
        assert ax.get_ylabel() == "Distance"

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation._add_hover_functionality_clustering"
    )
    @patch("scipy.cluster.hierarchy.leaves_list")
    @patch("scipy.cluster.hierarchy.linkage")
    @patch("scipy.spatial.distance.squareform")
    def test_plot_hierarchical_clustering_map(
        self, mock_squareform, mock_linkage, mock_leaves_list, mock_hover, mock_widget
    ):
        """Test hierarchical clustering map plotting."""
        mock_ax = Mock()
        correlation_matrix = np.array(
            [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]
        )
        rois_idxs = [1, 2, 3]

        # Mock scipy functions
        mock_squareform.return_value = np.array([0.2, 0.4, 0.3])
        mock_linkage.return_value = np.array([[0, 1, 0.2, 2], [2, 3, 0.3, 3]])
        mock_leaves_list.return_value = [0, 2, 1]  # Reordering

        _plot_hierarchical_clustering_map(
            mock_widget, mock_ax, correlation_matrix, rois_idxs
        )

        # Verify axis configuration
        expected_title = (
            "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
            "(Calcium Peaks Events)"
        )
        mock_ax.set_title.assert_called_once_with(expected_title)
        mock_ax.set_ylabel.assert_called_once_with("ROI")

        # Verify imshow was called
        mock_ax.imshow.assert_called_once()

        # Verify hover functionality was added
        mock_hover.assert_called_once()

    @patch("mplcursors.cursor")
    def test_add_hover_functionality_clustering(self, mock_cursor, mock_widget):
        """Test hover functionality for clustering plots."""
        mock_image = Mock()
        rois = [1, 2, 3]
        order = [0, 2, 1]
        values = np.array([[1.0, 0.6, 0.8], [0.6, 1.0, 0.7], [0.8, 0.7, 1.0]])

        # Mock cursor object
        mock_cursor_obj = Mock()
        mock_cursor.return_value = mock_cursor_obj

        _add_hover_functionality_clustering(
            mock_image, mock_widget, rois, order, values
        )

        # Verify cursor was created
        mock_cursor.assert_called_once()

        # Verify connect method was called
        mock_cursor_obj.connect.assert_called_once_with("add")

    def test_cross_correlation_mathematical_properties(self, sample_roi_data_active):
        """Test mathematical properties of cross-correlation."""
        correlation_matrix, rois_idxs = _calculate_cross_correlation(
            sample_roi_data_active
        )

        assert correlation_matrix is not None

        # Test that correlation values are in valid range [-1, 1]
        assert np.all(correlation_matrix >= -1.0)
        assert np.all(correlation_matrix <= 1.1)  # Allow slight numerical tolerance

        # Test diagonal elements (self-correlation should be close to 1)
        diagonal = np.diag(correlation_matrix)
        np.testing.assert_array_almost_equal(
            diagonal, np.ones_like(diagonal), decimal=1
        )

    def test_roi_filtering_edge_cases(self, sample_roi_data_active):
        """Test edge cases for ROI filtering."""
        # Test with non-existent ROIs
        correlation_matrix, rois_idxs = _calculate_cross_correlation(
            sample_roi_data_active, rois=[999, 1000]
        )
        assert correlation_matrix is None
        assert rois_idxs is None

        # Test with mix of existing and non-existing ROIs
        correlation_matrix, rois_idxs = _calculate_cross_correlation(
            sample_roi_data_active, rois=[1, 999]
        )
        assert correlation_matrix is None  # Only one valid ROI
        assert rois_idxs is None

    def test_correlation_with_identical_traces(self):
        """Test correlation with identical traces."""
        identical_trace = [0.1, 0.2, 0.3, 0.4, 0.5] * 10

        data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=identical_trace,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=identical_trace,
                active=True,
            ),
        }

        correlation_matrix, rois_idxs = _calculate_cross_correlation(data)

        # Identical traces should have correlation close to 1
        assert correlation_matrix is not None
        np.testing.assert_array_almost_equal(
            correlation_matrix, [[1.0, 1.0], [1.0, 1.0]], decimal=2
        )

    def test_correlation_with_uncorrelated_traces(self):
        """Test correlation with uncorrelated/anti-correlated traces."""
        # Create uncorrelated traces
        trace1 = [1.0, 0.0, 1.0, 0.0, 1.0] * 10
        trace2 = [0.0, 1.0, 0.0, 1.0, 0.0] * 10  # Anti-correlated

        data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=trace1,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=trace2,
                active=True,
            ),
        }

        correlation_matrix, rois_idxs = _calculate_cross_correlation(data)

        assert correlation_matrix is not None
        # Off-diagonal elements should show low or negative correlation
        assert abs(correlation_matrix[0, 1]) < 1.0
        assert abs(correlation_matrix[1, 0]) < 1.0

    def test_empty_or_none_dec_dff_handling(self):
        """Test handling of empty or None dec_dff traces."""
        data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=[],  # Empty trace
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=None,  # None trace
                active=True,
            ),
            "3": ROIData(
                well_fov_position="A1_0000_p0",
                dec_dff=[0.1, 0.2, 0.3],
                active=True,
            ),
        }

        correlation_matrix, rois_idxs = _calculate_cross_correlation(data)

        # Should handle gracefully - only one valid trace, so return None
        assert correlation_matrix is None
        assert rois_idxs is None

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_calcium_peaks_correlation._calculate_cross_correlation"
    )
    def test_plot_functions_handle_none_correlation(self, mock_calc, mock_widget):
        """Test that plot functions handle None correlation matrix gracefully."""
        mock_calc.return_value = (None, None)

        # Test cross-correlation plot
        _plot_cross_correlation_data(mock_widget, {})
        mock_widget.figure.clear.assert_called()

        # Test hierarchical clustering plot
        _plot_hierarchical_clustering_data(mock_widget, {})
        assert mock_widget.figure.clear.call_count == 2
