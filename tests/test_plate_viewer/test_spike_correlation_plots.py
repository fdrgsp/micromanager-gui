"""Tests for spike correlation plots functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_spike_correlation import (  # noqa: E501
    _add_hover_functionality_spike_clustering,
    _add_hover_functionality_spike_corr,
    _calculate_spike_cross_correlation,
    _plot_spike_cross_correlation_data,
    _plot_spike_hierarchical_clustering_data,
    _plot_spike_hierarchical_clustering_dendrogram,
    _plot_spike_hierarchical_clustering_map,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestSpikeCorrelationPlots:
    """Test suite for spike correlation plot functionality."""

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
    def sample_spike_data_active(self):
        """Create sample ROI data with spike trains for correlation testing."""
        # Create synthetic spike trains with different patterns
        # ROI 1: regular spikes every 10 frames
        spikes1 = [1.0 if i % 10 == 0 else 0.0 for i in range(100)]
        # ROI 2: similar pattern shifted by 2 frames (correlated)
        spikes2 = [1.0 if (i - 2) % 10 == 0 and i >= 2 else 0.0 for i in range(100)]
        # ROI 3: different pattern, spikes every 15 frames
        spikes3 = [1.0 if i % 15 == 0 else 0.0 for i in range(100)]

        return {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes1,
                inferred_spikes_threshold=0.5,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes2,
                inferred_spikes_threshold=0.5,
                active=True,
            ),
            "3": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes3,
                inferred_spikes_threshold=0.5,
                active=True,
            ),
        }

    @pytest.fixture
    def sample_spike_data_mixed(self):
        """Create sample ROI data with mix of active and inactive ROIs."""
        spikes1 = [1.0 if i % 8 == 0 else 0.0 for i in range(50)]
        spikes2 = [1.0 if i % 12 == 0 else 0.0 for i in range(50)]

        return {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes1,
                inferred_spikes_threshold=0.5,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes2,
                inferred_spikes_threshold=0.5,
                active=False,  # Inactive ROI
            ),
            "3": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=[],  # Empty spikes
                active=True,
            ),
        }

    @pytest.fixture
    def empty_spike_data(self):
        """Create ROI data with no spikes."""
        return {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=[],
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=[],
                active=True,
            ),
        }

    def test_calculate_spike_cross_correlation_basic(self, sample_spike_data_active):
        """Test basic spike cross-correlation calculation."""
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            sample_spike_data_active
        )

        assert correlation_matrix is not None
        assert roi_indices is not None
        assert correlation_matrix.shape == (3, 3)
        assert len(roi_indices) == 3
        assert roi_indices == [1, 2, 3]

        # Check that diagonal is 1 (self-correlation)
        np.testing.assert_allclose(np.diag(correlation_matrix), 1.0, atol=1e-10)

        # Check symmetry
        np.testing.assert_allclose(correlation_matrix, correlation_matrix.T)

    def test_calculate_spike_cross_correlation_with_roi_filter(
        self, sample_spike_data_active
    ):
        """Test spike cross-correlation with specific ROI filter."""
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            sample_spike_data_active, rois=[1, 3]
        )

        assert correlation_matrix is not None
        assert roi_indices is not None
        assert correlation_matrix.shape == (2, 2)
        assert roi_indices == [1, 3]

    def test_calculate_spike_cross_correlation_inactive_rois(
        self, sample_spike_data_mixed
    ):
        """Test that inactive ROIs are excluded from correlation calculation."""
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            sample_spike_data_mixed
        )

        # Should return None because only 1 ROI has valid spikes (need at least 2)
        assert correlation_matrix is None
        assert roi_indices is None

    def test_calculate_spike_cross_correlation_single_roi(
        self, sample_spike_data_active
    ):
        """Test spike cross-correlation with single ROI (should return None)."""
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            sample_spike_data_active, rois=[1]
        )

        assert correlation_matrix is None
        assert roi_indices is None

    def test_calculate_spike_cross_correlation_empty_data(self, empty_spike_data):
        """Test spike cross-correlation with empty spike data."""
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            empty_spike_data
        )

        assert correlation_matrix is None
        assert roi_indices is None

    def test_plot_spike_cross_correlation_data_success(
        self, mock_widget, sample_spike_data_active
    ):
        """Test successful spike cross-correlation plotting."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_spike_cross_correlation_data(mock_widget, sample_spike_data_active)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()
        mock_ax.imshow.assert_called_once()
        mock_widget.figure.colorbar.assert_called_once()

    def test_plot_spike_cross_correlation_data_insufficient_rois(
        self, mock_widget, sample_spike_data_active
    ):
        """Test spike cross-correlation plotting with insufficient ROIs."""
        result = _plot_spike_cross_correlation_data(
            mock_widget, sample_spike_data_active, rois=[1]
        )

        assert result is None
        mock_widget.figure.clear.assert_called_once()

    def test_plot_spike_cross_correlation_data_with_roi_filter(
        self, mock_widget, sample_spike_data_active
    ):
        """Test spike cross-correlation plotting with ROI filter."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_spike_cross_correlation_data(
            mock_widget, sample_spike_data_active, rois=[1, 2]
        )

        mock_widget.figure.clear.assert_called_once()
        mock_ax.imshow.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots."
        "_plot_spike_correlation.mplcursors"
    )
    def test_add_hover_functionality_spike_cross_corr(
        self, mock_mplcursors, mock_widget
    ):
        """Test adding hover functionality to spike cross-correlation plot."""
        mock_image = Mock()
        mock_cursor = Mock()
        mock_mplcursors.cursor.return_value = mock_cursor

        _add_hover_functionality_spike_corr(
            mock_image, mock_widget, [1, 2, 3], np.array([[1.0, 0.5], [0.5, 1.0]])
        )

        mock_mplcursors.cursor.assert_called_once()
        mock_cursor.connect.assert_called_once_with("add")

    def test_plot_spike_hierarchical_clustering_data_map_mode(
        self, mock_widget, sample_spike_data_active
    ):
        """Test spike hierarchical clustering plotting in map mode."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax

        _plot_spike_hierarchical_clustering_data(
            mock_widget, sample_spike_data_active, use_dendrogram=False
        )

        mock_widget.figure.clear.assert_called_once()
        mock_ax.imshow.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots."
        "_plot_spike_correlation.dendrogram"
    )
    def test_plot_spike_hierarchical_clustering_data_dendrogram_mode(
        self, mock_dendrogram, mock_widget, sample_spike_data_active
    ):
        """Test spike hierarchical clustering plotting in dendrogram mode."""
        mock_ax = Mock()
        mock_widget.figure.add_subplot.return_value = mock_ax
        mock_dendrogram.return_value = {"leaves": [0, 1, 2]}

        _plot_spike_hierarchical_clustering_data(
            mock_widget, sample_spike_data_active, use_dendrogram=True
        )

        mock_widget.figure.clear.assert_called_once()
        mock_dendrogram.assert_called_once()

    def test_plot_spike_hierarchical_clustering_data_insufficient_rois(
        self, mock_widget, sample_spike_data_active
    ):
        """Test spike clustering with insufficient ROIs."""
        result = _plot_spike_hierarchical_clustering_data(
            mock_widget, sample_spike_data_active, rois=[1], use_dendrogram=False
        )

        assert result is None
        mock_widget.figure.clear.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots."
        "_plot_spike_correlation.dendrogram"
    )
    def test_plot_spike_hierarchical_clustering_dendrogram(
        self, mock_dendrogram, mock_widget, sample_spike_data_active
    ):
        """Test spike hierarchical clustering dendrogram plotting."""
        mock_ax = Mock()
        mock_dendrogram.return_value = {"leaves": [0, 1, 2]}
        correlation_matrix = np.array(
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]]
        )

        _plot_spike_hierarchical_clustering_dendrogram(
            mock_ax, correlation_matrix, [1, 2, 3]
        )

        mock_dendrogram.assert_called_once()
        mock_ax.set_title.assert_called_once()

    def test_plot_spike_hierarchical_clustering_map(
        self, mock_widget, sample_spike_data_active
    ):
        """Test spike hierarchical clustering map plotting."""
        mock_ax = Mock()
        correlation_matrix = np.array(
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]]
        )
        rois = [1, 2, 3]

        _plot_spike_hierarchical_clustering_map(
            mock_widget, mock_ax, correlation_matrix, rois
        )

        mock_ax.imshow.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_widget.figure.colorbar.assert_called_once()

    @patch(
        "micromanager_gui._plate_viewer._plot_methods._single_wells_plots."
        "_plot_spike_correlation.mplcursors"
    )
    def test_add_hover_functionality_spike_clustering(
        self, mock_mplcursors, mock_widget
    ):
        """Test adding hover functionality to spike clustering heatmap."""
        mock_image = Mock()
        mock_cursor = Mock()
        mock_mplcursors.cursor.return_value = mock_cursor

        _add_hover_functionality_spike_clustering(
            mock_image, mock_widget, [1, 2], [0, 1], np.array([[1.0, 0.5], [0.5, 1.0]])
        )

        mock_mplcursors.cursor.assert_called_once()
        mock_cursor.connect.assert_called_once_with("add")

    def test_spike_cross_correlation_mathematical_properties(
        self, sample_spike_data_active
    ):
        """Test mathematical properties of spike cross-correlation."""
        correlation_matrix, _ = _calculate_spike_cross_correlation(
            sample_spike_data_active
        )

        assert correlation_matrix is not None

        # Test range: correlation values should be between -1 and 1 (with tolerance)
        assert np.all(correlation_matrix >= -1.0 - 1e-10)
        assert np.all(correlation_matrix <= 1.0 + 1e-10)

        # Test diagonal: self-correlation should be 1
        np.testing.assert_allclose(np.diag(correlation_matrix), 1.0, atol=1e-10)

        # Test symmetry: correlation matrix should be symmetric
        np.testing.assert_allclose(correlation_matrix, correlation_matrix.T)

    def test_roi_filtering_edge_cases(self, sample_spike_data_active):
        """Test edge cases for ROI filtering in spike correlation."""
        # Test with empty ROI list
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            sample_spike_data_active, rois=[]
        )
        assert correlation_matrix is None
        assert roi_indices is None

        # Test with non-existent ROI indices
        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(
            sample_spike_data_active, rois=[999, 1000]
        )
        assert correlation_matrix is None
        assert roi_indices is None

    def test_spike_correlation_with_identical_trains(self):
        """Test spike correlation with identical spike trains."""
        identical_spikes = [1.0, 0.0, 1.0, 0.0, 1.0] * 10

        data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=identical_spikes,
                inferred_spikes_threshold=0.5,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=identical_spikes,
                inferred_spikes_threshold=0.5,
                active=True,
            ),
        }

        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(data)

        assert correlation_matrix is not None
        assert roi_indices is not None
        # Identical trains should have correlation of 1
        np.testing.assert_allclose(correlation_matrix[0, 1], 1.0, atol=1e-10)

    def test_spike_correlation_with_uncorrelated_trains(self):
        """Test spike correlation with completely uncorrelated spike trains."""
        # Create two uncorrelated random spike trains
        np.random.seed(42)  # For reproducible tests
        spikes1 = (np.random.rand(100) > 0.9).astype(float)
        spikes2 = (np.random.rand(100) > 0.9).astype(float)

        data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes1.tolist(),
                inferred_spikes_threshold=0.5,
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=spikes2.tolist(),
                inferred_spikes_threshold=0.5,
                active=True,
            ),
        }

        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(data)

        assert correlation_matrix is not None
        assert roi_indices is not None
        # Cross-correlation should be relatively low (but not necessarily 0)
        assert abs(correlation_matrix[0, 1]) < 0.5

    def test_empty_or_none_spike_handling(self):
        """Test handling of empty or None spike data."""
        data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=[],
                active=True,
            ),
            "2": ROIData(
                well_fov_position="A1_0000_p0",
                inferred_spikes=None,
                active=True,
            ),
        }

        correlation_matrix, roi_indices = _calculate_spike_cross_correlation(data)

        assert correlation_matrix is None
        assert roi_indices is None

    def test_plot_functions_handle_none_correlation(self, mock_widget):
        """Test that plot functions handle None correlation matrix gracefully."""
        # Test with data that will result in None correlation
        empty_data = {
            "1": ROIData(
                well_fov_position="A1_0000_p0", inferred_spikes=[], active=True
            )
        }

        result = _plot_spike_cross_correlation_data(mock_widget, empty_data)
        assert result is None

        result = _plot_spike_hierarchical_clustering_data(
            mock_widget, empty_data, use_dendrogram=False
        )
        assert result is None
