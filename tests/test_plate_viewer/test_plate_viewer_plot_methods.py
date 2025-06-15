"""Tests for plate viewer plot methods."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from micromanager_gui._plate_viewer._graph_widgets import (
    _MultilWellGraphWidget,
    _SingleWellGraphWidget,
)
from micromanager_gui._plate_viewer._plot_methods import (
    plot_multi_well_data,
    plot_single_well_data,
)
from micromanager_gui._plate_viewer._plot_methods._main_plot import (
    AMPLITUDE_AND_FREQUENCY_GROUP,
    CSV_BAR_PLOT_AMPLITUDE,
    CSV_BAR_PLOT_FREQUENCY,
    DEC_DFF,
    DEC_DFF_AMPLITUDE,
    DEC_DFF_FREQUENCY,
    DEC_DFF_NORMALIZED,
    DEC_DFF_NORMALIZED_ACTIVE_ONLY,
    DEC_DFF_WITH_PEAKS,
    DFF,
    MULTI_WELL_COMBO_OPTIONS_DICT,
    MW_EVOKED_GROUP,
    MW_GENERAL_GROUP,
    NORMALIZED_TRACES,
    RASTER_PLOT,
    RAW_TRACES,
    SINGLE_WELL_COMBO_OPTIONS_DICT,
    TRACES_GROUP,
)
from micromanager_gui._plate_viewer._plot_methods._multi_wells_plots._csv_bar_plot import (  # noqa: E501
    plot_csv_bar_plot,
)
from micromanager_gui._plate_viewer._util import ROIData

# Create MULTI_WELL_COMBO_OPTIONS from the available groups
MULTI_WELL_COMBO_OPTIONS = list(MW_GENERAL_GROUP.keys()) + list(MW_EVOKED_GROUP.keys())

# Test data paths
TEST_DATA_SPONTANEOUS = Path(__file__).parent / "data" / "spontaneous"
TEST_DATA_EVOKED = Path(__file__).parent / "data" / "evoked"

SPONT_ANALYSIS_PATH = str(TEST_DATA_SPONTANEOUS / "spont_analysis")
EVOKED_ANALYSIS_PATH = str(TEST_DATA_EVOKED / "evk_analysis")

# Sample ROI data for testing
SAMPLE_ROI_DATA = {
    "1": ROIData(
        well_fov_position="B5_0000_p0",
        raw_trace=[100.0, 105.0, 110.0, 115.0, 120.0] * 20,
        dff=[0.0, 0.05, 0.10, 0.15, 0.20] * 20,
        dec_dff=[0.0, 0.04, 0.08, 0.12, 0.16] * 20,
        inferred_spikes=[0.0, 0.2, 0.0, 0.8, 0.3] * 20,
        peaks_dec_dff=[10, 30, 50, 70, 90],
        peaks_amplitudes_dec_dff=[0.08, 0.12, 0.09, 0.11, 0.10],
        dec_dff_frequency=0.5,
        iei=[20.0, 20.0, 20.0, 20.0],
        cell_size=150.5,
        evoked_experiment=False,
        stimulated=False,
        amplitudes_stimulated_peaks={"stim1": [0.15, 0.25, 0.20]},
        amplitudes_non_stimulated_peaks={"stim1": [0.02, 0.04, 0.03]},
        active=True,
    ),
    "2": ROIData(
        well_fov_position="B5_0000_p0",
        raw_trace=[95.0, 100.0, 105.0, 110.0, 115.0] * 20,
        dff=[0.0, 0.053, 0.105, 0.158, 0.211] * 20,
        dec_dff=[0.0, 0.042, 0.084, 0.126, 0.168] * 20,
        inferred_spikes=[0.0, 0.1, 0.0, 0.6, 0.2] * 20,
        peaks_dec_dff=[15, 35, 55, 75, 95],
        peaks_amplitudes_dec_dff=[0.07, 0.11, 0.08, 0.10, 0.09],
        dec_dff_frequency=0.4,
        iei=[20.0, 20.0, 20.0, 20.0],
        cell_size=140.2,
        evoked_experiment=False,
        stimulated=False,
        amplitudes_stimulated_peaks={"stim1": [0.12, 0.22, 0.18]},
        amplitudes_non_stimulated_peaks={"stim1": [0.015, 0.030, 0.025]},
        active=True,
    ),
}


class TestSingleWellPlotMethods:
    """Test single well plot methods."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_plot_traces_raw(self, mock_widget):
        """Test plotting raw traces."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, RAW_TRACES)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_dff(self, mock_widget):
        """Test plotting delta F/F traces."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, DFF)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_deconvolved(self, mock_widget):
        """Test plotting deconvolved traces."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, DEC_DFF)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_normalized(self, mock_widget):
        """Test plotting normalized traces."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, DEC_DFF_NORMALIZED)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_traces_with_peaks(self, mock_widget):
        """Test plotting traces with peaks."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, DEC_DFF_WITH_PEAKS)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_amplitude_data(self, mock_widget):
        """Test plotting amplitude data."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, DEC_DFF_AMPLITUDE)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_frequency_data(self, mock_widget):
        """Test plotting frequency data."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, DEC_DFF_FREQUENCY)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_raster_plot(self, mock_widget):
        """Test plotting raster plot."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, RASTER_PLOT)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_with_roi_filter(self, mock_widget):
        """Test plotting with ROI filter."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, RAW_TRACES, rois=[1])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once_with(111)

    def test_plot_empty_text(self, mock_widget):
        """Test plotting with empty text returns early."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, "")
        # Should call clear but not add_subplot
        mock_widget.figure.clear.assert_called_once()

    def test_plot_none_text(self, mock_widget):
        """Test plotting with None text returns early."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, "None")
        # Should call clear but not add_subplot
        mock_widget.figure.clear.assert_called_once()

    def test_plot_section_header(self, mock_widget):
        """Test plotting with section header returns early."""
        section_header = next(iter(SINGLE_WELL_COMBO_OPTIONS_DICT.keys()))
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, section_header)
        # Should call clear but not add_subplot
        mock_widget.figure.clear.assert_called_once()

    @pytest.mark.parametrize("trace_type", list(TRACES_GROUP.keys()))
    def test_all_trace_types(self, mock_widget, trace_type):
        """Test all trace types from TRACES_GROUP."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, trace_type)
        mock_widget.figure.clear.assert_called_once()

    @pytest.mark.parametrize(
        "amplitude_type", list(AMPLITUDE_AND_FREQUENCY_GROUP.keys())
    )
    def test_all_amplitude_types(self, mock_widget, amplitude_type):
        """Test all amplitude types from AMPLITUDE_AND_FREQUENCY_GROUP."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, amplitude_type)
        mock_widget.figure.clear.assert_called_once()


class TestMultiWellPlotMethods:
    """Test multi well plot methods."""

    @pytest.fixture
    def mock_widget(self) -> _MultilWellGraphWidget:
        """Create a mock multi well graph widget."""
        widget = Mock(spec=_MultilWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        # Mock the conditions property that's needed by the CSV plotting functions
        widget.conditions = {}
        return widget

    def test_plot_multi_well_amplitude_bar(self, mock_widget):
        """Test plotting amplitude bar plot."""
        # This will look for CSV files in the analysis path
        plot_multi_well_data(mock_widget, CSV_BAR_PLOT_AMPLITUDE, SPONT_ANALYSIS_PATH)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_multi_well_frequency_bar(self, mock_widget):
        """Test plotting frequency bar plot."""
        plot_multi_well_data(mock_widget, CSV_BAR_PLOT_FREQUENCY, SPONT_ANALYSIS_PATH)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_empty_text(self, mock_widget):
        """Test plotting with empty text returns early."""
        plot_multi_well_data(mock_widget, "", SPONT_ANALYSIS_PATH)

        mock_widget.figure.clear.assert_called_once()
        # Should return early after clearing

    def test_plot_none_text(self, mock_widget):
        """Test plotting with None text returns early."""
        plot_multi_well_data(mock_widget, "None", SPONT_ANALYSIS_PATH)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_no_analysis_path(self, mock_widget):
        """Test plotting with no analysis path returns early."""
        plot_multi_well_data(mock_widget, CSV_BAR_PLOT_AMPLITUDE, None)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_nonexistent_analysis_path(self, mock_widget):
        """Test plotting with nonexistent analysis path."""
        with patch("builtins.print"):
            plot_multi_well_data(
                mock_widget, CSV_BAR_PLOT_AMPLITUDE, "/nonexistent/path"
            )
            mock_widget.figure.clear.assert_called_once()

    @pytest.mark.parametrize("plot_type", MULTI_WELL_COMBO_OPTIONS)
    def test_all_multi_well_plot_types(self, mock_widget, plot_type):
        """Test all multi well plot types."""
        analysis_path = SPONT_ANALYSIS_PATH
        if "stimulated" in plot_type.lower():
            analysis_path = EVOKED_ANALYSIS_PATH

        plot_multi_well_data(mock_widget, plot_type, analysis_path)
        mock_widget.figure.clear.assert_called_once()


class TestPlotMethodsWithRealData:
    """Test plot methods with real data from test files."""

    @pytest.fixture
    def real_roi_data(self) -> dict[str, ROIData]:
        """Load real ROI data from test JSON file."""
        json_file = TEST_DATA_SPONTANEOUS / "spont_analysis" / "B5_0000_p0.json"

        if not json_file.exists():
            pytest.skip(f"Test data file not found: {json_file}")

        with open(json_file) as f:
            data = json.load(f)

        # Convert to ROIData objects
        roi_data = {}
        for roi_id, roi_info in data.items():
            roi_data[roi_id] = ROIData(
                well_fov_position=roi_info.get("well_fov_position", ""),
                raw_trace=roi_info.get("raw_trace", []),
                dff=roi_info.get("dff", []),
                dec_dff=roi_info.get("dec_dff", []),
                peaks_dec_dff=roi_info.get("peaks_dec_dff", []),
                peaks_amplitudes_dec_dff=roi_info.get("peaks_amplitudes_dec_dff", []),
                dec_dff_frequency=roi_info.get("dec_dff_frequency", 0.0),
                iei=roi_info.get("iei", []),
                cell_size=roi_info.get("cell_size", 0.0),
                evoked_experiment=roi_info.get("evoked_experiment", False),
                stimulated=roi_info.get("stimulated", False),
                amplitudes_stimulated_peaks=roi_info.get(
                    "amplitudes_stimulated_peaks", {}
                ),
                amplitudes_non_stimulated_peaks=roi_info.get(
                    "amplitudes_non_stimulated_peaks", {}
                ),
                active=roi_info.get("active", True),
            )

        return roi_data

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    @pytest.fixture
    def mock_multi_widget(self) -> _MultilWellGraphWidget:
        """Create a mock multi well graph widget."""
        widget = Mock(spec=_MultilWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        # Mock the conditions property that's needed by the CSV plotting functions
        widget.conditions = {}
        return widget

    def test_plot_real_traces_data(self, mock_widget, real_roi_data):
        """Test plotting with real traces data."""
        plot_single_well_data(mock_widget, real_roi_data, RAW_TRACES)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_real_dff_data(self, mock_widget, real_roi_data):
        """Test plotting with real DFF data."""
        plot_single_well_data(mock_widget, real_roi_data, DFF)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_real_amplitude_data(self, mock_widget, real_roi_data):
        """Test plotting with real amplitude data."""
        plot_single_well_data(mock_widget, real_roi_data, DEC_DFF_AMPLITUDE)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_real_multi_well_data(self, mock_multi_widget):
        """Test plotting real multi well data with CSV files."""
        if not (Path(SPONT_ANALYSIS_PATH) / "grouped").exists():
            pytest.skip("Test CSV files not found")

        plot_multi_well_data(
            mock_multi_widget, CSV_BAR_PLOT_AMPLITUDE, SPONT_ANALYSIS_PATH
        )

        mock_multi_widget.figure.clear.assert_called_once()


class TestPlotMethodsErrorHandling:
    """Test error handling in plot methods."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_plot_with_empty_data(self, mock_widget):
        """Test plotting with empty data."""
        plot_single_well_data(mock_widget, {}, RAW_TRACES)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_with_invalid_roi_data(self, mock_widget):
        """Test plotting with invalid ROI data structure."""
        invalid_data = {"1": "not_a_roi_data_object"}

        # Should not raise an exception, but may not plot anything meaningful
        plot_single_well_data(mock_widget, invalid_data, RAW_TRACES)

        mock_widget.figure.clear.assert_called_once()

    def test_plot_with_missing_trace_data(self, mock_widget):
        """Test plotting with ROI data missing trace information."""
        incomplete_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[],  # Empty trace
                dff=None,
                dec_dff=None,
                peaks_dec_dff=[],
                peaks_amplitudes_dec_dff=[],
                dec_dff_frequency=0.0,
                iei=[],
                cell_size=0.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        plot_single_well_data(mock_widget, incomplete_data, RAW_TRACES)

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()

    def test_plot_with_roi_filter_no_match(self, mock_widget):
        """Test plotting with ROI filter that matches no ROIs."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, RAW_TRACES, rois=[999])

        mock_widget.figure.clear.assert_called_once()
        mock_widget.figure.add_subplot.assert_called_once()


class TestPlotMethodConstants:
    """Test plot method constants and configuration."""

    def test_traces_group_contains_expected_options(self):
        """Test that TRACES_GROUP contains expected plotting options."""
        assert RAW_TRACES in TRACES_GROUP
        assert DFF in TRACES_GROUP
        assert DEC_DFF in TRACES_GROUP
        assert DEC_DFF_NORMALIZED in TRACES_GROUP

    def test_amplitude_group_contains_expected_options(self):
        """Test that AMPLITUDE_AND_FREQUENCY_GROUP contains expected options."""
        assert DEC_DFF_AMPLITUDE in AMPLITUDE_AND_FREQUENCY_GROUP

    def test_single_well_combo_options_structure(self):
        """Test that single well combo options are properly structured."""
        assert isinstance(SINGLE_WELL_COMBO_OPTIONS_DICT, dict)
        assert len(SINGLE_WELL_COMBO_OPTIONS_DICT) > 0

        # Check that all sections start with dashes (indicating they are headers)
        for section in SINGLE_WELL_COMBO_OPTIONS_DICT.keys():
            assert section.startswith("-")

    def test_multi_well_combo_options_structure(self):
        """Test that multi well combo options are properly structured."""
        assert isinstance(MULTI_WELL_COMBO_OPTIONS, list)
        assert len(MULTI_WELL_COMBO_OPTIONS) > 0
        assert CSV_BAR_PLOT_AMPLITUDE in MULTI_WELL_COMBO_OPTIONS
        assert CSV_BAR_PLOT_FREQUENCY in MULTI_WELL_COMBO_OPTIONS

        # Also test the dictionary structure
        assert isinstance(MULTI_WELL_COMBO_OPTIONS_DICT, dict)
        assert len(MULTI_WELL_COMBO_OPTIONS_DICT) > 0


class TestPlotMethodsIntegration:
    """Integration tests using real CSV data files."""

    def test_csv_amplitude_plotting_with_real_data(self):
        """Test CSV amplitude plotting with real CSV files."""
        csv_file = (
            TEST_DATA_SPONTANEOUS
            / "spont_analysis"
            / "grouped"
            / "spont_analysis_amplitude.csv"
        )
        if not csv_file.exists():
            pytest.skip("Real CSV data not available")

        widget = Mock(spec=_MultilWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget.conditions = {}

        info = {
            "parameter": "Amplitude",
            "suffix": "amplitude",
            "add_to_title": " (Deconvolved ΔF/F)",
        }
        plot_csv_bar_plot(widget, str(csv_file), info)

        widget.figure.add_subplot.assert_called_once()
        widget.canvas.draw.assert_called_once()

    def test_csv_frequency_plotting_with_real_data(self):
        """Test CSV frequency plotting with real CSV files."""
        csv_file = (
            TEST_DATA_SPONTANEOUS
            / "spont_analysis"
            / "grouped"
            / "spont_analysis_frequency.csv"
        )
        if not csv_file.exists():
            pytest.skip("Real CSV data not available")

        widget = Mock(spec=_MultilWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget.conditions = {}

        info = {
            "parameter": "Frequency",
            "suffix": "frequency",
            "add_to_title": " (Deconvolved ΔF/F)",
            "units": "Hz",
        }
        plot_csv_bar_plot(widget, str(csv_file), info)

        widget.figure.add_subplot.assert_called_once()
        widget.canvas.draw.assert_called_once()

    def test_csv_synchrony_plotting_special_case(self):
        """Test CSV synchrony plotting which uses mean_n_sem=False."""
        csv_file = (
            TEST_DATA_SPONTANEOUS
            / "spont_analysis"
            / "grouped"
            / "spont_analysis_synchrony.csv"
        )
        if not csv_file.exists():
            pytest.skip("Real CSV data not available")

        widget = Mock(spec=_MultilWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        widget.conditions = {}

        info = {
            "parameter": "Global Synchrony",
            "suffix": "synchrony",
            "add_to_title": "(Median)",
            "units": "Index",
        }
        plot_csv_bar_plot(widget, str(csv_file), info, mean_n_sem=False)

        widget.figure.add_subplot.assert_called_once()
        widget.canvas.draw.assert_called_once()


class TestPlotMethodsPerformance:
    """Performance tests for plot methods with larger datasets."""

    @pytest.fixture
    def large_roi_data(self) -> dict[str, ROIData]:
        """Create large dataset for performance testing."""
        large_data = {}
        for i in range(100):  # 100 ROIs
            large_data[str(i)] = ROIData(
                well_fov_position=f"B5_000{i % 10}_p0",
                raw_trace=[
                    100.0 + np.random.rand() for _ in range(1000)
                ],  # 1000 frames
                dff=[np.random.rand() for _ in range(1000)],
                dec_dff=[np.random.rand() for _ in range(1000)],
                peaks_dec_dff=sorted(
                    np.random.choice(range(1000), size=50, replace=False).tolist()
                ),
                peaks_amplitudes_dec_dff=[np.random.rand() for _ in range(50)],
                dec_dff_frequency=np.random.rand() * 5,
                iei=[np.random.rand() * 100 for _ in range(20)],
                cell_size=100.0 + np.random.rand() * 50,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        return large_data

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_large_dataset_performance_raw_traces(self, mock_widget, large_roi_data):
        """Test performance with large dataset for raw traces."""
        start_time = time.time()
        plot_single_well_data(mock_widget, large_roi_data, RAW_TRACES)
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        mock_widget.figure.clear.assert_called_once()

    def test_large_dataset_performance_with_roi_filter(
        self, mock_widget, large_roi_data
    ):
        """Test performance with large dataset and ROI filtering."""
        # Filter to only first 10 ROIs
        rois = list(range(10))

        start_time = time.time()
        plot_single_well_data(mock_widget, large_roi_data, DEC_DFF_AMPLITUDE, rois=rois)
        end_time = time.time()

        # Should complete within reasonable time (< 2 seconds with filtering)
        assert end_time - start_time < 2.0
        mock_widget.figure.clear.assert_called_once()

    def test_large_dataset_memory_usage(self, mock_widget, large_roi_data):
        """Test that large datasets don't cause memory issues."""
        # This test just ensures no memory-related crashes occur
        plot_single_well_data(mock_widget, large_roi_data, DEC_DFF_NORMALIZED)
        plot_single_well_data(mock_widget, large_roi_data, DEC_DFF_WITH_PEAKS)
        plot_single_well_data(mock_widget, large_roi_data, RASTER_PLOT)

        # If we get here without crashing, memory usage is acceptable
        assert mock_widget.figure.clear.call_count == 3


class TestPlotMethodsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_single_frame_trace(self, mock_widget):
        """Test plotting with single-frame traces."""
        single_frame_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0],  # Single frame
                dff=[0.0],
                dec_dff=[0.0],
                peaks_dec_dff=[],
                peaks_amplitudes_dec_dff=[],
                dec_dff_frequency=0.0,
                iei=[],
                cell_size=150.5,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        plot_single_well_data(mock_widget, single_frame_data, RAW_TRACES)
        mock_widget.figure.clear.assert_called_once()

    def test_zero_amplitude_data(self, mock_widget):
        """Test plotting with zero amplitude data."""
        zero_amp_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 100,
                dff=[0.0] * 100,
                dec_dff=[0.0] * 100,
                peaks_dec_dff=[],
                peaks_amplitudes_dec_dff=[0.0, 0.0, 0.0],  # Zero amplitudes
                dec_dff_frequency=0.0,
                iei=[],
                cell_size=0.0,  # Zero cell size
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        plot_single_well_data(mock_widget, zero_amp_data, DEC_DFF_AMPLITUDE)
        mock_widget.figure.clear.assert_called_once()

    def test_negative_values_in_trace(self, mock_widget):
        """Test plotting with negative values in traces."""
        negative_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[-50.0, -25.0, 0.0, 25.0, 50.0] * 20,  # Negative values
                dff=[-0.5, -0.25, 0.0, 0.25, 0.5] * 20,
                dec_dff=[-0.3, -0.15, 0.0, 0.15, 0.3] * 20,
                peaks_dec_dff=[10, 30, 50, 70, 90],
                peaks_amplitudes_dec_dff=[0.15, 0.25, 0.20, 0.18, 0.22],
                dec_dff_frequency=0.8,
                iei=[15.0, 20.0, 25.0, 18.0],
                cell_size=120.5,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        plot_single_well_data(mock_widget, negative_data, DFF)
        mock_widget.figure.clear.assert_called_once()

    def test_very_large_roi_numbers(self, mock_widget):
        """Test plotting with very large ROI numbers."""
        large_roi_data = {
            "999999": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 50,
                dff=[0.1] * 50,
                dec_dff=[0.08] * 50,
                peaks_dec_dff=[10, 20, 30, 40],
                peaks_amplitudes_dec_dff=[0.12, 0.15, 0.11, 0.14],
                dec_dff_frequency=1.2,
                iei=[10.0, 15.0, 12.0],
                cell_size=180.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        plot_single_well_data(mock_widget, large_roi_data, RAW_TRACES)
        mock_widget.figure.clear.assert_called_once()

    def test_missing_optional_data_fields(self, mock_widget):
        """Test plotting when optional data fields are None or missing."""
        minimal_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 50,
                dff=None,  # Missing dff data
                dec_dff=[0.08] * 50,
                peaks_dec_dff=[],  # No peaks
                peaks_amplitudes_dec_dff=[],  # No amplitudes
                dec_dff_frequency=None,  # Missing frequency
                iei=[],  # No IEI data
                cell_size=180.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        # Should handle missing data gracefully
        plot_single_well_data(mock_widget, minimal_data, RAW_TRACES)
        plot_single_well_data(mock_widget, minimal_data, DEC_DFF_AMPLITUDE)
        plot_single_well_data(mock_widget, minimal_data, DEC_DFF_FREQUENCY)

        assert mock_widget.figure.clear.call_count == 3


class TestPlotMethodsRegression:
    """Regression tests for specific plot functionality."""

    @pytest.fixture
    def mock_widget(self) -> _SingleWellGraphWidget:
        """Create a mock single well graph widget."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget.figure = Mock(spec=Figure)
        widget.figure.clear = Mock()
        widget.figure.add_subplot = Mock()
        widget.canvas = Mock()
        widget.canvas.draw = Mock()
        return widget

    def test_normalization_with_identical_values(self, mock_widget):
        """Regression test: normalization when all values are identical."""
        identical_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 100,  # All identical values
                dff=[0.1] * 100,
                dec_dff=[0.08] * 100,
                peaks_dec_dff=[10, 20, 30],
                peaks_amplitudes_dec_dff=[0.08, 0.08, 0.08],
                dec_dff_frequency=1.0,
                iei=[10.0, 10.0, 10.0],
                cell_size=150.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        # Should handle normalization without division by zero
        plot_single_well_data(mock_widget, identical_data, NORMALIZED_TRACES)
        mock_widget.figure.clear.assert_called_once()

    def test_plot_with_inactive_rois_only(self, mock_widget):
        """Regression test: plotting when all ROIs are inactive."""
        inactive_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 50,
                dff=[0.1] * 50,
                dec_dff=[0.08] * 50,
                peaks_dec_dff=[],
                peaks_amplitudes_dec_dff=[],
                dec_dff_frequency=0.0,
                iei=[],
                cell_size=150.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=False,  # Inactive ROI
            ),
            "2": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[95.0] * 50,
                dff=[0.05] * 50,
                dec_dff=[0.04] * 50,
                peaks_dec_dff=[],
                peaks_amplitudes_dec_dff=[],
                dec_dff_frequency=0.0,
                iei=[],
                cell_size=140.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=False,  # Inactive ROI
            ),
        }

        # Should handle all inactive ROIs gracefully
        plot_single_well_data(
            mock_widget, inactive_data, DEC_DFF_NORMALIZED_ACTIVE_ONLY
        )
        mock_widget.figure.clear.assert_called_once()

    def test_very_high_frequency_data(self, mock_widget):
        """Regression test: very high frequency values."""
        high_freq_data = {
            "1": ROIData(
                well_fov_position="B5_0000_p0",
                raw_trace=[100.0] * 50,
                dff=[0.1] * 50,
                dec_dff=[0.08] * 50,
                peaks_dec_dff=list(range(0, 50, 2)),  # High frequency peaks
                peaks_amplitudes_dec_dff=[0.1] * 25,
                dec_dff_frequency=50.0,  # Very high frequency
                iei=[1.0] * 24,  # Very short intervals
                cell_size=150.0,
                evoked_experiment=False,
                stimulated=False,
                amplitudes_stimulated_peaks={},
                amplitudes_non_stimulated_peaks={},
                active=True,
            )
        }

        plot_single_well_data(mock_widget, high_freq_data, DEC_DFF_FREQUENCY)
        mock_widget.figure.clear.assert_called_once()

    @pytest.mark.parametrize(
        "plot_option",
        [
            RAW_TRACES,
            DFF,
            DEC_DFF,
            DEC_DFF_NORMALIZED,
            DEC_DFF_WITH_PEAKS,
            DEC_DFF_AMPLITUDE,
            DEC_DFF_FREQUENCY,
            RASTER_PLOT,
        ],
    )
    def test_all_major_plot_types_regression(self, mock_widget, plot_option):
        """Regression test: ensure all major plot types work with standard data."""
        plot_single_well_data(mock_widget, SAMPLE_ROI_DATA, plot_option)
        mock_widget.figure.clear.assert_called_once()
        # Reset the mock for next iteration
        mock_widget.figure.clear.reset_mock()
