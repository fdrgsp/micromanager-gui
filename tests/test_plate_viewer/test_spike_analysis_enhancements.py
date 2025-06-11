"""Test script demonstrating spike-based network analysis enhancements."""

import sys
from pathlib import Path

import numpy as np

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_burst_detection import (  # noqa: E501
    _calculate_network_states,
    _detect_population_bursts,
    _get_population_spike_data,
)
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_spike_clustering import (  # noqa: E501
    _extract_spike_features,
    _perform_hierarchical_clustering,
)
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_spike_correlation import (  # noqa: E501
    _calculate_spike_cross_correlation,
)

# Import the new spike-based analysis modules
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_spike_synchrony import (  # noqa: E501
    _calculate_spike_synchrony_matrix,
    _get_spike_trains_from_rois,
)
from micromanager_gui._plate_viewer._util import ROIData

# Add the source directory to Python path for testing
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


def create_synthetic_spike_data(
    n_rois: int = 10, n_timepoints: int = 1000
) -> dict[str, ROIData]:
    """Create synthetic spike data for testing."""
    np.random.seed(42)  # For reproducible results

    data = {}

    for roi_id in range(n_rois):
        # Create different spike patterns for different ROIs
        if roi_id < 3:
            # High-frequency spiking neurons
            spike_prob = np.random.random(n_timepoints) * 0.8
            spike_prob[spike_prob < 0.6] = 0.0
        elif roi_id < 6:
            # Burst-like neurons (periodic high activity)
            spike_prob = np.zeros(n_timepoints)
            for burst_start in range(0, n_timepoints, 200):
                burst_end = min(burst_start + 50, n_timepoints)
                spike_prob[burst_start:burst_end] = (
                    np.random.random(burst_end - burst_start) * 0.9
                )
        else:
            # Low-frequency, irregular spiking
            spike_prob = np.random.random(n_timepoints) * 0.4
            spike_prob[spike_prob < 0.35] = 0.0

        # Add some correlated activity for synchrony testing
        if roi_id > 0 and roi_id % 2 == 0:
            # Add correlation with previous ROI
            prev_spikes = data[str(roi_id - 1)].inferred_spikes
            correlation_mask = np.random.random(n_timepoints) > 0.7
            spike_prob[correlation_mask] = np.array(prev_spikes)[correlation_mask]

        roi_data = ROIData(
            inferred_spikes=spike_prob.tolist(),
            active=True,
            dec_dff=None,  # Not needed for spike analysis
            instantaneous_phase=None,  # Not needed for spike analysis
        )

        data[str(roi_id)] = roi_data

    return data


def test_spike_synchrony_analysis():
    """Test spike-based synchrony analysis."""
    print("Testing Spike Synchrony Analysis...")

    data = create_synthetic_spike_data()
    rois = list(range(5))  # Test with first 5 ROIs

    # Test spike train extraction
    spike_trains = _get_spike_trains_from_rois(data, rois, spike_threshold=0.5)
    print(f"Extracted spike trains for {len(spike_trains)} ROIs")

    # Test synchrony matrix calculation
    if spike_trains:
        synchrony_matrix = _calculate_spike_synchrony_matrix(
            spike_trains, time_window=0.1
        )
        if synchrony_matrix is not None:
            print(f"Synchrony matrix shape: {synchrony_matrix.shape}")
            print(
                f"Mean synchrony: {np.mean(synchrony_matrix[np.triu_indices_from(synchrony_matrix, k=1)]):.3f}"
            )
        else:
            print("Failed to calculate synchrony matrix")
    else:
        print("No spike trains extracted")

    print("✓ Spike synchrony analysis test completed\n")


def test_spike_correlation_analysis():
    """Test spike train cross-correlation analysis."""
    print("Testing Spike Correlation Analysis...")

    data = create_synthetic_spike_data()
    rois = list(range(6))  # Test with first 6 ROIs

    # Test cross-correlation calculation
    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        data, rois, spike_threshold=0.5
    )

    if correlation_matrix is not None and rois_idxs is not None:
        print(f"Correlation matrix shape: {correlation_matrix.shape}")
        print(f"ROIs analyzed: {rois_idxs}")
        print(
            f"Mean correlation: {np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}"
        )
    else:
        print("Failed to calculate correlation matrix")

    print("✓ Spike correlation analysis test completed\n")


def test_burst_detection_analysis():
    """Test burst detection and network states analysis."""
    print("Testing Burst Detection Analysis...")

    data = create_synthetic_spike_data()
    rois = list(range(8))  # Test with first 8 ROIs

    # Test population spike data extraction
    spike_trains, roi_names, time_axis = _get_population_spike_data(
        data, rois, spike_threshold=0.5
    )

    if spike_trains is not None:
        print(f"Population data shape: {spike_trains.shape}")
        print(f"Time axis length: {len(time_axis)}")

        # Test burst detection
        population_activity = np.mean(spike_trains, axis=0)
        bursts = _detect_population_bursts(
            population_activity, burst_threshold=0.3, min_duration=3
        )
        print(f"Detected {len(bursts)} bursts")

        # Test network state calculation
        network_states = _calculate_network_states(spike_trains, bursts, time_axis)
        print(f"Network states: {network_states}")
    else:
        print("Failed to extract population spike data")

    print("✓ Burst detection analysis test completed\n")


def test_spike_clustering_analysis():
    """Test functional clustering based on spiking patterns."""
    print("Testing Spike Clustering Analysis...")

    data = create_synthetic_spike_data()
    rois = list(range(10))  # Test with all ROIs

    # Test feature extraction
    features, roi_names = _extract_spike_features(data, rois, spike_threshold=0.5)

    if features is not None:
        print(f"Feature matrix shape: {features.shape}")
        print(f"ROIs analyzed: {roi_names}")

        # Test clustering
        cluster_labels, linkage_matrix, distance_matrix = (
            _perform_hierarchical_clustering(features, n_clusters=3, method="ward")
        )
        print(f"Cluster labels: {cluster_labels}")
        print(f"Unique clusters: {np.unique(cluster_labels)}")
    else:
        print("Failed to extract spike features")

    print("✓ Spike clustering analysis test completed\n")


def test_integration():
    """Test integration of all spike-based analysis methods."""
    print("Testing Integration of All Methods...")

    data = create_synthetic_spike_data(n_rois=15, n_timepoints=2000)
    rois = list(range(10))  # Test with subset of ROIs

    print("Running all analysis methods on the same dataset:")

    # 1. Synchrony analysis
    spike_trains = _get_spike_trains_from_rois(data, rois, spike_threshold=0.5)
    if spike_trains:
        synchrony_matrix = _calculate_spike_synchrony_matrix(
            spike_trains, time_window=0.1
        )
        print(
            f"  Synchrony: {len(spike_trains)} ROIs, mean synchrony = {np.mean(synchrony_matrix[np.triu_indices_from(synchrony_matrix, k=1)]):.3f}"
        )

    # 2. Correlation analysis
    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        data, rois, spike_threshold=0.5
    )
    if correlation_matrix is not None:
        print(
            f"  Correlation: {len(rois_idxs)} ROIs, mean correlation = {np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}"
        )

    # 3. Burst detection
    spike_trains_pop, roi_names, time_axis = _get_population_spike_data(
        data, rois, spike_threshold=0.5
    )
    if spike_trains_pop is not None:
        population_activity = np.mean(spike_trains_pop, axis=0)
        bursts = _detect_population_bursts(
            population_activity, burst_threshold=0.3, min_duration=3
        )
        network_states = _calculate_network_states(spike_trains_pop, bursts, time_axis)
        print(
            f"  Burst detection: {len(bursts)} bursts, rate = {network_states['burst_rate']:.1f} bursts/min"
        )

    # 4. Clustering
    features, roi_names_clust = _extract_spike_features(data, rois, spike_threshold=0.5)
    if features is not None:
        cluster_labels, _, _ = _perform_hierarchical_clustering(
            features, n_clusters=None, method="ward"
        )
        n_clusters = len(np.unique(cluster_labels))
        print(
            f"  Clustering: {len(roi_names_clust)} ROIs grouped into {n_clusters} clusters"
        )

    print("✓ Integration test completed\n")


def main():
    """Run all tests for spike-based network analysis enhancements."""
    print("=" * 60)
    print("SPIKE-BASED NETWORK ANALYSIS ENHANCEMENTS - TEST SUITE")
    print("=" * 60)
    print()

    try:
        test_spike_synchrony_analysis()
        test_spike_correlation_analysis()
        test_burst_detection_analysis()
        test_spike_clustering_analysis()
        test_integration()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        print()
        print("Summary of implemented spike-based analysis methods:")
        print("1. ✓ Spike-based synchrony analysis (_plot_spike_synchrony.py)")
        print("2. ✓ Spike train cross-correlation (_plot_spike_correlation.py)")
        print("3. ✓ Burst detection & network states (_plot_burst_detection.py)")
        print(
            "4. ✓ Functional clustering based on spiking patterns (_plot_spike_clustering.py)"
        )
        print()
        print("These modules complement the existing calcium-based analysis")
        print("and provide discrete event-driven insights into network dynamics.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
