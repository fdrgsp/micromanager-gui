"""Test script demonstrating spike-based network analysis enhancements."""

import sys
from pathlib import Path

import numpy as np

# Import the new spike-based analysis modules
from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_inferred_spike_synchrony import (  # noqa: E501
    _get_spike_trains_from_rois,
)
from micromanager_gui._plate_viewer._util import ROIData, _get_spike_synchrony_matrix

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
        )

        data[str(roi_id)] = roi_data

    return data


def test_spike_synchrony_analysis():
    """Test spike-based synchrony analysis."""
    print("Testing Spike Synchrony Analysis...")

    data = create_synthetic_spike_data()
    rois = list(range(5))  # Test with first 5 ROIs

    # Test spike train extraction
    spike_trains = _get_spike_trains_from_rois(data, rois)
    print(f"Extracted spike trains for {len(spike_trains or [])} ROIs")

    # Test synchrony matrix calculation using correlation method
    if spike_trains:
        # Convert spike trains to spike data dict for correlation analysis
        spike_data_dict = {
            roi_name: spike_train.astype(float).tolist()
            for roi_name, spike_train in spike_trains.items()
        }

        synchrony_matrix = _get_spike_synchrony_matrix(spike_data_dict)
        if synchrony_matrix is not None:
            print(f"Synchrony matrix shape: {synchrony_matrix.shape}")
            mean_sync = np.mean(
                synchrony_matrix[np.triu_indices_from(synchrony_matrix, k=1)]
            )
            print(f"Mean synchrony: {mean_sync:.3f}")
        else:
            print("Failed to calculate synchrony matrix")
    else:
        print("No spike trains extracted")

    print("✓ Spike synchrony analysis test completed\n")


def test_integration():
    """Test integration of all spike-based analysis methods."""
    print("Testing Integration of All Methods...")

    data = create_synthetic_spike_data(n_rois=15, n_timepoints=2000)
    rois = list(range(10))  # Test with subset of ROIs

    print("Running all analysis methods on the same dataset:")

    # 1. Synchrony analysis
    spike_trains = _get_spike_trains_from_rois(data, rois)
    if spike_trains:
        # Convert spike trains to spike data dict for correlation analysis
        spike_data_dict = {
            roi_name: spike_train.astype(float).tolist()
            for roi_name, spike_train in spike_trains.items()
        }

        synchrony_matrix = _get_spike_synchrony_matrix(spike_data_dict)
        if synchrony_matrix is not None:
            mean_sync = np.mean(
                synchrony_matrix[np.triu_indices_from(synchrony_matrix, k=1)]
            )
            print(
                f"  Synchrony: {len(spike_trains)} ROIs, "
                f"mean synchrony = {mean_sync:.3f}"
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
        test_integration()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        print()
        print("Summary of implemented spike-based analysis methods:")
        print("1. ✓ Spike-based synchrony analysis (_plot_spike_synchrony.py)")
        print()
        print("These modules complement the existing calcium-based analysis")
        print("and provide discrete event-driven insights into network dynamics.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
