#!/usr/bin/env python3
"""Test script to demonstrate exposure time integration in spike synchrony analysis."""

import numpy as np

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_inferred_spike_synchrony import (  # noqa: E501
    _get_spike_trains_from_rois,
)
from micromanager_gui._plate_viewer._util import ROIData, _get_spike_synchrony_matrix


def create_test_data_with_temporal_info():
    """Create test data with realistic temporal information."""
    # Simulate 100 Hz acquisition (10ms exposure) for 10 seconds = 1000 frames
    exposure_time_ms = 10.0  # 10ms exposure = 100 Hz
    total_time_sec = 10.0
    n_frames = 1000

    # Create spike data for 3 ROIs
    data = {}
    for roi_id in range(3):
        # Create spike pattern: sparse spikes with different rates
        spike_probs = np.random.random(n_frames) * 0.3
        spike_probs[spike_probs < 0.25] = 0.0  # Only keep ~5% as spikes

        # Add some synchronous events
        if roi_id > 0:
            sync_frames = [200, 400, 600, 800]  # Some common spike times
            for frame in sync_frames:
                if frame < n_frames:
                    spike_probs[frame] = 0.8

        data[str(roi_id)] = ROIData(
            well_fov_position=f"A{roi_id+1}",
            inferred_spikes=spike_probs.tolist(),
            total_recording_time_sec=total_time_sec,
            active=True,
        )

    return data, exposure_time_ms


def main():
    """Demonstrate exposure time integration."""
    print("Testing Exposure Time Integration in Spike Synchrony Analysis")
    print("=" * 60)

    # Create test data
    data, expected_exposure_ms = create_test_data_with_temporal_info()
    print(f"Expected exposure time: {expected_exposure_ms:.1f} ms")

    # Test spike synchrony with correlation-based approach
    print("\nTesting Spike Synchrony Analysis:")
    spike_trains = _get_spike_trains_from_rois(data, rois=None)
    if spike_trains:
        print(f"Extracted {len(spike_trains)} spike trains")

        # Convert spike trains to spike data dict for correlation analysis
        spike_data_dict = {
            roi_name: spike_train.astype(float).tolist()
            for roi_name, spike_train in spike_trains.items()
        }

        synchrony_matrix = _get_spike_synchrony_matrix(spike_data_dict)
        if synchrony_matrix is not None:
            print(f"Synchrony matrix shape: {synchrony_matrix.shape}")

            # Show diagonal (should be 1.0) and off-diagonal values
            diagonal = np.diag(synchrony_matrix)
            off_diagonal = synchrony_matrix[np.triu_indices_from(synchrony_matrix, k=1)]

            print(f"Diagonal values (self-synchrony): {diagonal}")
            print(f"Off-diagonal mean: {np.mean(off_diagonal):.3f}")
            print(
                f"Off-diagonal range: {np.min(off_diagonal):.3f} "
                "to {np.max(off_diagonal):.3f}"
            )
        else:
            print("Failed to calculate synchrony matrix")
    else:
        print("No spike trains extracted")

    print("\n✓ Exposure time integration test completed")


if __name__ == "__main__":
    main()
