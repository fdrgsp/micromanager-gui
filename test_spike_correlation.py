"""Test the spike correlation functionality."""

import numpy as np

from micromanager_gui._plate_viewer._plot_methods._single_wells_plots._plot_spike_correlation import (  # noqa: E501
    _calculate_spike_cross_correlation,
)
from micromanager_gui._plate_viewer._util import ROIData

# Create sample ROI data with spike information
test_data = {
    "1": ROIData(
        well_fov_position="A1",
        inferred_spikes=[0.0, 0.8, 0.0, 0.6, 0.0, 0.9, 0.0, 0.7],
        inferred_spikes_threshold=0.5,
        active=True,
    ),
    "2": ROIData(
        well_fov_position="A2",
        inferred_spikes=[0.0, 0.0, 0.7, 0.0, 0.8, 0.0, 0.6, 0.0],
        inferred_spikes_threshold=0.5,
        active=True,
    ),
    "3": ROIData(
        well_fov_position="A3",
        inferred_spikes=[
            0.2,
            0.1,
            0.3,
            0.1,
            0.4,
            0.2,
            0.1,
            0.2,
        ],  # No spikes above threshold
        inferred_spikes_threshold=0.5,
        active=True,
    ),
}

# Test the correlation calculation
corr_matrix, roi_indices = _calculate_spike_cross_correlation(test_data)

print("Spike Correlation Analysis Test")
print("=" * 40)
if corr_matrix is not None:
    print(f"Successfully calculated correlation matrix for ROIs: {roi_indices}")
    print(f"Matrix shape: {corr_matrix.shape}")
    print(f"Correlation matrix:\n{corr_matrix}")
    print(f"Max correlation: {np.max(corr_matrix):.3f}")
    print(f"Min correlation: {np.min(corr_matrix):.3f}")
else:
    print("Failed to calculate correlation matrix")

print("\nTest completed successfully!")
