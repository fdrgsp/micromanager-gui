from __future__ import annotations

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import mda_listeners_connected
from useq import MDASequence

from micromanager_gui._writers._tensorstore_zarr import _TensorStoreHandler

core = CMMCorePlus.instance()
core.loadSystemConfiguration()

sequence = MDASequence(
    channels=["DAPI", {"config": "FITC", "exposure": 1}],
    stage_positions=[{"x": 1, "y": 1, "name": "some position"}, {"x": 0, "y": 0}],
    time_plan={"interval": 2, "loops": 3},
    z_plan={"range": 4, "step": 0.5},
    axis_order="tpcz",
)

writer = _TensorStoreHandler(
    path="/example_ts",
    delete_existing=True,
    driver="zarr",
    # Use 2GB in-memory cache.
    spec={
        "context": {"cache_pool": {"total_bytes_limit": 2_000_000_000}},
    },
)

with mda_listeners_connected(writer):
    core.mda.run(sequence)
