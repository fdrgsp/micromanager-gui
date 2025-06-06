from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from micromanager_gui import PlateViewer

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


# NO HCS TEST DATA
TEST_DATA_PATH = Path(__file__).parent / "data" / "no_hcs" / "no_hcs.tensorstore.zarr"


@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
@pytest.mark.usefixtures("dummy_data_loader")
def test_plate_viewer_init_no_hcs(qtbot: QtBot, dummy_data_loader) -> None:
    """Test plate viewer initialization when user clicks 'No' in wizard.

    This test simulates the scenario where:
    1. No existing plate plan is found (no JSON file, no old metadata)
    2. The plate plan wizard is triggered
    3. User clicks "No" (wizard.exec() returns False)
    4. PlateViewer falls back to DEFAULT_PLATE_PLAN (coverslip-18mm-square)
    """
    pv = PlateViewer()
    qtbot.addWidget(pv)

    # Mock the plate plan wizard to return False (simulate "No" click)
    with patch.object(pv._plate_plan_wizard, "exec", return_value=False):
        pv.initialize_widget(str(TEST_DATA_PATH))

        # Verify that the default plate plan flag is set to True
        assert pv._default_plate_plan is True, (
            "Expected _default_plate_plan to be True when wizard returns False"
        )

        # Verify that the DEFAULT_PLATE_PLAN (coverslip-18mm-square) is used
        # Check that the plate viewer has well items (indicating plate loaded)
        assert hasattr(pv._plate_view, "_well_items"), (
            "Expected plate view to have _well_items"
        )
        well_items = pv._plate_view._well_items

        # Verify it's a coverslip plate (1x1 well)
        assert len(well_items) == 1, (
            f"Expected 1 well for coverslip plate, got {len(well_items)}"
        )

        # Verify the well is at position (0, 0) as expected for 1x1 plate
        assert (
            0,
            0,
        ) in well_items, "Expected well at position (0, 0) for coverslip plate"

        # Verify that PlateViewer initializes properly despite using default plan
        assert pv._data is not None, "Expected data to be loaded"
        assert pv._plate_view is not None, "Expected plate view to be initialized"
