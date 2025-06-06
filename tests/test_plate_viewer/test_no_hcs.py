from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import useq

from micromanager_gui import PlateViewer
from micromanager_gui._plate_viewer._util import PLATE_PLAN, SETTINGS_PATH

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
        assert (
            pv._default_plate_plan is True
        ), "Expected _default_plate_plan to be True when wizard returns False"

        # Verify that the DEFAULT_PLATE_PLAN (coverslip-18mm-square) is used
        # Check that the plate viewer has well items (indicating plate loaded)
        assert hasattr(
            pv._plate_view, "_well_items"
        ), "Expected plate view to have _well_items"
        well_items = pv._plate_view._well_items

        # Verify it's a coverslip plate (1x1 well)
        assert (
            len(well_items) == 1
        ), f"Expected 1 well for coverslip plate, got {len(well_items)}"

        # Verify the well is at position (0, 0) as expected for 1x1 plate
        assert (
            0,
            0,
        ) in well_items, "Expected well at position (0, 0) for coverslip plate"

        # Verify that PlateViewer initializes properly despite using default plan
        assert pv._data is not None, "Expected data to be loaded"
        assert pv._plate_view is not None, "Expected plate view to be initialized"


@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
@pytest.mark.usefixtures("dummy_data_loader")
def test_save_plate_plan_json_no_analysis_path(qtbot, dummy_data_loader) -> None:
    """Test _save_plate_plan_json when analysis path is not set."""
    pv = PlateViewer()
    qtbot.addWidget(pv)

    # Ensure analysis path is None
    pv._pv_analysis_path = None

    result = pv._load_plate_plan_from_json()

    assert result is None

    # Create a plate plan
    plate = useq.WellPlate(
        name="test",
        rows=1,
        columns=1,
        well_spacing=(1.0, 1.0),
        well_size=(1.0, 1.0),
    )
    plate_plan = useq.WellPlatePlan(
        plate=plate, a1_center_xy=(0.0, 0.0), selected_wells=((0,), (0,))
    )

    # Should return early without error
    pv._save_plate_plan_json(plate_plan)
    # No exception should be raised, method should simply return


@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
@pytest.mark.usefixtures("dummy_data_loader")
def test_load_plate_plan_from_json(qtbot, dummy_data_loader, tmp_path) -> None:
    """Test _load_plate_plan_from_json and _save_plate_plan_json methods."""
    pv = PlateViewer()
    qtbot.addWidget(pv)

    # Test 1: Valid file loading
    analysis_path1 = tmp_path / "analysis1"
    analysis_path1.mkdir()
    settings_file1 = analysis_path1 / SETTINGS_PATH

    # Create a valid plate plan
    plate = useq.WellPlate(
        name="test-plate",
        rows=8,
        columns=12,
        well_spacing=(9.0, 9.0),
        well_size=(6.0, 6.0),
    )
    plate_plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0.0, 0.0),
        selected_wells=((0, 4), (1, 5)),  # wells (0,1) and (4,5)
    )

    # Write valid plate plan to settings.json
    settings_data = {PLATE_PLAN: plate_plan.model_dump()}
    with open(settings_file1, "w") as f:
        json.dump(settings_data, f)

    # Set analysis path and test loading
    pv._pv_analysis_path = str(analysis_path1)
    result = pv._load_plate_plan_from_json()

    assert result is not None
    assert isinstance(result, useq.WellPlatePlan)
    assert result.plate.name == "test-plate"
    assert result.plate.rows == 8
    assert result.plate.columns == 12
    assert result.selected_wells == ((0, 4), (1, 5))

    # Test 2: No settings file
    analysis_path2 = tmp_path / "analysis2"
    analysis_path2.mkdir()
    pv._pv_analysis_path = str(analysis_path2)

    result = pv._load_plate_plan_from_json()
    assert result is None

    # Test 3: Malformed JSON
    analysis_path3 = tmp_path / "analysis3"
    analysis_path3.mkdir()
    settings_file3 = analysis_path3 / SETTINGS_PATH

    # Write malformed JSON
    with open(settings_file3, "w") as f:
        f.write("{ invalid json content")

    pv._pv_analysis_path = str(analysis_path3)

    # Should return None and log warning
    result = pv._load_plate_plan_from_json()
    assert result is None

    # Test 4: Missing plate_plan key
    analysis_path4 = tmp_path / "analysis4"
    analysis_path4.mkdir()
    settings_file4 = analysis_path4 / SETTINGS_PATH

    # Write settings without plate_plan key
    settings_data = {"other_setting": "value"}
    with open(settings_file4, "w") as f:
        json.dump(settings_data, f)

    pv._pv_analysis_path = str(analysis_path4)
    result = pv._load_plate_plan_from_json()

    assert result is None

    # Test 5: Null plate_plan
    analysis_path5 = tmp_path / "analysis5"
    analysis_path5.mkdir()
    settings_file5 = analysis_path5 / SETTINGS_PATH

    # Write settings with null plate_plan
    settings_data = {PLATE_PLAN: None}
    with open(settings_file5, "w") as f:
        json.dump(settings_data, f)

    pv._pv_analysis_path = str(analysis_path5)
    result = pv._load_plate_plan_from_json()

    assert result is None

    # Test 6: Invalid plate plan data
    analysis_path6 = tmp_path / "analysis6"
    analysis_path6.mkdir()
    settings_file6 = analysis_path6 / SETTINGS_PATH

    # Write settings with invalid plate plan data
    settings_data = {PLATE_PLAN: {"invalid": "data", "missing": "required_fields"}}
    with open(settings_file6, "w") as f:
        json.dump(settings_data, f)

    pv._pv_analysis_path = str(analysis_path6)

    # Should return None and log warning due to ValidationError
    result = pv._load_plate_plan_from_json()
    assert result is None

    # Test 7: Successful save
    analysis_path7 = tmp_path / "analysis7"
    analysis_path7.mkdir()
    pv._pv_analysis_path = str(analysis_path7)

    # Create a plate plan to save
    plate = useq.WellPlate(
        name="save-test-plate",
        rows=6,
        columns=8,
        well_spacing=(9.0, 9.0),
        well_size=(6.0, 6.0),
    )
    plate_plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(1.0, 2.0),
        selected_wells=((0, 1, 2), (0, 1, 2)),  # wells (0,0), (1,1), (2,2)
    )

    # Save the plate plan
    pv._save_plate_plan_json(plate_plan)

    # Verify the file was created and contains correct data
    settings_file7 = analysis_path7 / SETTINGS_PATH
    assert settings_file7.exists()

    with open(settings_file7) as f:
        saved_data = json.load(f)

    assert PLATE_PLAN in saved_data

    # Verify the saved data can be loaded back as a valid WellPlatePlan
    loaded_plate_plan = useq.WellPlatePlan.model_validate(saved_data[PLATE_PLAN])
    assert loaded_plate_plan.plate.name == "save-test-plate"
    assert loaded_plate_plan.plate.rows == 6
    assert loaded_plate_plan.plate.columns == 8
    assert loaded_plate_plan.a1_center_xy == (1.0, 2.0)
    assert loaded_plate_plan.selected_wells == ((0, 1, 2), (0, 1, 2))

    # Test 8: Save with file permission error
    pv._pv_analysis_path = str(tmp_path / "nonexistent_dir")

    # Create a plate plan
    plate = useq.WellPlate(
        name="test",
        rows=1,
        columns=1,
        well_spacing=(1.0, 1.0),
        well_size=(1.0, 1.0),
    )
    plate_plan = useq.WellPlatePlan(
        plate=plate, a1_center_xy=(0.0, 0.0), selected_wells=((0,), (0,))
    )

    # Should handle the OSError gracefully and log error
    pv._save_plate_plan_json(plate_plan)
    # No exception should be raised, error should be logged

    # Test 9: Overwrite existing settings
    analysis_path9 = tmp_path / "analysis9"
    analysis_path9.mkdir()
    settings_file9 = analysis_path9 / SETTINGS_PATH
    pv._pv_analysis_path = str(analysis_path9)

    # Create existing settings file with different content
    existing_data = {"existing_key": "existing_value"}
    with open(settings_file9, "w") as f:
        json.dump(existing_data, f)

    # Create and save new plate plan
    plate = useq.WellPlate(
        name="new-plate",
        rows=2,
        columns=3,
        well_spacing=(5.0, 5.0),
        well_size=(3.0, 3.0),
    )
    plate_plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0.5, 1.5),
        selected_wells=((0, 1), (0, 2)),  # wells (0,0) and (1,2)
    )

    pv._save_plate_plan_json(plate_plan)

    # Verify file was updated with new plate plan while preserving existing settings
    with open(settings_file9) as f:
        saved_data = json.load(f)

    # Should contain both existing data and plate_plan
    assert "existing_key" in saved_data
    assert saved_data["existing_key"] == "existing_value"
    assert PLATE_PLAN in saved_data

    # Verify the plate plan data is correct
    loaded_plate_plan = useq.WellPlatePlan.model_validate(saved_data[PLATE_PLAN])
    assert loaded_plate_plan.plate.name == "new-plate"
    assert loaded_plate_plan.plate.rows == 2
    assert loaded_plate_plan.plate.columns == 3

    # Test 10: Round-trip save and load
    analysis_path10 = tmp_path / "analysis10"
    analysis_path10.mkdir()
    pv._pv_analysis_path = str(analysis_path10)

    # Create original plate plan with complex data
    plate = useq.WellPlate(
        name="round-trip-test",
        rows=8,
        columns=12,
        well_spacing=(9.0, 9.0),
        well_size=(6.85, 6.85),
        circular_wells=True,
    )
    original_plate_plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(14.38, 11.24),
        selected_wells=((0, 0, 1, 1, 7), (0, 1, 0, 1, 11)),  # multiple wells
    )

    # Save the plate plan
    pv._save_plate_plan_json(original_plate_plan)

    # Load it back
    loaded_plate_plan = pv._load_plate_plan_from_json()

    # Verify all data is preserved
    assert loaded_plate_plan is not None
    assert loaded_plate_plan.plate.name == original_plate_plan.plate.name
    assert loaded_plate_plan.plate.rows == original_plate_plan.plate.rows
    assert loaded_plate_plan.plate.columns == original_plate_plan.plate.columns
    assert (
        loaded_plate_plan.plate.well_spacing == original_plate_plan.plate.well_spacing
    )
    assert loaded_plate_plan.plate.well_size == original_plate_plan.plate.well_size
    assert (
        loaded_plate_plan.plate.circular_wells
        == original_plate_plan.plate.circular_wells
    )
    assert loaded_plate_plan.a1_center_xy == original_plate_plan.a1_center_xy
    assert loaded_plate_plan.selected_wells == original_plate_plan.selected_wells
