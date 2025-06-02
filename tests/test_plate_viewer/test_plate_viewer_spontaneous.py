from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from cvxpy import pos
import pytest
import useq
from rich import print

from micromanager_gui import PlateViewer
from micromanager_gui._plate_viewer._fov_table import WellInfo
from micromanager_gui._plate_viewer._plate_map import PlateMapData
from micromanager_gui._plate_viewer._to_csv import save_to_csv
from micromanager_gui._plate_viewer._util import ROIData

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


# SPONTANEOUS TEST DATA
TEST_DATA_PATH = (
    Path(__file__).parent / "data" / "spontaneous" / "spont.tensorstore.zarr"
)

TEST_LABELS_PATH = str(Path(__file__).parent / "data" / "spontaneous" / "spont_labels")
TEST_ANALYSIS_PATH = str(
    Path(__file__).parent / "data" / "spontaneous" / "spont_analysis"
)

G_MAP = [PlateMapData(name="B5", row_col=(1, 4), condition=("c1", "indigo"))]
T_MAP = [PlateMapData(name="B5", row_col=(1, 4), condition=("t1", "darkturquoise"))]

SAVE_MAP = {
    "raw_data": ["test_analysis_raw_data.csv"],
    "dff_data": ["test_analysis_dff_data.csv"],
    "dec_dff_data": ["test_analysis_dec_dff_data.csv"],
    "grouped": [
        "test_analysis_amplitude.csv",
        "test_analysis_percentage_active.csv",
        "test_analysis_cell_size.csv",
        "test_analysis_iei.csv",
        "test_analysis_frequency.csv",
        "test_analysis_synchrony.csv",
    ],
}


@pytest.fixture
def dummy_data_loader():
    def fake_generator(*args, **kwargs):
        yield 1  # simulate progress update
        return

    with patch.object(PlateViewer, "_load_and_set_data_from_json", fake_generator):
        yield


def test_plate_viewer_init(qtbot: QtBot, dummy_data_loader) -> None:
    pv = PlateViewer()
    qtbot.addWidget(pv)

    pv.initialize_widget(str(TEST_DATA_PATH), TEST_LABELS_PATH, TEST_ANALYSIS_PATH)

    # data
    assert pv.data is not None
    assert pv.data.store is not None
    assert list(pv.data.store.shape) == [1, 153, 1, 256, 256]
    assert list(pv.data.store.domain.labels) == ["p", "t", "c", "y", "x"]
    # labels and analysis paths
    assert pv.pv_labels_path == TEST_LABELS_PATH
    assert pv.pv_analysis_path == TEST_ANALYSIS_PATH
    # plate view
    assert pv._plate_view.selectedIndices() == ()  # No wells selected
    assert len(pv._plate_view._well_items) == 96  # 96 well plate
    # fov table
    assert pv._fov_table.value() is None  # No FOV selected
    # image viewer
    assert pv._image_viewer._viewer.image is None  # No image loaded
    assert pv._image_viewer._viewer.labels_image is None  # No labels image loaded
    assert pv._image_viewer._viewer.contours_image is None  # No contours image loaded
    # plate map
    assert pv._plate_map_genotype.value() == G_MAP
    assert pv._plate_map_treatment.value() == T_MAP

    # trigger well selection
    with qtbot.wait_signal(pv._plate_view.selectionChanged, timeout=2000):
        pv._plate_view.setSelectedIndices([(1, 4)])  # B5_0000

    fov_val = pv._fov_table.value()
    assert isinstance(fov_val, WellInfo)
    assert fov_val.pos_idx == 0
    assert fov_val.fov.name == "B5_0000"
    assert round(fov_val.fov.x, 2) == -14549.11
    assert round(fov_val.fov.y, 2) == 21805.05

    assert pv._image_viewer._viewer.image is not None  # Image loaded
    assert pv._image_viewer._viewer.labels_image is not None  # Labels image loaded
    assert pv._image_viewer._viewer.contours_image is not None  # Contours image loaded
    assert not pv._image_viewer._viewer.contours_image.visible  # Contours not visible
    # trigger contours visibility
    pv._image_viewer._show_labels(True)
    assert pv._image_viewer._viewer.contours_image.visible  # Contours visible


# ignore futore warnings and userwarning
@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
@pytest.mark.usefixtures("dummy_data_loader")
def test_analysis_code(qtbot: QtBot, dummy_data_loader, tmp_path: Path) -> None:
    pv = PlateViewer()
    qtbot.addWidget(pv)
    # create temporary analysis path and initialize widget
    tmp_analysis_path = tmp_path / "test_analysis/"
    tmp_analysis_path.mkdir(parents=True, exist_ok=True)
    pv.initialize_widget(str(TEST_DATA_PATH), TEST_LABELS_PATH, str(tmp_analysis_path))

    # add plate map
    # fmt: off
    pv._plate_map_genotype.setValue(Path(TEST_ANALYSIS_PATH) / "genotype_plate_map.json")  # noqa: E501
    pv._plate_map_treatment.setValue(Path(TEST_ANALYSIS_PATH) / "treatment_plate_map.json")  # noqa: E501
    assert pv._plate_map_genotype.value() == G_MAP
    assert pv._plate_map_treatment.value() == T_MAP
    # fmt: on

    # autoselect the only 1 position in the plate map
    assert pv._analysis_wdg._prepare_for_running() == [0]

    # save the plate map data
    pv._analysis_wdg._handle_plate_map()

    # trigget analysis code
    pv._analysis_wdg._extract_trace_data_per_position(0)

    # trigger save to csv
    save_to_csv(tmp_analysis_path, pv.pv_analysis_data)

    # assert that the analysis path is created and contains the expected files
    files = [f.name for f in tmp_analysis_path.iterdir() if f.is_file()]

    assert set(files) == {
        "treatment_plate_map.json",
        "genotype_plate_map.json",
        "B5_0000_p0.json",
    }

    # assert that the subfolders are created and contain the expected files
    subfolders = [f.name for f in tmp_analysis_path.iterdir() if f.is_dir()]
    assert set(subfolders) == {"raw_data", "dff_data", "dec_dff_data", "grouped"}
    for dir_name in subfolders:
        dir_path = tmp_analysis_path / dir_name
        assert dir_path.iterdir()
        file_list = [f.name for f in dir_path.iterdir() if f.is_file()]
        assert file_list == SAVE_MAP[dir_name]

    # TODO:
    # - open B5_0000_p0.json as ROIData
    f = tmp_analysis_path / "B5_0000_p0.json"
    with open(f) as file:
        data = json.load(file)
        roi_data = ROIData(**data["1"])
        print(roi_data)
    # - assert that the ROIData object is created successfully
    # - assert that some of the ROIData attributes are as expected
