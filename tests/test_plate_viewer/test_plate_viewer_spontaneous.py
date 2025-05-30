from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import useq

from micromanager_gui import PlateViewer
from micromanager_gui._plate_viewer._fov_table import WellInfo
from micromanager_gui._plate_viewer._plate_map import PlateMapData
from micromanager_gui._plate_viewer._to_csv import save_to_csv

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


# TO CHANGE
TEST_DATA_PATH = "/Volumes/T7 Shield/neurons/SSADH_Fam005_CC240711_IG_NC_PlateC_240925_GCaMP6s/SSADH_Fam005_CC240711_IG_NC_PlateC_240925_GCaMP6s.tensorstore.zarr"  # noqa: E501

# CHANGE CONTENT IN data/spontaneous/...
TEST_LABELS_PATH = str(Path(__file__).parent / "data" / "spontaneous" / "labels")
TEST_ANALYSIS_PATH = str(Path(__file__).parent / "data" / "spontaneous" / "analysis")

# TO CHANGE
G_MAP = [
    PlateMapData(name="B7", row_col=(1, 6), condition=("g1", "orchid")),
    PlateMapData(name="B8", row_col=(1, 7), condition=("g1", "orchid")),
    PlateMapData(name="B9", row_col=(1, 8), condition=("g1", "orchid")),
    PlateMapData(name="B10", row_col=(1, 9), condition=("g1", "orchid")),
    PlateMapData(name="B11", row_col=(1, 10), condition=("g1", "orchid")),
    PlateMapData(name="C7", row_col=(2, 6), condition=("g1", "orchid")),
    PlateMapData(name="C8", row_col=(2, 7), condition=("g1", "orchid")),
    PlateMapData(name="C9", row_col=(2, 8), condition=("g1", "orchid")),
    PlateMapData(name="C10", row_col=(2, 9), condition=("g1", "orchid")),
    PlateMapData(name="C11", row_col=(2, 10), condition=("g1", "orchid")),
    PlateMapData(name="F7", row_col=(5, 6), condition=("g2", "chartreuse")),
    PlateMapData(name="F8", row_col=(5, 7), condition=("g2", "chartreuse")),
    PlateMapData(name="F9", row_col=(5, 8), condition=("g2", "chartreuse")),
    PlateMapData(name="F10", row_col=(5, 9), condition=("g2", "chartreuse")),
    PlateMapData(name="F11", row_col=(5, 10), condition=("g2", "chartreuse")),
    PlateMapData(name="G7", row_col=(6, 6), condition=("g2", "chartreuse")),
    PlateMapData(name="G8", row_col=(6, 7), condition=("g2", "chartreuse")),
    PlateMapData(name="G9", row_col=(6, 8), condition=("g2", "chartreuse")),
    PlateMapData(name="G10", row_col=(6, 9), condition=("g2", "chartreuse")),
    PlateMapData(name="G11", row_col=(6, 10), condition=("g2", "chartreuse")),
]
# TO CHANGE
T_MAP = [
    PlateMapData(name="B7", row_col=(1, 6), condition=("t1", "darkturquoise")),
    PlateMapData(name="B8", row_col=(1, 7), condition=("t1", "darkturquoise")),
    PlateMapData(name="B9", row_col=(1, 8), condition=("t2", "purple")),
    PlateMapData(name="B10", row_col=(1, 9), condition=("t2", "purple")),
    PlateMapData(name="B11", row_col=(1, 10), condition=("ctrl", "navy")),
    PlateMapData(name="C7", row_col=(2, 6), condition=("t1", "darkturquoise")),
    PlateMapData(name="C8", row_col=(2, 7), condition=("t1", "darkturquoise")),
    PlateMapData(name="C9", row_col=(2, 8), condition=("t2", "purple")),
    PlateMapData(name="C10", row_col=(2, 9), condition=("t2", "purple")),
    PlateMapData(name="C11", row_col=(2, 10), condition=("ctrl", "navy")),
    PlateMapData(name="F7", row_col=(5, 6), condition=("t1", "darkturquoise")),
    PlateMapData(name="F8", row_col=(5, 7), condition=("t1", "darkturquoise")),
    PlateMapData(name="F9", row_col=(5, 8), condition=("t2", "purple")),
    PlateMapData(name="F10", row_col=(5, 9), condition=("t2", "purple")),
    PlateMapData(name="F11", row_col=(5, 10), condition=("ctrl", "navy")),
    PlateMapData(name="G7", row_col=(6, 6), condition=("t1", "darkturquoise")),
    PlateMapData(name="G8", row_col=(6, 7), condition=("t1", "darkturquoise")),
    PlateMapData(name="G9", row_col=(6, 8), condition=("t2", "purple")),
    PlateMapData(name="G10", row_col=(6, 9), condition=("t2", "purple")),
    PlateMapData(name="G11", row_col=(6, 10), condition=("ctrl", "navy")),
]

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


@pytest.mark.skip(reason="Test data to be acquired")
def test_plate_viewer_init(qtbot: QtBot, dummy_data_loader) -> None:
    pv = PlateViewer()
    qtbot.addWidget(pv)

    pv.initialize_widget(TEST_DATA_PATH, TEST_LABELS_PATH, TEST_ANALYSIS_PATH)

    # data
    assert pv.data is not None
    assert pv.data.store is not None
    assert list(pv.data.store.shape) == [80, 800, 1, 1024, 1024]  # TO CHANGE
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
        pv._plate_view.setSelectedIndices([(1, 6)])  # B7_0000, TO CHANGE

    assert pv._fov_table.value() == WellInfo(  # TO CHANGE
        pos_idx=0, fov=useq.AbsolutePosition(x=4193.99, y=22857.6, name="B7_0000")
    )
    assert pv._image_viewer._viewer.image is not None  # Image loaded
    assert pv._image_viewer._viewer.labels_image is not None  # Labels image loaded
    assert pv._image_viewer._viewer.contours_image is not None  # Contours image loaded
    assert not pv._image_viewer._viewer.contours_image.visible  # Contours not visible
    # trigger contours visibility
    pv._image_viewer._show_labels(True)
    assert pv._image_viewer._viewer.contours_image.visible  # Contours visible


@pytest.mark.skip(reason="Test data to be acquired")
@pytest.mark.filterwarnings("ignore:.*multidimensional input.*:FutureWarning")  # oasis
def test_analysis_code(qtbot: QtBot, dummy_data_loader, tmp_path: Path) -> None:
    pv = PlateViewer()
    qtbot.addWidget(pv)
    # create temporary analysis path and initialize widget
    tmp_analysis_path = tmp_path / "test_analysis/"
    tmp_analysis_path.mkdir(parents=True, exist_ok=True)
    pv.initialize_widget(TEST_DATA_PATH, TEST_LABELS_PATH, str(tmp_analysis_path))

    # add plate map
    # fmt: off
    pv._plate_map_genotype.setValue(Path(TEST_ANALYSIS_PATH) / "genotype_plate_map.json")  # noqa: E501
    pv._plate_map_treatment.setValue(Path(TEST_ANALYSIS_PATH) / "treatment_plate_map.json")  # noqa: E501
    assert pv._plate_map_genotype.value() == G_MAP
    assert pv._plate_map_treatment.value() == T_MAP
    # fmt: on

    # autoselect the only 2 positions in the plate map
    assert pv._analysis_wdg._prepare_for_running() == [0, 1]

    # save the plate map data
    pv._analysis_wdg._handle_plate_map()

    # trigget analysis code
    pv._analysis_wdg._extract_trace_data_per_position(0)

    # trigger save to csv
    save_to_csv(tmp_analysis_path, pv.pv_analysis_data)

    # assert that the analysis path is created and contains the expected files
    files = [f.name for f in tmp_analysis_path.iterdir() if f.is_file()]
    assert files == [
        "treatment_plate_map.json",
        "B7_0000_p0.json",
        "genotype_plate_map.json",
    ]

    # assert that the subfolders are created and contain the expected files
    subfolders = [f.name for f in tmp_analysis_path.iterdir() if f.is_dir()]
    assert subfolders == ["raw_data", "dff_data", "dec_dff_data", "grouped"]
    for dir_name in subfolders:
        dir_path = tmp_analysis_path / dir_name
        assert dir_path.iterdir()
        file_list = [f.name for f in dir_path.iterdir() if f.is_file()]
        assert file_list == SAVE_MAP[dir_name]

    # TODO:
    # - open B7_0000_p0.json as ROIData
    # - assert that some of the ROIData attributes are as expected
