from __future__ import annotations

from typing import TYPE_CHECKING

from micromanager_gui._widgets._stage_control import StagesControlWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_stage_widget(qtbot: QtBot, global_mmcore: CMMCorePlus, _run_after_each_test):
    s = StagesControlWidget(mmcore=global_mmcore)
    qtbot.addWidget(s)
    group1 = s._layout.takeAt(0).widget()
    group2 = s._layout.takeAt(0).widget()
    assert group1.title() == "XY"
    assert group2.title() == "Z"
    global_mmcore.loadSystemConfiguration()
    global_mmcore.loadSystemConfiguration()
