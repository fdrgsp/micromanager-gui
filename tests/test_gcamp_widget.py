from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from micromanager_gui._widgets._mda_widget._gcamp_delay_widget import (
    GCaMPDelayDialog,
    GCaMPDelayWidget,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


# ---------------------------------------------------------------------------
# GCaMPDelayWidget (outer groupbox)
# ---------------------------------------------------------------------------


def test_widget_default_state(qtbot: QtBot) -> None:
    wdg = GCaMPDelayWidget()
    qtbot.addWidget(wdg)

    assert not wdg._enable.isChecked()
    assert not wdg._settings_btn.isEnabled()
    assert wdg.value() is None


def test_widget_enable_shows_button(qtbot: QtBot) -> None:
    wdg = GCaMPDelayWidget()
    qtbot.addWidget(wdg)

    wdg._enable.setChecked(True)

    assert wdg._settings_btn.isEnabled()


def test_widget_disable_hides_dialog(qtbot: QtBot) -> None:
    wdg = GCaMPDelayWidget()
    qtbot.addWidget(wdg)

    wdg._enable.setChecked(True)
    wdg._dialog.show()
    assert wdg._dialog.isVisible()

    wdg._enable.setChecked(False)
    assert not wdg._dialog.isVisible()


def test_widget_value_when_enabled(qtbot: QtBot) -> None:
    wdg = GCaMPDelayWidget()
    qtbot.addWidget(wdg)
    wdg._dialog._channel_combo.addItems(["FITC", "Rhodamine", "DAPI", "Cy5"])

    wdg._enable.setChecked(True)
    wdg._dialog._channel_combo.setCurrentText("FITC")
    wdg._dialog._delay_spin.setValue(300.0)

    val = wdg.value()
    assert val is not None
    assert val["channel"] == "FITC"
    assert val["delay_ms"] == 300.0


def test_widget_set_value(qtbot: QtBot) -> None:
    wdg = GCaMPDelayWidget()
    qtbot.addWidget(wdg)
    wdg._dialog._channel_combo.addItems(["FITC", "Rhodamine", "DAPI", "Cy5"])

    wdg.setValue({"channel": "Cy5", "delay_ms": 500.0})

    assert wdg._enable.isChecked()
    val = wdg.value()
    assert val is not None
    assert val["channel"] == "Cy5"
    assert val["delay_ms"] == 500.0


# ---------------------------------------------------------------------------
# GCaMPDelayDialog (settings dialog)
# ---------------------------------------------------------------------------


def test_dialog_default_state(qtbot: QtBot) -> None:
    dlg = GCaMPDelayDialog()
    qtbot.addWidget(dlg)

    assert dlg._delay_spin.value() == 0.0
    assert dlg._channel_combo.currentText() == ""


def test_dialog_value_and_set_value(qtbot: QtBot) -> None:
    dlg = GCaMPDelayDialog()
    qtbot.addWidget(dlg)
    dlg._channel_combo.addItems(["FITC", "Rhodamine", "DAPI", "Cy5"])

    dlg._channel_combo.setCurrentText("Rhodamine")
    dlg._delay_spin.setValue(150.0)

    assert dlg.value() == {"channel": "Rhodamine", "delay_ms": 150.0}


def test_dialog_set_value_roundtrip(qtbot: QtBot) -> None:
    dlg = GCaMPDelayDialog()
    qtbot.addWidget(dlg)
    dlg._channel_combo.addItems(["FITC", "Rhodamine", "DAPI", "Cy5"])

    dlg.setValue({"channel": "DAPI", "delay_ms": 800.0})

    assert dlg.value() == {"channel": "DAPI", "delay_ms": 800.0}


def test_dialog_populates_channels_from_mmcore(
    qtbot: QtBot, global_mmcore: CMMCorePlus
) -> None:
    """Channel combo is populated from the core's channel group."""
    dlg = GCaMPDelayDialog(mmcore=global_mmcore)
    qtbot.addWidget(dlg)

    # test config has a "Channel" group with FITC, Rhodamine, DAPI, Cy5
    channels = [
        dlg._channel_combo.itemText(i) for i in range(dlg._channel_combo.count())
    ]
    assert len(channels) > 0
    assert "Cy5" in channels


def test_dialog_repopulates_on_sys_cfg_loaded(
    qtbot: QtBot, global_mmcore: CMMCorePlus
) -> None:
    """Channel combo refreshes when a new system configuration is loaded."""
    dlg = GCaMPDelayDialog(mmcore=global_mmcore)
    qtbot.addWidget(dlg)

    count_before = dlg._channel_combo.count()
    global_mmcore.events.systemConfigurationLoaded.emit()
    assert dlg._channel_combo.count() == count_before
