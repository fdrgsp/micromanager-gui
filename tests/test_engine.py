from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from useq import MDAEvent

from micromanager_gui._engine import ArduinoEngine

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


@pytest.fixture()
def engine(global_mmcore: CMMCorePlus) -> ArduinoEngine:
    return ArduinoEngine(global_mmcore)


# ---------------------------------------------------------------------------
# GCaMP setters
# ---------------------------------------------------------------------------


def test_gcamp_default_state(engine: ArduinoEngine) -> None:
    assert engine._gcamp_channel is None
    assert engine._gcamp_delay_ms == 0.0


def test_gcamp_setters(engine: ArduinoEngine) -> None:
    engine.setGCaMPChannel("FITC")
    assert engine._gcamp_channel == "FITC"

    engine.setGCaMPDelayMs(500.0)
    assert engine._gcamp_delay_ms == 500.0

    engine.setGCaMPChannel(None)
    assert engine._gcamp_channel is None


# ---------------------------------------------------------------------------
# GCaMP shutter delay behaviour in exec_event
# ---------------------------------------------------------------------------


def _run_event(
    engine: ArduinoEngine, event: MDAEvent, channel_config: str
) -> tuple[MagicMock, MagicMock]:
    """Helper: run exec_event with mocked core calls; return (shutter_mock, sleep_mock)."""
    with (
        patch.object(
            engine._mmc, "getCurrentConfig", return_value=channel_config
        ),
        patch.object(engine._mmc, "setShutterOpen") as mock_shutter,
        patch("time.sleep") as mock_sleep,
        patch.object(engine, "exec_single_event", return_value=iter([])),
    ):
        list(engine.exec_event(event))
    return mock_shutter, mock_sleep


def test_gcamp_shutter_opens_on_matching_channel_at_t0(
    engine: ArduinoEngine,
) -> None:
    engine.setGCaMPChannel("FITC")
    engine.setGCaMPDelayMs(200.0)

    shutter, sleep = _run_event(engine, MDAEvent(index={"t": 0}), "FITC")

    shutter.assert_called_once_with(True)
    sleep.assert_any_call(0.2)  # 200 ms → 0.2 s


def test_gcamp_shutter_skipped_wrong_channel(engine: ArduinoEngine) -> None:
    engine.setGCaMPChannel("FITC")
    engine.setGCaMPDelayMs(200.0)

    shutter, _ = _run_event(engine, MDAEvent(index={"t": 0}), "Cy5")

    shutter.assert_not_called()


def test_gcamp_shutter_skipped_at_nonzero_t(engine: ArduinoEngine) -> None:
    engine.setGCaMPChannel("FITC")
    engine.setGCaMPDelayMs(200.0)

    shutter, _ = _run_event(engine, MDAEvent(index={"t": 1}), "FITC")

    shutter.assert_not_called()


def test_gcamp_shutter_skipped_when_delay_is_zero(engine: ArduinoEngine) -> None:
    engine.setGCaMPChannel("FITC")
    engine.setGCaMPDelayMs(0.0)

    shutter, _ = _run_event(engine, MDAEvent(index={"t": 0}), "FITC")

    shutter.assert_not_called()


def test_gcamp_shutter_skipped_when_channel_is_none(engine: ArduinoEngine) -> None:
    engine.setGCaMPDelayMs(200.0)  # channel stays None

    shutter, _ = _run_event(engine, MDAEvent(index={"t": 0}), "FITC")

    shutter.assert_not_called()
