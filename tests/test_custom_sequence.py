from __future__ import annotations

from typing import TYPE_CHECKING

import useq
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY

from micromanager_gui._widgets._mda_widget import MDAWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_custom_mda_sequence(qtbot: QtBot, global_mmcore: CMMCorePlus) -> None:
    """Test the custom MDA sequence."""

    wdg = MDAWidget()
    qtbot.addWidget(wdg)
    wdg.show()

    mda = useq.MDASequence(
        stage_positions=[(0, 0, 0), (1, 1, 1)],
        time_plan=useq.TIntervalLoops(interval=0.1, loops=4),
        channels=["Cy5"],
        metadata={PYMMCW_METADATA_KEY: {}},
        axis_order=("p", "t", "c"),
    )

    stimulation_settings = {
        "arduino_port": "/dev/cu.usbmodem1101",
        "arduino_led_pin": "d:3:p",
        "initial_delay": 0,
        "interval": 3,
        "num_pulses": 3,
        "led_start_power": 100,
        "led_power_increment": 0,
        "led_pulse_duration": 100,
        "pulse_on_frame": {0: 100, 2: 100},
    }

    custom_seq = wdg._update_value_with_arduino(mda, stimulation_settings)

    # make sure the new metadata is added
    assert mda.metadata[PYMMCW_METADATA_KEY] == custom_seq.metadata[PYMMCW_METADATA_KEY]
    # make sure the rest of the sequence is the same
    assert mda.replace(metadata={}) == custom_seq.replace(metadata={})

    ev_list = [(ev.index["t"], ev.action.type) for ev in custom_seq]
    assert ev_list == [
        (0, "custom"),
        (0, "acquire_image"),
        (1, "acquire_image"),
        (2, "custom"),
        (2, "acquire_image"),
        (3, "acquire_image"),
        (0, "custom"),
        (0, "acquire_image"),
        (1, "acquire_image"),
        (2, "custom"),
        (2, "acquire_image"),
        (3, "acquire_image"),
    ]

    # add initial delay
    stimulation_settings.update(
        {
            "initial_delay": 2,
            "pulse_on_frame": {2: 100, 4: 100},
        }
    )
    mda = mda.replace(time_plan=useq.TIntervalLoops(interval=0.1, loops=6))
    custom_seq = wdg._update_value_with_arduino(mda, stimulation_settings)
    ev_list = [(ev.index["t"], ev.action.type) for ev in custom_seq]

    assert ev_list == [
        (0, "acquire_image"),
        (1, "acquire_image"),
        (2, "custom"),
        (2, "acquire_image"),
        (3, "acquire_image"),
        (4, "custom"),
        (4, "acquire_image"),
        (5, "acquire_image"),
        (0, "acquire_image"),
        (1, "acquire_image"),
        (2, "custom"),
        (2, "acquire_image"),
        (3, "acquire_image"),
        (4, "custom"),
        (4, "acquire_image"),
        (5, "acquire_image"),
    ]

    # add autofocus (and use previous stimulation_settings with initial_delay)
    mda = mda.replace(
        autofocus_plan=useq.AxesBasedAF(autofocus_motor_offset=2, axes=("p",))
    )
    custom_seq = wdg._update_value_with_arduino(mda, stimulation_settings)
    ev_list = [(ev.index["t"], ev.action.type) for ev in custom_seq]

    assert ev_list == [
        (0, "hardware_autofocus"),
        (0, "acquire_image"),
        (1, "acquire_image"),
        (2, "custom"),
        (2, "acquire_image"),
        (3, "acquire_image"),
        (4, "custom"),
        (4, "acquire_image"),
        (5, "acquire_image"),
        (0, "hardware_autofocus"),
        (0, "acquire_image"),
        (1, "acquire_image"),
        (2, "custom"),
        (2, "acquire_image"),
        (3, "acquire_image"),
        (4, "custom"),
        (4, "acquire_image"),
        (5, "acquire_image"),
    ]
