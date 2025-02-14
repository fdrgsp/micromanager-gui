import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pyfirmata2 import Arduino, Pin
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda.handlers import (
    ImageSequenceWriter,
    OMETiffWriter,
    OMEZarrWriter,
    TensorStoreHandler,
)
from pymmcore_widgets import MDAWidget
from pymmcore_widgets.useq_widgets._mda_sequence import PYMMCW_METADATA_KEY
from qtpy.QtWidgets import QBoxLayout, QMessageBox, QWidget
from useq import CustomAction, MDAEvent, MDASequence

from micromanager_gui._writers import (
    _OMETiffWriter,
    _TensorStoreHandler,
    _TiffSequenceWriter,
)

from ._arduino import ArduinoLedWidget
from ._save_widget import (
    OME_TIFF,
    OME_ZARR,
    WRITERS,
    ZARR_TESNSORSTORE,
    SaveGroupBox,
)

if TYPE_CHECKING:
    from micromanager_gui._mmcore_engine._engine import ArduinoEngine

NUM_SPLIT = re.compile(r"(.*?)(?:_(\d{3,}))?$")
OME_TIFFS = tuple(WRITERS[OME_TIFF])
GB_CACHE = 2_000_000_000  # 2 GB for tensorstore cache
STIMULATION = "stimulation"
CRITICAL_MSG = (
    "'Arduino LED Stimulation' is selected but an error occurred while trying "
    "to communicate with the Arduino. \nPlease, verify that the device is "
    "connected and try again."
)
POWER_EXCEEDED_MSG = (
    "The maximum power of the LED has been exceeded. \nPlease, reduce "
    "the power and try again."
)


def get_next_available_path(requested_path: Path | str, min_digits: int = 3) -> Path:
    """Get the next available paths (filepath or folderpath if extension = "").

    This method adds a counter of min_digits to the filename or foldername to ensure
    that the path is unique.

    Parameters
    ----------
    requested_path : Path | str
        A path to a file or folder that may or may not exist.
    min_digits : int, optional
        The min_digits number of digits to be used for the counter. By default, 3.
    """
    if isinstance(requested_path, str):  # pragma: no cover
        requested_path = Path(requested_path)

    directory = requested_path.parent
    extension = requested_path.suffix
    # ome files like .ome.tiff or .ome.zarr are special,treated as a single extension
    if (stem := requested_path.stem).endswith(".ome"):
        extension = f".ome{extension}"
        stem = stem[:-4]
    # NOTE: added in micromanager_gui ---------------------------------------------
    elif (stem := requested_path.stem).endswith(".tensorstore"):
        extension = f".tensorstore{extension}"
        stem = stem[:-12]
    # -----------------------------------------------------------------------------

    # look for ANY existing files in the folder that follow the pattern of
    # stem_###.extension
    current_max = 0
    for existing in directory.glob(f"*{extension}"):
        # cannot use existing.stem because of the ome (2-part-extension) special case
        base = existing.name.replace(extension, "")
        # if base name ends with a number and stem is the same, increase current_max
        if (
            (match := NUM_SPLIT.match(base))
            and (num := match.group(2))
            # NOTE: added in micromanager_gui -------------------------------------
            # this breaks pymmcore_widgets test_get_next_available_paths_special_cases
            and match.group(1) == stem
            # ---------------------------------------------------------------------
        ):
            current_max = max(int(num), current_max)
            # if it has more digits than expected, update the ndigits
            if len(num) > min_digits:
                min_digits = len(num)
    # if the path does not exist and there are no existing files,
    # return the requested path
    if not requested_path.exists() and current_max == 0:
        return requested_path

    current_max += 1
    # otherwise return the next path greater than the current_max
    # remove any existing counter from the stem
    if match := NUM_SPLIT.match(stem):
        stem, num = match.groups()
        if num:
            # if the requested path has a counter that is greater than any other files
            # use it
            current_max = max(int(num), current_max)
    return directory / f"{stem}_{current_max:0{min_digits}d}{extension}"


class CustomMDASequence(MDASequence):
    """A subclass of `useq.MDASequence`.

    The particularity of this class is that it has an events attribute that is a list
    of `useq.MDAEvent`. If this attribute is empty, the parent __iter__ method is called.
    Otherwise, the events attribute is iterated instead of the MDASequence.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Bypass Pydantic's frozen model restriction
        object.__setattr__(self, "events", [])

    def __iter__(self) -> Iterable[MDAEvent]:
        """Iterate over the events in the sequence.

        If the events attribute is empty, the parent __iter__ method is called.
        """
        return iter(self.events) if self.events else super().__iter__()

    def events(self) -> list[MDAEvent]:
        """Return the events."""
        return self.events

    def clear_events(self) -> None:
        """Clear the events."""
        object.__setattr__(self, "events", [])

    def add_events(self, event: MDAEvent | list[MDAEvent]) -> None:
        """Add an event to the sequence."""
        if isinstance(event, list):
            object.__setattr__(self, "events", self.events + event)
        else:
            object.__setattr__(self, "events", [*self.events, event])


class MDAWidget_(MDAWidget):
    """Multi-dimensional acquisition widget."""

    def __init__(
        self, *, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)

        # writer for saving the MDA sequence. This is used by the MDAViewer to set its
        # internal datastore. If _writer is None, the MDAViewer will use its default
        # internal datastore.
        self.writer: OMETiffWriter | OMEZarrWriter | TensorStoreHandler | None = None

        main_layout = cast(QBoxLayout, self.layout())

        # remove the existing save_info widget from the layout and replace it with
        # the custom SaveGroupBox widget that also handles tensorstore-zarr
        if hasattr(self, "save_info"):
            self.save_info.valueChanged.disconnect(self.valueChanged)
            main_layout.removeWidget(self.save_info)
            self.save_info.deleteLater()
        self.save_info: SaveGroupBox = SaveGroupBox(parent=self)
        self.save_info.valueChanged.connect(self.valueChanged)
        main_layout.insertWidget(0, self.save_info)

        # ------------ Arduino -------------------------------
        self._arduino_led_wdg = ArduinoLedWidget(self)
        main_layout.insertWidget(4, self._arduino_led_wdg)
        # ----------------------------------------------------

    def value(self) -> MDASequence:
        """Set the current state of the widget from a [`useq.MDASequence`][]."""
        val = super().value()

        arduino_settings = self._arduino_led_wdg.value()
        if not arduino_settings:
            return val

        meta = val.metadata.get(PYMMCW_METADATA_KEY, {})
        meta[STIMULATION] = arduino_settings
        val_with_stim = CustomMDASequence(**val.model_dump())

        # TODO: if stimulation is selected and there are multiple positions but the
        # axis order is not starting with 'p', raise a warning message

        pulse_on_frame = arduino_settings.get("pulse_on_frame", {})
        duration = arduino_settings.get("led_pulse_duration", None)
        initial_delay = arduino_settings.get("initial_delay", 0)

        pos_lists = self._group_by_position(list(val))
        for idx, pos_list in enumerate(pos_lists):
            # copy the first event of the position list. If we have an initial delay,
            # copy the event at the index of the delay
            stim_event = pos_list[initial_delay if idx == 0 else 0].model_copy()
            # get if the first event is an autofocus event
            has_af = stim_event.action.type == "hardware_autofocus"
            for pulse_on, power in pulse_on_frame.items():
                stim_event = stim_event.replace(
                    action=CustomAction(
                        name="arduino_stimulation",
                        data={"led_power": power, "led_pulse_duration": duration},
                    ),
                )
                # if the first event is an autofocus event, insert the stimulation event
                # after the autofocus event. we are also considering the initial delay
                if initial_delay:
                    i = pulse_on + 2 if has_af else pulse_on + 1
                else:
                    i = pulse_on + 1 if has_af else pulse_on
                pos_list.insert(i, stim_event)

        # concatenate the list of lists into a single list
        val_with_stim.add_events(
            [event for pos_list in pos_lists for event in pos_list]
        )
        return val_with_stim

    def setValue(self, sequence: MDASequence) -> None:
        """Set the current state of the widget from a [`useq.MDASequence`][]."""
        super().setValue(sequence)
        meta: dict = sequence.metadata.get(PYMMCW_METADATA_KEY, {})
        # if stimulation is in the metadata, set the Arduino LED widget
        if stim := meta.get(STIMULATION):
            self._arduino_led_wdg.setValue(stim)

    def get_next_available_path(self, requested_path: Path) -> Path:
        """Get the next available path.

        Overwrites the method in the parent class to use the custom
        'get_next_available_path' function.
        """
        return get_next_available_path(requested_path=requested_path)

    def prepare_mda(
        self,
    ) -> (
        bool
        | OMEZarrWriter
        | OMETiffWriter
        | TensorStoreHandler
        | ImageSequenceWriter
        | None
    ):
        """Prepare the MDA sequence experiment.

        This method sets the writer to use for saving the MDA sequence.
        """
        # in case the user does not press enter after editing the save name.
        self.save_info.save_name.editingFinished.emit()

        # if autofocus has been requested, but the autofocus device is not engaged,
        # and position-specific offsets haven't been set, show a warning
        pos = self.stage_positions
        if (
            self.af_axis.value()
            and not self._mmc.isContinuousFocusLocked()
            and (not self.tab_wdg.isChecked(pos) or not pos.af_per_position.isChecked())
            and not self._confirm_af_intentions()
        ):
            return False

        # Arduino checks --------------------------------------------------------------
        # hide the Arduino LED control widget if visible
        self._arduino_led_wdg._arduino_led_control.hide()
        if not self._arduino_led_wdg.isChecked():
            self._set_arduino_props(None, None)
        else:
            # check if the Arduino and the LED pin are available
            arduino = self._arduino_led_wdg.board()
            led = self._arduino_led_wdg.ledPin()
            if arduino is None or led is None or not self._test_arduino_connection(led):
                self._set_arduino_props(None, None)
                self._arduino_led_wdg._arduino_led_control._enable(False)
                self._show_critical_led_message(CRITICAL_MSG)
                return False

            # check if power exceeded
            if self._arduino_led_wdg.is_max_power_exceeded():
                self._set_arduino_props(None, None)
                self._show_critical_led_message(POWER_EXCEEDED_MSG)
                return False

            # enable the Arduino board and the LED pin in the MDA engine
            self._set_arduino_props(arduino, led)
        # -----------------------------------------------------------------------------

        sequence = self.value()

        # technically, this is in the metadata as well, but isChecked is more direct
        if self.save_info.isChecked():
            save_path = self._update_save_path_from_metadata(
                sequence, update_metadata=True
            )
            if isinstance(save_path, Path):
                # get save format from metadata
                save_meta = sequence.metadata.get(PYMMCW_METADATA_KEY, {})
                save_format = save_meta.get("format")
                # set the writer to use for saving the MDA sequence.
                # NOTE: 'self._writer' is used by the 'MDAViewer' to set its datastore
                self.writer = self._create_writer(save_format, save_path)
                # at this point, if self.writer is None, it means that a
                # ImageSequenceWriter should be used to save the sequence.
                if self.writer is None:
                    # Since any other type of writer will be handled by the 'MDAViewer',
                    # we need to pass a writer to the engine only if it is a
                    # 'ImageSequenceWriter'.
                    return _TiffSequenceWriter(save_path)
        return None

    def run_mda(self) -> None:
        """Run the MDA experiment."""
        save_path = self.prepare_mda()
        if save_path is False:
            return
        self.execute_mda(save_path)

    def execute_mda(self, output: Path | str | object | None) -> None:
        """Execute the MDA experiment corresponding to the current value."""
        sequence = self.value()
        # run the MDA experiment asynchronously
        self._mmc.run_mda(sequence, output=output)

    # ------------------- private Methods ----------------------

    def _group_by_position(self, events: list[MDAEvent]) -> list[list[MDAEvent]]:
        """Group the MDA events by position."""
        grouped_events = defaultdict(list)
        for event in events:
            pos_index = event.index.get("p")
            grouped_events[pos_index].append(event)
        return list(grouped_events.values())

    def _set_arduino_props(self, arduino: Arduino | None, led: Pin | None) -> None:
        """Enable the Arduino board and the LED pin in the MDA engine."""
        if not self._mmc.mda.engine:
            return

        # this can only work if using our custom ArduinoEngine
        if not hasattr(self._mmc.mda.engine, "setArduinoBoard") or not hasattr(
            self._mmc.mda.engine, "setArduinoLedPin"
        ):
            return

        engine: ArduinoEngine = self._mmc.mda.engine
        engine.setArduinoBoard(arduino)
        engine.setArduinoLedPin(led)

    def _test_arduino_connection(self, led: Pin) -> bool:
        """Test the connection with the Arduino."""
        try:
            led.write(0.0)
            return True
        except Exception:
            return False

    def _show_critical_led_message(self, msg: str) -> None:
        QMessageBox.critical(self, "Arduino Error", msg, QMessageBox.StandardButton.Ok)
        return

    def _on_mda_finished(self, sequence: MDASequence) -> None:
        self.writer = None
        super()._on_mda_finished(sequence)

    def _create_writer(
        self, save_format: str, save_path: Path
    ) -> OMEZarrWriter | _OMETiffWriter | _TensorStoreHandler | None:
        """Create a writer for the MDAViewer based on the save format."""
        # use internal OME-TIFF writer if selected
        if OME_TIFF in save_format:
            # if OME-TIFF, save_path should be a directory without extension, so
            # we need to add the ".ome.tif" to correctly use the OMETiffWriter
            if not save_path.name.endswith(OME_TIFFS):
                save_path = save_path.with_suffix(OME_TIFF)
            return _OMETiffWriter(save_path)
        elif OME_ZARR in save_format:
            return OMEZarrWriter(save_path)
        elif ZARR_TESNSORSTORE in save_format:
            return self._create_zarr_tensorstore(save_path)
        # cannot use the ImageSequenceWriter here because the MDAViewer will not be
        # able to handle it.
        return None

    def _create_zarr_tensorstore(self, save_path: Path) -> _TensorStoreHandler:
        """Create a Zarr TensorStore writer."""
        return _TensorStoreHandler(
            driver="zarr",
            path=save_path,
            delete_existing=True,
            spec={"context": {"cache_pool": {"total_bytes_limit": GB_CACHE}}},
        )

    def _update_time_estimate(self) -> None:
        """Update the time estimate for the MDA experiment."""
        # this is a hack to avoid the error since MDASEquence does how to include
        # the stimulation events in the estimate_duration method. Need to fix this
        val = super().value()
        try:
            self._time_estimate = val.estimate_duration()
        except ValueError as e:  # pragma: no cover
            self._duration_label.setText(f"Error estimating time:\n{e}")
            return
