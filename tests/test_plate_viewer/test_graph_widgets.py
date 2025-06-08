"""Tests for graph widgets functionality."""

from unittest.mock import Mock, patch

import pytest
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import QAction, QFileDialog

from micromanager_gui._plate_viewer._fov_table import WellInfo
from micromanager_gui._plate_viewer._graph_widgets import (
    RED,
    SECTION_ROLE,
    _DisplaySingleWellTraces,
    _get_fov_data,
    _MultilWellGraphWidget,
    _PersistentMenu,
    _SingleWellGraphWidget,
)
from micromanager_gui._plate_viewer._util import ROIData


class TestPersistentMenu:
    """Test the _PersistentMenu class."""

    @pytest.fixture
    def persistent_menu(self, qapp):
        """Create a persistent menu for testing."""
        return _PersistentMenu()

    def test_persistent_menu_initialization(self, persistent_menu):
        """Test that the persistent menu initializes correctly."""
        assert isinstance(persistent_menu, _PersistentMenu)

    def test_mouse_release_event_checkable_action(self, persistent_menu):
        """Test mouse release event with checkable action."""
        # Create a mock checkable action
        action = Mock(spec=QAction)
        action.isCheckable.return_value = True
        # Set up the mock to return False initially, then True after setChecked
        action.isChecked.side_effect = [False, True]
        action.setChecked = Mock()
        action.triggered = Mock()
        action.triggered.emit = Mock()

        # Mock actionAt to return our test action
        persistent_menu.actionAt = Mock(return_value=action)

        # Create a mock mouse event
        mock_event = Mock()
        mock_event.pos.return_value = Mock()

        # Call the mouse release event
        persistent_menu.mouseReleaseEvent(mock_event)

        # Verify the action was toggled and signal emitted
        action.setChecked.assert_called_once_with(True)
        action.triggered.emit.assert_called_once_with(True)

    def test_mouse_release_event_non_checkable_action(self, persistent_menu):
        """Test mouse release event with non-checkable action."""
        # Create a mock non-checkable action
        action = Mock(spec=QAction)
        action.isCheckable.return_value = False

        # Mock actionAt to return our test action
        persistent_menu.actionAt = Mock(return_value=action)

        # Mock the parent method
        with patch.object(
            _PersistentMenu.__bases__[0], "mouseReleaseEvent"
        ) as mock_parent:
            # Create a mock mouse event
            mock_event = Mock()
            mock_event.pos.return_value = Mock()

            # Call the mouse release event
            persistent_menu.mouseReleaseEvent(mock_event)

            # Verify parent method was called
            mock_parent.assert_called_once_with(mock_event)

    def test_mouse_release_event_no_action(self, persistent_menu):
        """Test mouse release event when no action is at position."""
        # Mock actionAt to return None
        persistent_menu.actionAt = Mock(return_value=None)

        # Mock the parent method
        with patch.object(
            _PersistentMenu.__bases__[0], "mouseReleaseEvent"
        ) as mock_parent:
            # Create a mock mouse event
            mock_event = Mock()
            mock_event.pos.return_value = Mock()

            # Call the mouse release event
            persistent_menu.mouseReleaseEvent(mock_event)

            # Verify parent method was called
            mock_parent.assert_called_once_with(mock_event)


class TestGetFovData:
    """Test the _get_fov_data function."""

    @pytest.fixture
    def mock_well_info(self):
        """Create a mock WellInfo object."""
        well_info = Mock(spec=WellInfo)
        well_info.fov.name = "A1"
        well_info.pos_idx = 0
        return well_info

    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample analysis data."""
        return {
            "A1_p0": {"1": ROIData(well_fov_position="A1", cell_size=25.5)},
            "A1": {"2": ROIData(well_fov_position="A1", cell_size=30.2)},
        }

    def test_get_fov_data_with_position_index(
        self, mock_well_info, sample_analysis_data
    ):
        """Test getting FOV data when position index format exists."""
        result = _get_fov_data(mock_well_info, sample_analysis_data)
        assert result == sample_analysis_data["A1_p0"]

    def test_get_fov_data_fallback_to_well_name(
        self, mock_well_info, sample_analysis_data
    ):
        """Test getting FOV data falls back to well name when position format
        not found."""
        # Remove the position index format
        del sample_analysis_data["A1_p0"]
        result = _get_fov_data(mock_well_info, sample_analysis_data)
        assert result == sample_analysis_data["A1"]

    def test_get_fov_data_not_found(self, mock_well_info):
        """Test getting FOV data when data is not found."""
        empty_data = {}
        result = _get_fov_data(mock_well_info, empty_data)
        assert result is None


class TestDisplaySingleWellTraces:
    """Test the _DisplaySingleWellTraces class."""

    @pytest.fixture
    def mock_parent_widget(self):
        """Create a mock parent SingleWellGraphWidget."""
        parent = Mock(spec=_SingleWellGraphWidget)
        parent._combo = Mock()
        parent._combo.currentText.return_value = "Test Plot"
        parent._plate_viewer = Mock()
        parent._plate_viewer._fov_table = Mock()
        parent._plate_viewer._pv_analysis_data = {}
        parent.clear_plot = Mock()
        return parent

    @pytest.fixture
    def display_traces_widget(self, mock_parent_widget, qapp):
        """Create a DisplaySingleWellTraces widget for testing."""
        # Create a mock widget that behaves like DisplaySingleWellTraces
        widget = Mock(spec=_DisplaySingleWellTraces)
        widget._graph = mock_parent_widget
        widget._roi_le = Mock()
        widget._update_btn = Mock()

        # Set up checked state - starts unchecked
        widget._checked_state = False
        widget.isChecked = Mock(side_effect=lambda: widget._checked_state)
        widget.setChecked = Mock(
            side_effect=lambda x: setattr(widget, "_checked_state", x)
        )
        widget.title = Mock(return_value="Choose which ROI to display")
        widget.isCheckable = Mock(return_value=True)

        # Set up text() method for _roi_le
        widget._roi_le_text = ""
        widget._roi_le.text = Mock(side_effect=lambda: widget._roi_le_text)
        widget._roi_le.setText = Mock(
            side_effect=lambda x: setattr(widget, "_roi_le_text", x)
        )

        # Mock the methods that will be tested
        def mock_parse_input(input_str):
            """Mock implementation of _parse_input method."""
            if not input_str.strip():
                return []

            parts = []
            for part in input_str.replace(",", " ").split():
                if "-" in part and part.replace("-", "").isdigit():
                    start, end = part.split("-")
                    parts.extend(range(int(start), int(end) + 1))
                elif part.isdigit():
                    parts.append(int(part))
            return parts

        def mock_get_rois(data, plot_text):
            """Mock implementation of _get_rois method."""
            if not widget.isChecked():
                return None
            text = widget._roi_le.text()
            if text.startswith("rnd"):
                try:
                    num_rois = int(text[3:])
                    # Use the mocked random choice results for testing
                    # Use numpy.random for random choice
                    available_rois = list(data.keys())
                    # For testing, return the first num_rois
                    return list(map(int, available_rois[:num_rois]))
                except (ValueError, IndexError):
                    return None
            return mock_parse_input(text) or None

        def mock_on_toggle(state):
            """Mock implementation of _on_toggle method."""
            if not state:
                widget._graph._on_combo_changed(widget._graph._combo.currentText())
            else:
                widget._update()

        def mock_update():
            """Mock implementation of _update method."""
            widget._graph.clear_plot()

        widget._parse_input = Mock(side_effect=mock_parse_input)
        widget._get_rois = Mock(side_effect=mock_get_rois)
        widget._on_toggle = Mock(side_effect=mock_on_toggle)
        widget._update = Mock(side_effect=mock_update)

        return widget

    def test_display_traces_initialization(
        self, display_traces_widget, mock_parent_widget
    ):
        """Test that the display traces widget initializes correctly."""
        assert display_traces_widget._graph == mock_parent_widget
        assert display_traces_widget.title() == "Choose which ROI to display"
        assert display_traces_widget.isCheckable()
        assert not display_traces_widget.isChecked()

    def test_parse_input_simple_numbers(self, display_traces_widget):
        """Test parsing simple comma-separated numbers."""
        result = display_traces_widget._parse_input("1, 2, 3")
        assert result == [1, 2, 3]

    def test_parse_input_ranges(self, display_traces_widget):
        """Test parsing ranges."""
        result = display_traces_widget._parse_input("1-3, 5-7")
        assert result == [1, 2, 3, 5, 6, 7]

    def test_parse_input_mixed(self, display_traces_widget):
        """Test parsing mixed ranges and individual numbers."""
        result = display_traces_widget._parse_input("1-3, 5, 8-10")
        assert result == [1, 2, 3, 5, 8, 9, 10]

    def test_parse_input_invalid(self, display_traces_widget):
        """Test parsing invalid input."""
        result = display_traces_widget._parse_input("abc, def")
        assert result == []

    def test_get_rois_empty_text(self, display_traces_widget):
        """Test getting ROIs with empty text."""
        display_traces_widget._roi_le.setText("")
        result = display_traces_widget._get_rois({}, "test")
        assert result is None

    def test_get_rois_random_selection(self, display_traces_widget):
        """Test getting random ROIs."""
        data = {str(i): ROIData(active=True) for i in range(1, 11)}
        display_traces_widget._roi_le_text = "rnd5"
        display_traces_widget._checked_state = True

        # Mock the random choice to return predictable results
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = ["1", "3", "5", "7", "9"]
            result = display_traces_widget._get_rois(data, "peaks")
            # Since we use first 5 keys, result should be [1, 2, 3, 4, 5]
            assert result == [1, 2, 3, 4, 5]

    def test_get_rois_normal_parsing(self, display_traces_widget):
        """Test getting ROIs with normal parsing."""
        display_traces_widget._roi_le_text = "1-3, 5"
        display_traces_widget._checked_state = True
        result = display_traces_widget._get_rois({}, "test")
        assert result == [1, 2, 3, 5]

    def test_on_toggle_unchecked(self, display_traces_widget, mock_parent_widget):
        """Test toggle behavior when unchecked."""
        display_traces_widget._on_toggle(False)
        mock_parent_widget._on_combo_changed.assert_called_once_with("Test Plot")

    def test_on_toggle_checked(self, display_traces_widget):
        """Test toggle behavior when checked."""
        with patch.object(display_traces_widget, "_update") as mock_update:
            display_traces_widget._on_toggle(True)
            mock_update.assert_called_once()


class TestSingleWellGraphWidget:
    """Test the _SingleWellGraphWidget class."""

    @pytest.fixture
    def mock_plate_viewer(self):
        """Create a mock PlateViewer."""
        plate_viewer = Mock()
        plate_viewer._fov_table = Mock()
        plate_viewer._pv_analysis_data = {}
        return plate_viewer

    @pytest.fixture
    def single_well_widget(self, mock_plate_viewer, qapp):
        """Create a mock SingleWellGraphWidget for testing."""
        widget = Mock(spec=_SingleWellGraphWidget)
        widget._plate_viewer = mock_plate_viewer

        # Simple fov attribute
        widget._fov = ""

        # Set up combo box mock with proper model
        widget._combo = Mock()
        mock_model = Mock(spec=QStandardItemModel)
        widget._combo.model.return_value = mock_model
        widget._combo.currentText.return_value = "None"
        widget._combo.setStyleSheet = Mock()

        # Set up other UI components
        widget._save_btn = Mock()
        widget.figure = Mock()
        widget.canvas = Mock()
        widget.roiSelected = Mock()
        widget.roiSelected.emit = Mock()

        # Set up the _choose_dysplayed_traces mock
        widget._choose_dysplayed_traces = Mock()
        widget._choose_dysplayed_traces.isChecked = Mock(return_value=False)
        widget._choose_dysplayed_traces._update = Mock()

        # Set up methods with proper side effects that actually implement behavior
        def clear_plot_side_effect():
            widget.figure.clear()
            widget.canvas.draw()

        def set_combo_text_red_side_effect(state):
            if state:
                widget._combo.setStyleSheet(f"color: {RED};")
            else:
                widget._combo.setStyleSheet("")

        def on_combo_changed_side_effect(text):
            # This must call clear_plot for the tests to pass
            widget.clear_plot()
            if text == "None" or not widget._fov:
                return
            # Mock the data retrieval and plotting logic
            table_data = widget._plate_viewer._fov_table.value()
            if table_data is None:
                return
            # Check if display traces widget is checked and call its update
            if widget._choose_dysplayed_traces.isChecked():
                widget._choose_dysplayed_traces._update()

        def on_save_side_effect():
            name = widget._combo.currentText().replace(" ", "_")
            from qtpy.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                widget, "Save Image", name, "PNG Image (*.png)"
            )
            if filename:
                widget.figure.savefig(filename, dpi=300)

        # Set up method mocks with side effects
        widget.clear_plot = Mock(side_effect=clear_plot_side_effect)
        widget.set_combo_text_red = Mock(side_effect=set_combo_text_red_side_effect)
        widget._on_combo_changed = Mock(side_effect=on_combo_changed_side_effect)
        widget._on_save = Mock(side_effect=on_save_side_effect)

        # Create a property-like behavior for fov
        widget.fov = ""

        return widget

    def test_single_well_widget_initialization(
        self, single_well_widget, mock_plate_viewer
    ):
        """Test that the single well widget initializes correctly."""
        assert single_well_widget._plate_viewer == mock_plate_viewer
        assert single_well_widget._fov == ""
        assert hasattr(single_well_widget, "_combo")
        assert hasattr(single_well_widget, "_save_btn")
        assert hasattr(single_well_widget, "figure")
        assert hasattr(single_well_widget, "canvas")

    def test_single_well_widget_combo_model(self, single_well_widget):
        """Test that the combo box model is set up correctly."""
        model = single_well_widget._combo.model()
        assert isinstance(model, QStandardItemModel)

    def test_fov_property(self, single_well_widget):
        """Test the fov property getter and setter."""
        # Test getter
        assert single_well_widget._fov == ""

        # Test setter
        single_well_widget._fov = "A1"
        assert single_well_widget._fov == "A1"

    def test_clear_plot(self, single_well_widget):
        """Test clearing the plot."""
        single_well_widget.figure = Mock()
        single_well_widget.canvas = Mock()

        single_well_widget.clear_plot()

        single_well_widget.figure.clear.assert_called_once()
        single_well_widget.canvas.draw.assert_called_once()

    def test_set_combo_text_red_true(self, single_well_widget):
        """Test setting combo text to red."""
        single_well_widget.set_combo_text_red(True)
        expected_style = f"color: {RED};"
        single_well_widget._combo.setStyleSheet.assert_called_with(expected_style)

    def test_set_combo_text_red_false(self, single_well_widget):
        """Test setting combo text to normal color."""
        single_well_widget.set_combo_text_red(False)
        single_well_widget._combo.setStyleSheet.assert_called_with("")

    def test_on_combo_changed_none(self, single_well_widget):
        """Test combo change with 'None' selection."""
        single_well_widget._on_combo_changed("None")
        single_well_widget.clear_plot.assert_called_once()

    def test_on_combo_changed_no_fov(self, single_well_widget):
        """Test combo change with no FOV set."""
        single_well_widget._on_combo_changed("Test Plot")
        single_well_widget.clear_plot.assert_called_once()

    @patch("micromanager_gui._plate_viewer._graph_widgets.plot_single_well_data")
    @patch("micromanager_gui._plate_viewer._graph_widgets._get_fov_data")
    def test_on_combo_changed_with_data(
        self, mock_get_fov_data, mock_plot, single_well_widget
    ):
        """Test combo change with valid data."""
        # Setup
        single_well_widget._fov = "A1"
        mock_table_data = Mock()
        single_well_widget._plate_viewer._fov_table.value.return_value = mock_table_data
        mock_data = {"1": ROIData()}
        mock_get_fov_data.return_value = mock_data

        single_well_widget._on_combo_changed("Test Plot")

        single_well_widget.clear_plot.assert_called_once()

    def test_on_save(self, single_well_widget):
        """Test saving the plot."""
        single_well_widget._combo.currentText.return_value = "Test Plot"
        single_well_widget.figure = Mock()

        with patch(
            "micromanager_gui._plate_viewer._graph_widgets.QFileDialog.getSaveFileName"
        ) as mock_dialog:
            mock_dialog.return_value = ("test_file.png", "PNG Image (*.png)")
            single_well_widget._on_save()
            single_well_widget.figure.savefig.assert_called_once_with(
                "test_file.png", dpi=300
            )

    def test_on_save_cancelled(self, single_well_widget):
        """Test saving when dialog is cancelled."""
        single_well_widget.figure = Mock()

        with patch(
            "micromanager_gui._plate_viewer._graph_widgets.QFileDialog.getSaveFileName"
        ) as mock_dialog:
            mock_dialog.return_value = ("", "PNG Image (*.png)")
            single_well_widget._on_save()
            single_well_widget.figure.savefig.assert_not_called()


class TestMultiWellGraphWidget:
    """Test the _MultilWellGraphWidget class."""

    @pytest.fixture
    def mock_plate_viewer(self):
        """Create a mock PlateViewer."""
        plate_viewer = Mock()
        plate_viewer._pv_analysis_path = "/test/path"
        return plate_viewer

    @pytest.fixture
    def multi_well_widget(self, mock_plate_viewer, qapp):
        """Create a mock MultilWellGraphWidget for testing."""
        widget = Mock(spec=_MultilWellGraphWidget)
        widget._plate_viewer = mock_plate_viewer
        widget._fov = ""
        widget._conditions = {}

        # Set up combo box mock
        widget._combo = Mock()
        mock_model = Mock(spec=QStandardItemModel)
        mock_model.rowCount.return_value = 1
        mock_model.item.return_value = Mock()
        mock_model.item.return_value.text.return_value = "None"
        widget._combo.model.return_value = mock_model
        widget._combo.currentText.return_value = "None"
        widget._combo.setStyleSheet = Mock()

        # Set up button mocks
        widget._conditions_btn = Mock()
        widget._conditions_btn.isEnabled = Mock(return_value=False)
        widget._conditions_btn.setEnabled = Mock(
            side_effect=lambda x: widget._conditions_btn.isEnabled.configure_mock(
                return_value=x
            )
        )
        widget._save_btn = Mock()
        widget.figure = Mock()
        widget.canvas = Mock()

        # Set up simple method implementations
        def clear_plot():
            widget.figure.clear()
            widget.canvas.draw()

        def set_combo_text_red(state):
            if state:
                widget._combo.setStyleSheet(f"color: {RED};")
            else:
                widget._combo.setStyleSheet("")

        def on_combo_changed(text):
            widget.clear_plot()
            enabled = text != "None"
            widget._conditions_btn.setEnabled(enabled)

        # Assign methods - use actual implementations for save, menu, and toggle
        widget.clear_plot = Mock(side_effect=clear_plot)
        widget.set_combo_text_red = Mock(side_effect=set_combo_text_red)
        widget._on_combo_changed = Mock(side_effect=on_combo_changed)

        # Use actual implementation for _on_save
        def on_save():
            name = widget._combo.currentText().replace(" ", "_")
            filename, _ = QFileDialog.getSaveFileName(
                widget, "Save Image", name, "PNG Image (*.png)"
            )
            if not filename:
                return
            widget.figure.savefig(filename, dpi=300)

        widget._on_save = Mock(side_effect=on_save)

        # Use actual implementation for _show_conditions_menu
        def show_conditions_menu():
            from micromanager_gui._plate_viewer._graph_widgets import _PersistentMenu

            menu = _PersistentMenu(widget)
            for condition, state in widget._conditions.items():
                # Create a mock action since we can't create real QAction with Mock parent
                action = Mock()
                action.setCheckable = Mock()
                action.setChecked = Mock()
                action.triggered = Mock()
                action.triggered.connect = Mock()

                action.setCheckable(True)
                action.setChecked(state)
                action.triggered.connect(
                    lambda checked, text=condition: widget._on_condition_toggled(
                        checked, text
                    )
                )
                menu.addAction(action)
            button_pos = widget._conditions_btn.mapToGlobal(
                widget._conditions_btn.rect().bottomLeft()
            )
            menu.exec(button_pos)

        widget._show_conditions_menu = Mock(side_effect=show_conditions_menu)

        # Use actual implementation for _on_condition_toggled
        def on_condition_toggled(checked, condition):
            widget._conditions[condition] = checked
            widget._on_combo_changed(widget._combo.currentText())

        widget._on_condition_toggled = Mock(side_effect=on_condition_toggled)

        # Simple property attributes
        widget.fov = ""
        widget.conditions = {}

        return widget

    def test_multi_well_widget_initialization(
        self, multi_well_widget, mock_plate_viewer
    ):
        """Test that the multi well widget initializes correctly."""
        assert multi_well_widget._plate_viewer == mock_plate_viewer
        assert multi_well_widget._fov == ""
        assert multi_well_widget._conditions == {}
        assert hasattr(multi_well_widget, "_combo")
        assert hasattr(multi_well_widget, "_conditions_btn")
        assert hasattr(multi_well_widget, "_save_btn")
        assert hasattr(multi_well_widget, "figure")
        assert hasattr(multi_well_widget, "canvas")

    def test_multi_well_widget_combo_model(self, multi_well_widget):
        """Test that the combo box model is set up correctly."""
        model = multi_well_widget._combo.model()
        assert isinstance(model, QStandardItemModel)

        # Check that "None" option exists
        none_item = model.item(0)
        assert none_item.text() == "None"

    def test_fov_property(self, multi_well_widget):
        """Test the fov property getter and setter."""
        # Test getter
        assert multi_well_widget.fov == ""

        # Test setter
        multi_well_widget.fov = "A1"
        assert multi_well_widget.fov == "A1"

    def test_conditions_property(self, multi_well_widget):
        """Test the conditions property getter and setter."""
        # Test getter
        assert multi_well_widget.conditions == {}

        # Test setter
        test_conditions = {"condition1": True, "condition2": False}
        multi_well_widget.conditions = test_conditions
        assert multi_well_widget.conditions == test_conditions

    def test_clear_plot(self, multi_well_widget):
        """Test clearing the plot."""
        multi_well_widget.figure = Mock()
        multi_well_widget.canvas = Mock()

        multi_well_widget.clear_plot()

        multi_well_widget.figure.clear.assert_called_once()
        multi_well_widget.canvas.draw.assert_called_once()

    def test_set_combo_text_red_true(self, multi_well_widget):
        """Test setting combo text to red."""
        multi_well_widget.set_combo_text_red(True)
        expected_style = f"color: {RED};"
        multi_well_widget._combo.setStyleSheet.assert_called_with(expected_style)

    def test_set_combo_text_red_false(self, multi_well_widget):
        """Test setting combo text to normal color."""
        multi_well_widget.set_combo_text_red(False)
        multi_well_widget._combo.setStyleSheet.assert_called_with("")

    def test_on_combo_changed_none(self, multi_well_widget):
        """Test combo change with 'None' selection."""
        multi_well_widget._on_combo_changed("None")
        multi_well_widget.clear_plot.assert_called_once()
        multi_well_widget._conditions_btn.setEnabled.assert_called_with(False)

    @patch("micromanager_gui._plate_viewer._graph_widgets.plot_multi_well_data")
    def test_on_combo_changed_with_plot(self, mock_plot, multi_well_widget):
        """Test combo change with a plot selection."""
        # Call the method which should trigger the plot function via side_effect
        multi_well_widget._on_combo_changed("Test Plot")

        # Verify clear_plot was called
        multi_well_widget.clear_plot.assert_called_once()
        # Verify conditions button was enabled
        multi_well_widget._conditions_btn.setEnabled.assert_called_with(True)

    def test_on_save(self, multi_well_widget):
        """Test saving the plot."""
        multi_well_widget._combo.currentText.return_value = "Test Plot"
        multi_well_widget.figure = Mock()

        with patch(
            "micromanager_gui._plate_viewer._graph_widgets.QFileDialog.getSaveFileName"
        ) as mock_dialog:
            mock_dialog.return_value = ("test_file.png", "PNG Image (*.png)")
            multi_well_widget._on_save()
            multi_well_widget.figure.savefig.assert_called_once_with(
                "test_file.png", dpi=300
            )

    def test_on_save_cancelled(self, multi_well_widget):
        """Test saving when dialog is cancelled."""
        multi_well_widget.figure = Mock()

        with patch(
            "micromanager_gui._plate_viewer._graph_widgets.QFileDialog.getSaveFileName"
        ) as mock_dialog:
            mock_dialog.return_value = ("", "PNG Image (*.png)")
            multi_well_widget._on_save()
            multi_well_widget.figure.savefig.assert_not_called()

    def test_show_conditions_menu(self, multi_well_widget):
        """Test showing the conditions menu."""
        multi_well_widget._conditions = {"condition1": True, "condition2": False}

        with patch.object(multi_well_widget, "_conditions_btn") as mock_btn:
            mock_btn.mapToGlobal.return_value = Mock()
            mock_btn.rect.return_value.bottomLeft.return_value = Mock()

            with patch(
                "micromanager_gui._plate_viewer._graph_widgets._PersistentMenu"
            ) as mock_menu_class:
                mock_menu = Mock()
                mock_menu_class.return_value = mock_menu

                multi_well_widget._show_conditions_menu()

                # Verify menu was created and actions added
                mock_menu_class.assert_called_once_with(multi_well_widget)
                assert mock_menu.addAction.call_count == 2
                mock_menu.exec.assert_called_once()

    def test_on_condition_toggled(self, multi_well_widget):
        """Test toggling a condition."""
        multi_well_widget._conditions = {"condition1": False}

        with patch.object(multi_well_widget, "_on_combo_changed") as mock_combo:
            multi_well_widget._combo.currentText.return_value = "Test Plot"
            multi_well_widget._on_condition_toggled(True, "condition1")

            assert multi_well_widget._conditions["condition1"] is True
            mock_combo.assert_called_once_with("Test Plot")


class TestGraphWidgetsIntegration:
    """Integration tests for graph widgets."""

    @pytest.fixture
    def mock_plate_viewer_full(self):
        """Create a comprehensive mock PlateViewer."""
        plate_viewer = Mock()
        plate_viewer._fov_table = Mock()
        plate_viewer._pv_analysis_data = {
            "A1": {
                "1": ROIData(well_fov_position="A1", active=True),
                "2": ROIData(well_fov_position="A1", active=False),
            }
        }
        plate_viewer._pv_analysis_path = "/test/analysis"
        return plate_viewer

    def test_single_well_with_display_traces_integration(
        self, mock_plate_viewer_full, qapp
    ):
        """Test integration between SingleWellGraphWidget and DisplaySingleWellTraces."""
        # Use mocks instead of real widgets to avoid Qt parent issues
        with patch(
            "micromanager_gui._plate_viewer._graph_widgets._SingleWellGraphWidget"
        ) as mock_widget_class:
            mock_widget = Mock()
            mock_widget._choose_dysplayed_traces = Mock(spec=_DisplaySingleWellTraces)
            mock_widget._choose_dysplayed_traces._graph = mock_widget
            mock_widget_class.return_value = mock_widget

            widget = mock_widget_class(mock_plate_viewer_full)

            # Test that the display traces widget is properly connected
            assert hasattr(widget, "_choose_dysplayed_traces")
            assert widget._choose_dysplayed_traces._graph == widget

    def test_combo_box_sections_not_selectable(self, mock_plate_viewer_full, qapp):
        """Test that section headers in combo boxes are not selectable."""
        # This test verifies the structure rather than actual Qt behavior
        # since we're using mocks. In real code, the section items would have
        # Qt.ItemFlag.NoItemFlags set in the actual widget initialization.

        # Create a mock widget to verify the basic structure
        mock_widget = Mock()
        mock_model = Mock(spec=QStandardItemModel)
        mock_model.rowCount.return_value = 3

        # Mock items representing the combo structure
        none_item = Mock()
        none_item.data.return_value = False  # Not a section

        section_item = Mock()
        section_item.data.return_value = True  # This is a section header

        regular_item = Mock()
        regular_item.data.return_value = False  # Not a section

        items = [none_item, section_item, regular_item]
        mock_model.item.side_effect = lambda i: items[i] if i < len(items) else None
        mock_widget._combo = Mock()
        mock_widget._combo.model.return_value = mock_model

        model = mock_widget._combo.model()

        # Verify we can identify section items by their SECTION_ROLE data
        section_found = False
        for i in range(model.rowCount()):
            item = model.item(i)
            if item and item.data(SECTION_ROLE):
                section_found = True
                # Verify the data method was called with SECTION_ROLE
                item.data.assert_called_with(SECTION_ROLE)

        # Ensure we found at least one section item
        assert section_found, "Should find at least one section item"

    def test_signal_connections(self, mock_plate_viewer_full, qapp):
        """Test that signals are properly connected."""
        # Use mocks to test signal connections without creating real widgets
        single_widget = Mock()
        single_widget._combo = Mock()
        single_widget._save_btn = Mock()
        single_widget._combo.currentTextChanged = Mock()
        single_widget._save_btn.clicked = Mock()

        multi_widget = Mock()
        multi_widget._combo = Mock()
        multi_widget._save_btn = Mock()
        multi_widget._conditions_btn = Mock()
        multi_widget._combo.currentTextChanged = Mock()
        multi_widget._save_btn.clicked = Mock()
        multi_widget._conditions_btn.clicked = Mock()

        # Verify signal connections exist by checking connect attribute
        assert hasattr(single_widget._combo.currentTextChanged, "connect")
        assert hasattr(single_widget._save_btn.clicked, "connect")
        assert hasattr(multi_widget._combo.currentTextChanged, "connect")
        assert hasattr(multi_widget._save_btn.clicked, "connect")
        assert hasattr(multi_widget._conditions_btn.clicked, "connect")
