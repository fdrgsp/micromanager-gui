"""Tests for _InitDialog functionality."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from qtpy.QtWidgets import QApplication, QDialogButtonBox, QWidget

from micromanager_gui._plate_viewer._init_dialog import _InitDialog


class TestInitDialog:
    """Test cases for the _InitDialog class."""

    @pytest.fixture
    def app(self, qapp):
        """Get the QApplication instance."""
        return qapp

    @pytest.fixture
    def parent_widget(self, app):
        """Create a parent widget for the dialog."""
        return QWidget()

    @pytest.fixture
    def dialog(self, parent_widget):
        """Create a basic _InitDialog instance."""
        return _InitDialog(parent_widget)

    @pytest.fixture
    def dialog_with_paths(self, parent_widget):
        """Create an _InitDialog with predefined paths."""
        return _InitDialog(
            parent_widget,
            datastore_path="/path/to/datastore.zarr",
            labels_path="/path/to/labels",
            analysis_path="/path/to/analysis",
        )

    def test_init_dialog_creation_default(self, dialog):
        """Test dialog creation with default parameters."""
        assert dialog.windowTitle() == "Select Data Source"
        assert hasattr(dialog, "_browse_datastrore")
        assert hasattr(dialog, "_browse_labels")
        assert hasattr(dialog, "_browse_analysis")
        assert hasattr(dialog, "buttonBox")

    def test_init_dialog_creation_with_paths(self, dialog_with_paths):
        """Test dialog creation with predefined paths."""
        assert dialog_with_paths._browse_datastrore.value() == os.path.normpath(
            "/path/to/datastore.zarr"
        )
        assert dialog_with_paths._browse_labels.value() == os.path.normpath(
            "/path/to/labels"
        )
        assert dialog_with_paths._browse_analysis.value() == os.path.normpath(
            "/path/to/analysis"
        )

    def test_browse_widget_configuration(self, dialog):
        """Test that browse widgets are configured correctly."""
        # Test datastore browse widget
        datastore_widget = dialog._browse_datastrore
        assert "Datastore Path" in datastore_widget._label.text()
        assert "zarr datastore" in datastore_widget._label.toolTip()

        # Test labels browse widget
        labels_widget = dialog._browse_labels
        assert "Segmentation Path" in labels_widget._label.text()
        assert "labels images" in labels_widget._label.toolTip()
        assert "_on where n is the position number" in labels_widget._label.toolTip()

        # Test analysis browse widget
        analysis_widget = dialog._browse_analysis
        assert "Analysis Path" in analysis_widget._label.text()
        assert "json" in analysis_widget._label.toolTip()
        assert analysis_widget._is_dir is True

    def test_button_box_configuration(self, dialog):
        """Test that the button box is configured correctly."""
        button_box = dialog.buttonBox
        assert button_box is not None

        # Check that OK and Cancel buttons are present
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        assert ok_button is not None
        assert cancel_button is not None

    def test_label_width_styling(self, dialog):
        """Test that label widths are synchronized."""
        # Get the minimum width from the labels widget (which has the longest text)
        expected_width = dialog._browse_labels._label.minimumSizeHint().width()

        # Check that datastore and analysis labels have the same fixed width
        assert dialog._browse_datastrore._label.minimumWidth() == expected_width
        assert dialog._browse_analysis._label.minimumWidth() == expected_width

    def test_value_method_empty(self, dialog):
        """Test value method with empty fields."""
        result = dialog.value()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result == ("", "", "")

    def test_value_method_with_data(self, dialog):
        """Test value method with populated fields."""
        # Set values in the browse widgets
        dialog._browse_datastrore.setValue("/test/datastore.zarr")
        dialog._browse_labels.setValue("/test/labels")
        dialog._browse_analysis.setValue("/test/analysis")

        result = dialog.value()
        expected = (
            os.path.normpath("/test/datastore.zarr"),
            os.path.normpath("/test/labels"),
            os.path.normpath("/test/analysis"),
        )
        assert result == expected

    def test_value_method_with_pathlib_paths(self, dialog):
        """Test value method with pathlib.Path objects."""
        # Set values using Path objects
        dialog._browse_datastrore.setValue(Path("/test/datastore.zarr"))
        dialog._browse_labels.setValue(Path("/test/labels"))
        dialog._browse_analysis.setValue(Path("/test/analysis"))

        result = dialog.value()
        expected = (
            os.path.normpath("/test/datastore.zarr"),
            os.path.normpath("/test/labels"),
            os.path.normpath("/test/analysis"),
        )
        assert result == expected

    @patch("qtpy.QtWidgets.QFileDialog.getExistingDirectory")
    def test_browse_datastore_directory_selection(self, mock_file_dialog, dialog):
        """Test datastore browse functionality."""
        mock_file_dialog.return_value = "/selected/datastore/path"

        # Simulate clicking the browse button
        dialog._browse_datastrore._browse_btn.click()

        # Check that the path was set
        expected_path = os.path.normpath("/selected/datastore/path")
        assert dialog._browse_datastrore.value() == expected_path

    @patch("qtpy.QtWidgets.QFileDialog.getExistingDirectory")
    def test_browse_labels_directory_selection(self, mock_file_dialog, dialog):
        """Test labels browse functionality."""
        mock_file_dialog.return_value = "/selected/labels/path"

        # Simulate clicking the browse button
        dialog._browse_labels._browse_btn.click()

        # Check that the path was set
        expected_path = os.path.normpath("/selected/labels/path")
        assert dialog._browse_labels.value() == expected_path

    @patch("qtpy.QtWidgets.QFileDialog.getExistingDirectory")
    def test_browse_analysis_directory_selection(self, mock_file_dialog, dialog):
        """Test analysis browse functionality."""
        mock_file_dialog.return_value = "/selected/analysis/path"

        # Simulate clicking the browse button
        dialog._browse_analysis._browse_btn.click()

        # Check that the path was set
        expected_path = os.path.normpath("/selected/analysis/path")
        assert dialog._browse_analysis.value() == expected_path

    @patch("qtpy.QtWidgets.QFileDialog.getExistingDirectory")
    def test_browse_cancel_selection(self, mock_file_dialog, dialog):
        """Test behavior when user cancels file dialog."""
        mock_file_dialog.return_value = ""  # User cancelled

        original_value = dialog._browse_datastrore.value()
        dialog._browse_datastrore._browse_btn.click()

        # Value should remain unchanged
        assert dialog._browse_datastrore.value() == original_value

    def test_dialog_accept_behavior(self, dialog):
        """Test dialog accept behavior."""
        # Set some test values
        dialog._browse_datastrore.setValue("/test/datastore")
        dialog._browse_labels.setValue("/test/labels")
        dialog._browse_analysis.setValue("/test/analysis")

        # Track if the accepted signal was emitted
        accepted_signal_emitted = False

        def on_accepted():
            nonlocal accepted_signal_emitted
            accepted_signal_emitted = True

        # Connect to the accepted signal instead of mocking
        dialog.accepted.connect(on_accepted)

        # Simulate clicking OK button
        ok_button = dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.click()

        # Process Qt events to ensure signal is handled
        QApplication.processEvents()

        # Check that accepted signal was emitted
        assert accepted_signal_emitted

    def test_dialog_reject_behavior(self, dialog):
        """Test dialog reject behavior."""
        # Track if the rejected signal was emitted
        rejected_signal_emitted = False

        def on_rejected():
            nonlocal rejected_signal_emitted
            rejected_signal_emitted = True

        # Connect to the rejected signal instead of mocking
        dialog.rejected.connect(on_rejected)

        # Simulate clicking Cancel button
        cancel_button = dialog.buttonBox.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.click()

        # Process Qt events to ensure signal is handled
        QApplication.processEvents()

        # Check that rejected signal was emitted
        assert rejected_signal_emitted

    def test_dialog_layout_structure(self, dialog):
        """Test that the dialog layout is structured correctly."""
        layout = dialog.layout()
        assert layout is not None

        # Check that all widgets are in the layout
        widgets_in_layout = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget():
                widgets_in_layout.append(item.widget())

        # Should contain the three browse widgets and button box
        assert dialog._browse_datastrore in widgets_in_layout
        assert dialog._browse_labels in widgets_in_layout
        assert dialog._browse_analysis in widgets_in_layout
        assert dialog.buttonBox in widgets_in_layout

    def test_dialog_with_none_paths(self, parent_widget):
        """Test dialog creation with None paths."""
        dialog = _InitDialog(
            parent_widget,
            datastore_path=None,
            labels_path=None,
            analysis_path=None,
        )

        assert dialog._browse_datastrore.value() == ""
        assert dialog._browse_labels.value() == ""
        assert dialog._browse_analysis.value() == ""

    def test_dialog_resize_behavior(self, dialog):
        """Test that dialog can be resized."""
        original_size = dialog.size()

        # Set a specific size
        dialog.resize(800, 300)

        # Check that size changed
        new_size = dialog.size()
        assert new_size.width() == 800
        assert new_size.height() == 300
        assert new_size != original_size

    def test_signal_connections(self, dialog):
        """Test that signals are properly connected."""
        # Check that button box signals are connected
        button_box = dialog.buttonBox

        # These connections are tested indirectly through the behavior tests
        # but we can verify the signals exist
        assert hasattr(button_box, "accepted")
        assert hasattr(button_box, "rejected")

    def test_browse_widget_signals(self, dialog):
        """Test that browse widget signals work."""
        # Test pathSet signal for directory browse widgets
        datastore_signal_emitted = False
        labels_signal_emitted = False
        analysis_signal_emitted = False

        def on_datastore_path_set(path):
            nonlocal datastore_signal_emitted
            datastore_signal_emitted = True

        def on_labels_path_set(path):
            nonlocal labels_signal_emitted
            labels_signal_emitted = True

        def on_analysis_path_set(path):
            nonlocal analysis_signal_emitted
            analysis_signal_emitted = True

        dialog._browse_datastrore.pathSet.connect(on_datastore_path_set)
        dialog._browse_labels.pathSet.connect(on_labels_path_set)
        dialog._browse_analysis.pathSet.connect(on_analysis_path_set)

        # Test with mocked file dialogs
        with patch(
            "qtpy.QtWidgets.QFileDialog.getExistingDirectory", return_value="/test/path"
        ):
            dialog._browse_datastrore._browse_btn.click()
            dialog._browse_labels._browse_btn.click()
            dialog._browse_analysis._browse_btn.click()

        assert datastore_signal_emitted
        assert labels_signal_emitted
        assert analysis_signal_emitted

    def test_integration_workflow(self, parent_widget):
        """Test complete workflow from creation to value retrieval."""
        # Create dialog with initial paths
        initial_datastore = "/initial/datastore.zarr"
        initial_labels = "/initial/labels"
        initial_analysis = "/initial/analysis"

        dialog = _InitDialog(
            parent_widget,
            datastore_path=initial_datastore,
            labels_path=initial_labels,
            analysis_path=initial_analysis,
        )

        # Verify initial values
        expected_initial = (
            os.path.normpath(initial_datastore),
            os.path.normpath(initial_labels),
            os.path.normpath(initial_analysis),
        )
        assert dialog.value() == expected_initial

        # Modify values
        new_datastore = "/new/datastore.zarr"
        new_labels = "/new/labels"
        new_analysis = "/new/analysis"

        dialog._browse_datastrore.setValue(new_datastore)
        dialog._browse_labels.setValue(new_labels)
        dialog._browse_analysis.setValue(new_analysis)

        # Verify updated values
        expected_new = (
            os.path.normpath(new_datastore),
            os.path.normpath(new_labels),
            os.path.normpath(new_analysis),
        )
        assert dialog.value() == expected_new

        # Test that dialog structure is intact
        assert dialog.windowTitle() == "Select Data Source"
        assert dialog.buttonBox is not None

    def test_all_browse_widgets_are_directory_mode(self, dialog):
        """Test that all browse widgets are configured for directory selection."""
        assert dialog._browse_datastrore._is_dir is True
        assert dialog._browse_labels._is_dir is True
        assert dialog._browse_analysis._is_dir is True

    def test_empty_initialization(self, parent_widget):
        """Test initialization with no parameters."""
        dialog = _InitDialog(parent_widget)

        # All values should be empty strings
        assert dialog._browse_datastrore.value() == ""
        assert dialog._browse_labels.value() == ""
        assert dialog._browse_analysis.value() == ""

        # Dialog should still be functional
        assert dialog.windowTitle() == "Select Data Source"
        assert dialog.value() == ("", "", "")

    def test_widget_properties(self, dialog):
        """Test specific widget properties."""
        # Test that all browse widgets are properly configured
        widgets = [
            dialog._browse_datastrore,
            dialog._browse_labels,
            dialog._browse_analysis,
        ]

        for widget in widgets:
            assert hasattr(widget, "_label")
            assert hasattr(widget, "_path")
            assert hasattr(widget, "_browse_btn")
            assert widget._browse_btn.text() == "Browse"

    def test_browse_widget_tooltips(self, dialog):
        """Test that browse widgets have proper tooltips."""
        datastore_tooltip = dialog._browse_datastrore._label.toolTip()
        labels_tooltip = dialog._browse_labels._label.toolTip()
        analysis_tooltip = dialog._browse_analysis._label.toolTip()

        assert "zarr datastore" in datastore_tooltip
        assert "labels images" in labels_tooltip
        assert "tif files" in labels_tooltip
        assert "json" in analysis_tooltip

    def test_value_method_dynamic_changes(self, dialog):
        """Test value method after dynamic path changes."""
        # Set paths programmatically
        dialog._browse_datastrore.setValue("/new/datastore")
        dialog._browse_labels.setValue("/new/labels")
        dialog._browse_analysis.setValue("/new/analysis")

        result = dialog.value()
        expected = (
            os.path.normpath("/new/datastore"),
            os.path.normpath("/new/labels"),
            os.path.normpath("/new/analysis"),
        )
        assert result == expected

    def test_parent_assignment(self, parent_widget):
        """Test that parent is properly assigned."""
        dialog = _InitDialog(parent=parent_widget)
        assert dialog.parent() == parent_widget

    def test_browse_widget_signals_exist(self, dialog):
        """Test that browse widget signals are properly set up."""
        # Test that browse widgets have the correct signals
        assert hasattr(dialog._browse_datastrore, "pathSet")
        assert hasattr(dialog._browse_labels, "pathSet")
        assert hasattr(dialog._browse_analysis, "pathSet")

    def test_string_and_path_object_handling(self, parent_widget):
        """Test handling of both string and Path object inputs."""
        # Test with string paths
        dialog1 = _InitDialog(parent_widget, datastore_path="/string/path")
        expected_path = os.path.normpath("/string/path")
        assert dialog1._browse_datastrore.value() == expected_path

        # Test with Path objects converted to strings
        path_obj = Path("/path/object")
        dialog2 = _InitDialog(parent_widget, datastore_path=str(path_obj))
        assert dialog2._browse_datastrore.value() == str(path_obj)
