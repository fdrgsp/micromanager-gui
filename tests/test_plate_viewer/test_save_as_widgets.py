from unittest.mock import Mock, patch

import pytest
from qtpy.QtWidgets import QApplication

from micromanager_gui._plate_viewer._save_as_widgets import (
    _SaveAsTiff,
    _SaveAsCSV,
)


class TestSaveAsWidgets:
    @pytest.fixture(autouse=True)
    def setup_qt_app(self):
        """Ensure QApplication exists for Qt widgets."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_save_as_tiff_init(self):
        """Test _SaveAsTiff widget initialization."""
        widget = _SaveAsTiff()
        assert widget.windowTitle() == "Save As Tiff"
        assert hasattr(widget, '_pos_line_edit')
        assert hasattr(widget, '_browse_widget')

    def test_save_as_tiff_accept_valid_input(self):
        """Test _SaveAsTiff accept with valid input."""
        # Mock QFileDialog to prevent actual file dialogs from appearing
        with patch('qtpy.QtWidgets.QFileDialog.getExistingDirectory') as mock_file_dialog, \
             patch('qtpy.QtWidgets.QFileDialog.getOpenFileName') as mock_open_dialog, \
             patch('micromanager_gui._plate_viewer._save_as_widgets.parse_lineedit_text') as mock_parse, \
             patch('micromanager_gui._plate_viewer._save_as_widgets.Path') as mock_path, \
             patch('qtpy.QtWidgets.QDialog.accept') as mock_super_accept:

            # Prevent any file dialogs from showing
            mock_file_dialog.return_value = ""
            mock_open_dialog.return_value = ("", "")

            mock_parse.return_value = [0, 1, 2]

            # Mock Path operations to simulate valid directory
            mock_path_instance = Mock()
            mock_path_instance.is_dir.return_value = True
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            widget = _SaveAsTiff()
            widget._pos_line_edit.setText("0-2")
            widget._browse_widget.setValue("/fake/path")

            # Call accept which should proceed normally with valid path
            widget.accept()

            mock_parse.assert_called_once_with("0-2")
            # Path is called twice in accept(): once for is_dir() and once for exists()
            assert mock_path.call_count == 2
            mock_super_accept.assert_called_once()

    def test_save_as_tiff_accept_invalid_path(self):
        """Test _SaveAsTiff accept with invalid path (no directory exists)."""
        with patch('qtpy.QtWidgets.QFileDialog.getExistingDirectory') as mock_file_dialog, \
             patch('qtpy.QtWidgets.QFileDialog.getOpenFileName') as mock_open_dialog, \
             patch('micromanager_gui._plate_viewer._save_as_widgets.show_error_dialog') as mock_error, \
             patch('micromanager_gui._plate_viewer._save_as_widgets.parse_lineedit_text') as mock_parse:

            # Prevent any file dialogs from showing
            mock_file_dialog.return_value = ""
            mock_open_dialog.return_value = ("", "")

            mock_parse.return_value = [0, 1, 2]

            widget = _SaveAsTiff()
            widget._pos_line_edit.setText("0-2")
            widget._browse_widget.setValue("/nonexistent/path")

            widget.accept()

            mock_error.assert_called_once_with(
                widget, "Please select a path to save the .tiff files."
            )

    def test_save_as_tiff_value(self):
        """Test _SaveAsTiff value property."""
        widget = _SaveAsTiff()
        widget._browse_widget.setValue("/test/path")

        with patch(
            'micromanager_gui._plate_viewer._save_as_widgets.parse_lineedit_text'
        ) as mock_parse:
            mock_parse.return_value = [1, 2, 3]
            widget._pos_line_edit.setText("1-3")

            path, positions = widget.value()
            assert path == "/test/path"
            assert positions == [1, 2, 3]

    def test_save_as_csv_init(self):
        """Test _SaveAsCSV widget initialization."""
        widget = _SaveAsCSV()
        assert widget.windowTitle() == "Save Analysis As CSV"
        assert hasattr(widget, '_browse_widget')

    def test_save_as_csv_value(self):
        """Test _SaveAsCSV value property."""
        widget = _SaveAsCSV()
        widget._browse_widget.setValue("/test/csv/path")
        
        assert widget.value() == "/test/csv/path"
