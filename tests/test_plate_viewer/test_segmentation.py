"""Tests for plate viewer segmentation functionality."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from qtpy.QtWidgets import QMessageBox

from micromanager_gui._plate_viewer._segmentation import (
    CUSTOM_MODEL_PATH,
    _CellposeSegmentation,
    _SelectModelPath,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_data():
    """Create a mock data reader."""
    mock_reader = Mock()
    mock_reader.sequence = Mock()
    mock_reader.sequence.stage_positions = [Mock() for _ in range(5)]  # 5 positions

    # Mock data for isel method
    mock_data = np.random.randint(0, 255, (10, 512, 512), dtype=np.uint8)
    mock_meta = [{"mda_event": {"pos_name": "A01"}}]
    mock_reader.isel.return_value = (mock_data, mock_meta)

    return mock_reader


@pytest.fixture
def mock_plate_viewer(qtbot):
    """Create a mock plate viewer that can serve as a Qt parent."""
    from qtpy.QtWidgets import QWidget

    # Create a real QWidget to serve as parent
    parent_widget = QWidget()
    qtbot.addWidget(parent_widget)

    # Add necessary mock attributes
    parent_widget._plate_map_group = Mock()
    parent_widget._analysis_wdg = Mock()
    parent_widget._tab = Mock()
    parent_widget.pv_labels_path = None

    return parent_widget


class TestSelectModelPath:
    """Test the _SelectModelPath widget."""

    def test_init(self, qtbot):
        """Test initialization of _SelectModelPath."""
        widget = _SelectModelPath()
        qtbot.addWidget(widget)

        assert widget._label_text == "Custom Model"
        assert (
            widget._label.toolTip() == "Choose the path to the custom Cellpose model."
        )
        assert not widget._is_dir  # Should be False for file selection

    def test_init_with_custom_params(self, qtbot):
        """Test initialization with custom parameters."""
        widget = _SelectModelPath(label="Test Model", tooltip="Test tooltip")
        qtbot.addWidget(widget)

        assert widget._label_text == "Test Model"
        assert widget._label.toolTip() == "Test tooltip"

    @patch("micromanager_gui._plate_viewer._segmentation.QFileDialog.getOpenFileName")
    def test_on_browse_with_selected_file(self, mock_dialog, qtbot):
        """Test file selection through browse dialog."""
        # Mock dialog to return a file path
        test_path = "/path/to/model.pt"
        mock_dialog.return_value = (test_path, "")

        widget = _SelectModelPath()
        qtbot.addWidget(widget)

        # Trigger browse
        widget._on_browse()

        # Check that path was set
        assert widget._path.text() == test_path
        mock_dialog.assert_called_once()

    @patch("micromanager_gui._plate_viewer._segmentation.QFileDialog.getOpenFileName")
    def test_on_browse_no_file_selected(self, mock_dialog, qtbot):
        """Test behavior when no file is selected."""
        # Mock dialog to return empty path
        mock_dialog.return_value = ("", "")

        widget = _SelectModelPath()
        qtbot.addWidget(widget)
        original_path = widget._path.text()

        # Trigger browse
        widget._on_browse()

        # Check that path wasn't changed
        assert widget._path.text() == original_path


class TestCellposeSegmentation:
    """Test the _CellposeSegmentation widget."""

    def test_init_without_parent(self, qtbot):
        """Test initialization without parent."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        assert widget._plate_viewer is None
        assert widget._data is None
        assert widget._labels == {}
        assert widget._worker is None
        assert widget.groupbox.title() == "Cellpose Segmentation"

    def test_init_with_parent_and_data(self, qtbot, mock_plate_viewer, mock_data):
        """Test initialization with parent and data."""
        widget = _CellposeSegmentation(parent=mock_plate_viewer, data=mock_data)
        qtbot.addWidget(widget)

        assert widget._plate_viewer is mock_plate_viewer
        assert widget._data is mock_data
        assert widget._labels == {}

    def test_data_property(self, qtbot, mock_data):
        """Test data property getter and setter."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Test getter
        assert widget.data is None

        # Test setter
        widget.data = mock_data
        assert widget.data is mock_data

    def test_labels_property(self, qtbot):
        """Test labels property."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        assert widget.labels == {}

        # Add some labels
        test_labels = {"pos_A01": np.array([[1, 2], [3, 4]])}
        widget._labels = test_labels
        assert widget.labels == test_labels

    def test_output_path_property(self, qtbot, temp_dir):
        """Test output_path property getter and setter."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Test getter when empty
        assert widget.labels_path is None

        # Test setter
        test_path = str(temp_dir)
        widget.labels_path = test_path
        assert widget.labels_path == test_path

        # Test setter with None
        widget.labels_path = None
        assert widget.labels_path is None

    def test_model_combo_changed(self, qtbot):
        """Test model combo box changes."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)
        widget.show()  # Make sure parent widget is visible

        # Initially custom model should be hidden
        assert widget._browse_custom_model.isHidden()

        # Change to custom model - directly call the method
        widget._on_model_combo_changed("custom")
        # Check if widget received show() call by checking if it's not hidden
        assert not widget._browse_custom_model.isHidden()

        # Change back to built-in model
        widget._on_model_combo_changed("cyto3")
        assert widget._browse_custom_model.isHidden()

    def test_validate_segmentation_setup_no_data(self, qtbot):
        """Test validation when no data is available."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        with patch(
            "micromanager_gui._plate_viewer._segmentation.show_error_dialog"
        ) as mock_error:
            assert not widget._validate_segmentation_setup()
            mock_error.assert_called_once()

    def test_validate_segmentation_setup_no_output_path(self, qtbot, mock_data):
        """Test validation when no output path is set."""
        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)

        with patch(
            "micromanager_gui._plate_viewer._segmentation.show_error_dialog"
        ) as mock_error:
            assert not widget._validate_segmentation_setup()
            mock_error.assert_called_once()

    def test_validate_segmentation_setup_invalid_path(self, qtbot, mock_data):
        """Test validation with invalid output path."""
        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)

        widget.labels_path = "/nonexistent/path"
        with patch(
            "micromanager_gui._plate_viewer._segmentation.show_error_dialog"
        ) as mock_error:
            assert not widget._validate_segmentation_setup()
            mock_error.assert_called_once()

    def test_validate_segmentation_setup_no_sequence(self, qtbot, temp_dir):
        """Test validation when data has no sequence."""
        mock_data = Mock()
        mock_data.sequence = None

        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)
        widget.labels_path = str(temp_dir)

        with patch(
            "micromanager_gui._plate_viewer._segmentation.show_error_dialog"
        ) as mock_error:
            assert not widget._validate_segmentation_setup()
            mock_error.assert_called_once()

    def test_validate_segmentation_setup_success(self, qtbot, mock_data, temp_dir):
        """Test successful validation."""
        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)
        widget.labels_path = str(temp_dir)

        assert widget._validate_segmentation_setup()

    def test_get_positions_no_data(self, qtbot):
        """Test getting positions when no data is available."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        assert widget._get_positions() is None

    def test_get_positions_empty_text(self, qtbot, mock_data):
        """Test getting all positions when text field is empty."""
        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)

        positions = widget._get_positions()
        assert positions == [0, 1, 2, 3, 4]  # All 5 positions

    def test_get_positions_valid_text(self, qtbot, mock_data):
        """Test parsing valid position text."""
        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)

        # Mock parse_lineedit_text to return specific positions
        with patch(
            "micromanager_gui._plate_viewer._segmentation.parse_lineedit_text"
        ) as mock_parse:
            mock_parse.return_value = [0, 2, 4]
            widget._pos_le.setText("0, 2, 4")

            positions = widget._get_positions()
            assert positions == [0, 2, 4]
            mock_parse.assert_called_once_with("0, 2, 4")

    def test_get_positions_invalid_text(self, qtbot, mock_data):
        """Test handling invalid position text."""
        widget = _CellposeSegmentation(data=mock_data)
        qtbot.addWidget(widget)

        # Mock parse_lineedit_text to return out-of-range positions
        with patch(
            "micromanager_gui._plate_viewer._segmentation.parse_lineedit_text"
        ) as mock_parse:
            with patch(
                "micromanager_gui._plate_viewer._segmentation.show_error_dialog"
            ) as mock_error:
                mock_parse.return_value = [
                    0,
                    10,
                ]  # Position 10 is out of range (only 0-4 available)
                widget._pos_le.setText("0, 10")

                positions = widget._get_positions()
                assert positions is None
                mock_error.assert_called_once()

    def test_handle_existing_labels_no_files(self, qtbot, temp_dir):
        """Test handling when no existing label files."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)
        widget.labels_path = str(temp_dir)

        assert widget._handle_existing_labels()

    def test_handle_existing_labels_with_files_overwrite_yes(self, qtbot, temp_dir):
        """Test handling existing files with user choosing to overwrite."""
        # Create a dummy tif file
        dummy_file = temp_dir / "test.tif"
        dummy_file.touch()

        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)
        widget.labels_path = str(temp_dir)

        with patch.object(widget, "_overwrite_msgbox") as mock_msgbox:
            mock_msgbox.return_value = QMessageBox.StandardButton.Yes
            assert widget._handle_existing_labels()
            mock_msgbox.assert_called_once()

    def test_handle_existing_labels_with_files_overwrite_no(self, qtbot, temp_dir):
        """Test handling existing files with user choosing not to overwrite."""
        # Create a dummy tif file
        dummy_file = temp_dir / "test.tif"
        dummy_file.touch()

        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)
        widget.labels_path = str(temp_dir)

        with patch.object(widget, "_overwrite_msgbox") as mock_msgbox:
            mock_msgbox.return_value = QMessageBox.StandardButton.No
            assert not widget._handle_existing_labels()
            mock_msgbox.assert_called_once()

    @patch("micromanager_gui._plate_viewer._segmentation.models.Cellpose")
    @patch("micromanager_gui._plate_viewer._segmentation.core.use_gpu")
    def test_initialize_model_builtin(self, mock_use_gpu, mock_cellpose, qtbot):
        """Test initializing built-in Cellpose model."""
        mock_use_gpu.return_value = True
        mock_model = Mock()
        mock_cellpose.return_value = mock_model

        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Set to built-in model
        widget._models_combo.setCurrentText("cyto3")

        model = widget._initialize_model()
        assert model is mock_model
        mock_cellpose.assert_called_once_with(gpu=True, model_type="cyto3")

    @patch("micromanager_gui._plate_viewer._segmentation.CellposeModel")
    @patch("micromanager_gui._plate_viewer._segmentation.core.use_gpu")
    def test_initialize_model_custom_with_path(
        self, mock_use_gpu, mock_cellpose_model, qtbot
    ):
        """Test initializing custom Cellpose model with valid path."""
        mock_use_gpu.return_value = False
        mock_model = Mock()
        mock_cellpose_model.return_value = mock_model

        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Set to custom model with path
        widget._models_combo.setCurrentText("custom")
        widget._browse_custom_model.setValue("/path/to/model.pt")

        model = widget._initialize_model()
        assert model is mock_model
        # Use normpath to handle cross-platform path separators
        expected_path = os.path.normpath("/path/to/model.pt")
        mock_cellpose_model.assert_called_once_with(
            pretrained_model=expected_path, gpu=False
        )

    def test_initialize_model_custom_no_path(self, qtbot):
        """Test initializing custom model without path."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Set to custom model without path
        widget._models_combo.setCurrentText("custom")
        widget._browse_custom_model.setValue("")

        with patch(
            "micromanager_gui._plate_viewer._segmentation.show_error_dialog"
        ) as mock_error:
            model = widget._initialize_model()
            assert model is None
            mock_error.assert_called_once()

    def test_enable_disable_widgets(self, qtbot, mock_plate_viewer):
        """Test enabling/disabling widgets."""
        widget = _CellposeSegmentation(parent=mock_plate_viewer)
        qtbot.addWidget(widget)

        # Test disabling
        widget._enable(False)
        assert not widget._model_wdg.isEnabled()
        assert not widget._browse_custom_model.isEnabled()
        assert not widget._diameter_wdg.isEnabled()
        assert not widget._pos_wdg.isEnabled()
        assert not widget._run_btn.isEnabled()

        # Verify plate viewer widgets are disabled
        mock_plate_viewer._analysis_wdg.setEnabled.assert_called_with(False)
        mock_plate_viewer._tab.setTabEnabled.assert_any_call(1, False)
        mock_plate_viewer._tab.setTabEnabled.assert_any_call(2, False)

        # Test enabling
        widget._enable(True)
        assert widget._model_wdg.isEnabled()
        assert widget._browse_custom_model.isEnabled()
        assert widget._diameter_wdg.isEnabled()
        assert widget._pos_wdg.isEnabled()
        assert widget._run_btn.isEnabled()

    def test_enable_disable_widgets_no_parent(self, qtbot):
        """Test enabling/disabling widgets without parent."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Should not raise an error
        widget._enable(False)
        widget._enable(True)

    def test_reset_progress_bar(self, qtbot):
        """Test resetting progress bar."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Set some values first
        widget._progress_bar.setValue(50)
        widget._progress_label.setText("[Test]")
        widget._elapsed_time_label.setText("01:23:45")

        # Reset
        widget._reset_progress_bar()

        assert widget._progress_bar.value() == 0
        assert widget._progress_label.text() == "[0/0]"
        assert widget._elapsed_time_label.text() == "00:00:00"

    def test_update_progress_bar_with_string(self, qtbot):
        """Test updating progress bar with string value."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        test_label = "[Well A01 p0 (tot 5)]"
        widget._update_progress_bar(test_label)

        assert widget._progress_label.text() == test_label

    def test_update_progress_bar_with_int(self, qtbot):
        """Test updating progress bar with integer value."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        widget._update_progress_bar(3)
        assert widget._progress_bar.value() == 3

    def test_update_progress_label(self, qtbot):
        """Test updating progress label with elapsed time."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        test_time = "00:05:30"
        widget._update_progress_label(test_time)

        assert widget._elapsed_time_label.text() == test_time

    def test_cancel_no_worker(self, qtbot):
        """Test canceling when no worker is running."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Should not raise an error
        widget.cancel()

    def test_cancel_with_worker(self, qtbot):
        """Test canceling with active worker."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Mock worker
        mock_worker = Mock()
        widget._worker = mock_worker

        with patch.object(widget._elapsed_timer, "stop") as mock_timer_stop:
            widget.cancel()

            mock_worker.quit.assert_called_once()
            mock_timer_stop.assert_called_once()

    def test_run_validation_fails(self, qtbot):
        """Test run method when validation fails."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # No data, so validation should fail
        with patch.object(widget, "_validate_segmentation_setup", return_value=False):
            widget.run()
            # Should exit early, no further processing

    def test_overwrite_msgbox(self, qtbot):
        """Test overwrite message box creation."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # We can't easily test the actual dialog execution in automated tests,
        # but we can verify the method doesn't crash
        with patch(
            "micromanager_gui._plate_viewer._segmentation.QMessageBox.exec"
        ) as mock_exec:
            mock_exec.return_value = QMessageBox.StandardButton.Yes
            result = widget._overwrite_msgbox()
            assert result == QMessageBox.StandardButton.Yes

    def test_close_event_with_worker(self, qtbot):
        """Test close event when worker is running."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Mock worker
        mock_worker = Mock()
        widget._worker = mock_worker

        # Create proper QCloseEvent
        from qtpy.QtGui import QCloseEvent

        close_event = QCloseEvent()

        widget.closeEvent(close_event)

        mock_worker.quit.assert_called_once()

    def test_close_event_no_worker(self, qtbot):
        """Test close event when no worker is running."""
        widget = _CellposeSegmentation()
        qtbot.addWidget(widget)

        # Create proper QCloseEvent
        from qtpy.QtGui import QCloseEvent

        close_event = QCloseEvent()

        # Should not raise an error
        widget.closeEvent(close_event)


class TestConstants:
    """Test module constants."""

    def test_custom_model_path_exists(self):
        """Test that CUSTOM_MODEL_PATH is properly defined."""
        assert isinstance(CUSTOM_MODEL_PATH, Path)
        assert "cp3_img8_epoch7000_py" in str(CUSTOM_MODEL_PATH)
