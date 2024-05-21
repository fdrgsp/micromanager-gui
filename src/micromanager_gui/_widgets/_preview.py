from __future__ import annotations

import numpy as np
import tifffile
from fonticon_mdi6 import MDI6
from pymmcore_plus import CMMCorePlus, Metadata
from pymmcore_widgets import ImagePreview, LiveButton, SnapButton
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledRangeSlider
from superqt.fonticon import icon
from superqt.utils import signals_blocked

BTN_SIZE = (40, 40)
SS = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}

QLabel { font-size: 12px; }

QRangeSlider { qproperty-barColor: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 80, 120, 0.2),
        stop:1 rgba(100, 80, 120, 0.4)
    )}

SliderLabel {
    font-size: 12px;
    color: white;
}
"""


class _ImagePreview(ImagePreview):
    """Subclass of ImagePreview.

    This subclass updates the LUT slider when the image is updated.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mmcore: CMMCorePlus | None = None,
        preview_widget: Preview,
        use_with_mda: bool = False,
    ):
        super().__init__(parent=parent, mmcore=mmcore, use_with_mda=use_with_mda)

        self._preview_wdg = preview_widget

        # the metadata associated with the image
        self._meta: Metadata | dict = {}

    def _on_image_snapped(self) -> None:
        if self._mmc.mda.is_running() and not self._use_with_mda:
            return
        self._update_image(self._mmc.getTaggedImage())

    def _on_streaming_stop(self) -> None:
        self.streaming_timer.stop()
        self._meta = self._mmc.getTags()

    def _update_image(self, data: tuple[np.ndarray, Metadata] | np.ndarray) -> None:
        """Update the image and the _clims slider."""
        if isinstance(data, np.ndarray):
            image = data
        else:
            image, self._meta = data

        super()._update_image(image)

        if self.image is None:
            return

        with signals_blocked(self._preview_wdg._clims):
            self._preview_wdg._clims.setValue(self.image.clim)


class Preview(QWidget):
    """A widget containing an ImagePreview and buttons for image preview."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        mmcore: CMMCorePlus | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")

        self._mmc = mmcore or CMMCorePlus.instance()

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # preview
        self._image_preview = _ImagePreview(self, mmcore=self._mmc, preview_widget=self)
        self._image_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self._image_preview)

        # buttons
        btn_wdg = QGroupBox()
        btn_wdg_layout = QHBoxLayout(btn_wdg)
        btn_wdg_layout.setContentsMargins(0, 0, 0, 0)

        # auto contrast checkbox
        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setMaximumWidth(42)
        self._auto_clim.setCheckable(True)
        self._auto_clim.setChecked(True)
        self._auto_clim.toggled.connect(self._clims_auto)
        # LUT slider
        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._clims.setStyleSheet(SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**8)
        self._clims.valueChanged.connect(self._on_clims_changed)
        # snap and live buttons
        self._snap = Snap(mmcore=self._mmc)
        self._snap.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._live = Live(mmcore=self._mmc)
        self._live.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # reset view button
        self._reset_view = QPushButton()
        self._reset_view.clicked.connect(self._reset)
        self._reset_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._reset_view.setToolTip("Reset View")
        self._reset_view.setIcon(icon(MDI6.home_outline))
        self._reset_view.setIconSize(QSize(25, 25))
        self._reset_view.setFixedSize(*BTN_SIZE)
        # save button
        self._save = QPushButton()
        self._save.clicked.connect(self._on_save)
        self._save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save.setToolTip("Save Image")
        self._save.setIcon(icon(MDI6.content_save_outline))
        self._save.setIconSize(QSize(25, 25))
        self._save.setFixedSize(*BTN_SIZE)

        btn_wdg_layout.addWidget(self._clims)
        btn_wdg_layout.addWidget(self._auto_clim)
        btn_wdg_layout.addWidget(self._snap)
        btn_wdg_layout.addWidget(self._live)
        btn_wdg_layout.addWidget(self._reset_view)
        btn_wdg_layout.addWidget(self._save)
        main_layout.addWidget(btn_wdg)

        # connections
        self._mmc.events.systemConfigurationLoaded.connect(self._on_sys_cfg_loaded)
        self._mmc.events.roiSet.connect(self._reset)

        self.destroyed.connect(self._disconnect)

        self._reset()
        self._on_sys_cfg_loaded()

    def _disconnect(self) -> None:
        self._mmc.events.systemConfigurationLoaded.disconnect(self._on_sys_cfg_loaded)

    def _on_sys_cfg_loaded(self) -> None:
        """Update the LUT slider range and the canvas size."""
        # update the LUT slider range
        if bit := self._mmc.getImageBitDepth():
            with signals_blocked(self._clims):
                self._clims.setRange(0, 2**bit - 1)
                self._clims.setValue((0, 2**bit - 1))

    def _reset(self) -> None:
        """Reset the preview."""
        x = (0, self._mmc.getImageWidth()) if self._mmc.getImageWidth() else None
        y = (0, self._mmc.getImageHeight()) if self._mmc.getImageHeight() else None
        self._image_preview.view.camera.set_range(x, y, margin=0)

    def _on_clims_changed(self, range: tuple[float, float]) -> None:
        """Update the LUT range."""
        self._image_preview.clims = range
        self._auto_clim.setChecked(False)

    def _clims_auto(self, state: bool) -> None:
        """Set the LUT range to auto."""
        self._image_preview.clims = "auto" if state else self._clims.value()
        if self._image_preview.image is not None:
            data = self._image_preview.image._data
            with signals_blocked(self._clims):
                self._clims.setValue((data.min(), data.max()))

    def _on_save(self) -> None:
        """Save the image as tif."""
        if self._image_preview.image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        tifffile.imwrite(
            path,
            self._image_preview.image._data,
            imagej=True,
            # description=self._image_preview._meta, # TODO: ome-tiff
        )


class Snap(SnapButton):
    """A SnapButton."""

    def __init__(
        self, parent: QWidget | None = None, *, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)
        self.setToolTip("Snap Image")
        self.setIcon(icon(MDI6.camera_outline))
        self.setText("")
        self.setFixedSize(*BTN_SIZE)


class Live(LiveButton):
    """A LiveButton."""

    def __init__(
        self, parent: QWidget | None = None, *, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent=parent, mmcore=mmcore)
        self.setToolTip("Live Mode")
        self.button_text_on = ""
        self.button_text_off = ""
        self.icon_color_on = ()
        self.icon_color_off = "magenta"
        self.setFixedSize(*BTN_SIZE)
