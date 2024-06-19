from __future__ import annotations

from typing import TYPE_CHECKING

import cmap
from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledRangeSlider
from superqt.fonticon import icon
from superqt.utils import qthrottled, signals_blocked
from vispy import scene

from ._util import show_error_dialog

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from vispy.scene.events import SceneMouseEvent


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


class _ImageViewer(QGroupBox):
    """A widget for displaying an image."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._viewer = _ImageCanvas(parent=self)

        # roi number indicator
        self._roi_number = QLabel()
        self._roi_number.setText("ROI:")

        # LUT slider
        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._clims.setStyleSheet(SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**8)
        self._clims.valueChanged.connect(self._on_clims_changed)
        # auto contrast checkbox
        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setCheckable(True)
        self._auto_clim.setChecked(True)
        self._auto_clim.toggled.connect(self._clims_auto)
        # labels
        self._labels = QPushButton("Labels")
        self._labels.setCheckable(True)
        self._labels.setChecked(False)
        self._labels.toggled.connect(self._show_labels)
        # reset view button
        self._reset_view = QPushButton()
        self._reset_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._reset_view.setToolTip("Reset View")
        self._reset_view.setIcon(icon(MDI6.fullscreen))
        self._reset_view.clicked.connect(self._reset)
        # bottom widget
        bottom_wdg = QWidget()
        bottom_wdg_layout = QHBoxLayout(bottom_wdg)
        bottom_wdg_layout.setContentsMargins(0, 0, 0, 0)
        bottom_wdg_layout.addWidget(self._clims)
        bottom_wdg_layout.addWidget(self._auto_clim)
        bottom_wdg_layout.addWidget(self._labels)
        bottom_wdg_layout.addWidget(self._reset_view)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self._roi_number)
        main_layout.addWidget(self._viewer)
        main_layout.addWidget(bottom_wdg)

    def setData(self, data: np.ndarray | None, labels: np.ndarray | None) -> None:
        """Set the image data."""
        self._clear()
        if data is None:
            return

        if len(data.shape) > 2:
            show_error_dialog(self, "Only 2D images are supported!")
            return

        self._clims.setRange(data.min(), data.max())
        self._viewer.update_image(data, labels)
        self._auto_clim.setChecked(True)

        if labels is None:
            self._labels.setChecked(False)
        elif self._viewer.labels_image is not None:
            self._viewer.labels_image.visible = self._labels.isChecked()

    def data(self) -> np.ndarray | None:
        """Return the image data."""
        return self._viewer.image._data if self._viewer.image is not None else None

    def _on_clims_changed(self, range: tuple[float, float]) -> None:
        """Update the LUT range."""
        self._viewer.clims = range
        self._auto_clim.setChecked(False)

    def _clims_auto(self, state: bool) -> None:
        """Set the LUT range to auto."""
        self._viewer.clims = "auto" if state else self._clims.value()
        if self._viewer.image is not None:
            data = self._viewer.image._data
            with signals_blocked(self._clims):
                self._clims.setValue((data.min(), data.max()))

    def _reset(self) -> None:
        """Reset the view."""
        self._viewer.view.camera.set_range(margin=0)

    def _clear(self) -> None:
        """Clear the image."""
        if self._viewer.image is not None:
            self._viewer.image.parent = None
            self._viewer.image = None
        if self._viewer.labels_image is not None:
            self._viewer.labels_image.parent = None
            self._viewer.labels_image = None
        self._viewer.view.camera.set_range(margin=0)

    def _show_labels(self, state: bool) -> None:
        """Show the labels."""
        if self._viewer.labels_image is not None:
            self._viewer.labels_image.visible = state


class _ImageCanvas(QWidget):
    """A Widget that displays an image."""

    def __init__(self, parent: _ImageViewer):
        super().__init__(parent=parent)
        self._viewer = parent
        self._imcls = scene.visuals.Image
        self._clims: tuple[float, float] | Literal["auto"] = "auto"
        self._cmap: str = "grays"

        self._canvas = scene.SceneCanvas(keys="interactive", parent=self)
        self._canvas.events.mouse_move.connect(qthrottled(self._on_mouse_move, 60))
        self.view = self._canvas.central_widget.add_view(camera="panzoom")
        self.view.camera.aspect = 1

        self._lbl = None

        self.image: scene.visuals.Image | None = None
        self.labels_image: scene.visuals.Image | None = None

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self._canvas.native)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def clims(self) -> tuple[float, float] | Literal["auto"]:
        """Get the contrast limits of the image."""
        return self._clims

    @clims.setter
    def clims(self, clims: tuple[float, float] | Literal["auto"] = "auto") -> None:
        """Set the contrast limits of the image.

        Parameters
        ----------
        clims : tuple[float, float], or "auto"
            The contrast limits to set.
        """
        if self.image is not None:
            self.image.clim = clims
        self._clims = clims

    @property
    def cmap(self) -> str:
        """Get the colormap (lookup table) of the image."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: str = "grays") -> None:
        """Set the colormap (lookup table) of the image.

        Parameters
        ----------
        cmap : str
            The colormap to use.
        """
        if self.image is not None:
            self.image.cmap = cmap
        self._cmap = cmap

    def update_image(self, img: np.ndarray, labels: np.ndarray | None = None) -> None:
        clim = (img.min(), img.max())
        self.image = self._imcls(
            img, cmap=self._cmap, clim=clim, parent=self.view.scene
        )
        self.image.set_gl_state("additive", depth_test=False)
        self.image.interactive = True
        self.view.camera.set_range(margin=0)

        if labels is None:
            return

        self.labels_image = self._imcls(
            labels,
            cmap=cmap.Colormap("tab20").to_vispy(),
            clim=(labels.min(), labels.max()),
            parent=self.view.scene,
        )
        self.labels_image.set_gl_state("additive", depth_test=False)
        self.labels_image.interactive = True
        self.labels_image.visible = False

    def _on_mouse_move(self, event: SceneMouseEvent) -> None:
        """Update the pixel value when the mouse moves."""
        visual = self._canvas.visual_at(event.pos)
        image = self._find_image(visual)
        if image != self.labels_image or image is None:
            self._viewer._roi_number.setText("ROI:")
            return
        tform = image.get_transform("canvas", "visual")
        px, py, *_ = (int(x) for x in tform.map(event.pos))
        pixel_value = image._data[py, px]
        pixel_value = "" if pixel_value == 0 else pixel_value
        self._viewer._roi_number.setText(f"ROI: {pixel_value}")

    def _find_image(self, visual: scene.visuals.Visual) -> scene.visuals.Image | None:
        """Find the image visual in the visual tree."""
        if visual is None:
            return None
        if isinstance(visual, scene.visuals.Image):
            return visual
        for child in visual.children:
            image = self._find_image(child)
            if image is not None:
                return image
        return None