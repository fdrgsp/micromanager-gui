from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import cmap
import numpy as np
from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import find_contours
from superqt import QLabeledRangeSlider
from superqt.fonticon import icon
from superqt.utils import qthrottled, signals_blocked
from vispy import scene
from vispy.color import Colormap

from ._util import parse_lineedit_text, show_error_dialog

if TYPE_CHECKING:
    from typing import Literal

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

    valueChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._viewer = _ImageCanvas(parent=self)
        self._viewer._canvas.events.mouse_press.connect(self._on_mouse_press)

        # roi number indicator
        find_roi_lbl = QLabel("ROI:")
        find_roi_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._roi_number_le = QLineEdit()
        self._find_btn = QPushButton("Find")
        self._find_btn.clicked.connect(self._highlight_rois)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_highlight)
        roi_wdg = QWidget()
        roi_wdg.setToolTip(
            "Select the ROIs to highlight. You can input single ROIs (e.g. 30, 33) a "
            "range (e.g. 1-10), or a mix of single ROIs and ranges "
            "(e.g. 1-10, 30, 50-65). NOTE: The ROIs are 1-indexed."
        )
        roi_layout = QHBoxLayout(roi_wdg)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(find_roi_lbl)
        roi_layout.addWidget(self._roi_number_le)
        roi_layout.addWidget(self._find_btn)
        roi_layout.addWidget(self._clear_btn)

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
        main_layout.addWidget(roi_wdg)
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
        elif (
            self._viewer.labels_image is not None
            and self._viewer.contours_image is not None
        ):
            self._viewer.contours_image.visible = self._labels.isChecked()

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
        if self._viewer.contours_image is not None:
            self._viewer.contours_image.parent = None
            self._viewer.contours_image = None
        self._viewer.view.camera.set_range(margin=0)

    def _clear_highlight(self) -> None:
        """Clear the highlighted ROI."""
        if self._viewer.highlight_roi is not None:
            self._viewer.highlight_roi.parent = None
            self._viewer.highlight_roi = None
        self._roi_number_le.setText("")

    def _show_labels(self, state: bool) -> None:
        """Show the labels."""
        self._clear_highlight()

        if (
            self._viewer.labels_image is not None
            and self._viewer.contours_image is not None
        ):
            self._viewer.contours_image.visible = state

    def _highlight_rois(self, roi: int | bool | None = None) -> None:
        """Highlight the label set in the spinbox."""
        if self._viewer.labels_image is None:
            show_error_dialog(self, "No labels image to highlight.")
            return

        labels_data = self._viewer.labels_image._data

        # roi could be a bool when called from the find button. In that case, we
        # need to parse the text from the line edit
        if isinstance(roi, bool):
            roi = None
        rois = parse_lineedit_text(self._roi_number_le.text()) if roi is None else [roi]

        if not rois:
            show_error_dialog(self, "Invalid ROIs provided!")
            return None
        if max(rois) >= labels_data.max() + 1:
            show_error_dialog(self, "Input ROIs out of range!")
            return None

        # Clear the previous highlight image if it exists
        if self._viewer.highlight_roi is not None:
            self._viewer.highlight_roi.parent = None
            self._viewer.highlight_roi = None

        # Create a mask for the label to highlight it
        highlight = np.zeros_like(labels_data, dtype=np.uint8)
        for roi in rois:
            mask = labels_data == roi
            highlight[mask] = 255

        # Add the highlight image to the viewer
        self._viewer.highlight_roi = scene.visuals.Image(
            highlight,
            cmap=cmap.Colormap("green").to_vispy(),
            clim=(0, 255),
            parent=self._viewer.view.scene,
        )
        self._viewer.highlight_roi.set_gl_state("additive", depth_test=False)
        self._viewer.highlight_roi.interactive = True
        # self._viewer.view.camera.set_range(margin=0)

        self._viewer.labels_image.visible = False
        with signals_blocked(self._labels):
            self._labels.setChecked(False)

    def _on_mouse_press(self, event: SceneMouseEvent) -> None:
        """Emit the value of the clicked ROI.

        This is used to update the graphs when a ROI is clicked.
        """
        visual = self._viewer._canvas.visual_at(event.pos)
        image = self._viewer._find_image(visual)
        if image is None or self._viewer.labels_image is None:
            return
        if image != self._viewer.labels_image:
            image = self._viewer.labels_image
        tform = image.get_transform("canvas", "visual")
        px, py, *_ = (int(x) for x in tform.map(event.pos))
        try:
            roi = image._data[py, px]
        except IndexError:
            return
        # exclude background
        if roi == 0:
            return
        self._highlight_rois(roi)
        self.valueChanged.emit(roi)


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
        self.contours_image: scene.visuals.Image | None = None
        self.highlight_roi: scene.visuals.Image | None = None

        self._contour_cache: dict[str, np.ndarray] = {}

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
            cmap=self._labels_custom_cmap(labels.max()),
            clim=(labels.min(), labels.max()),
            parent=self.view.scene,
        )
        self.labels_image.set_gl_state("additive", depth_test=False)
        self.labels_image.interactive = True
        self.labels_image.visible = False

        contour_key = self._hash_labels(labels)
        if contour_key not in self._contour_cache:
            self._contour_cache[contour_key] = self._extract_label_contours(labels)

        self.contours_image = self._imcls(
            self._contour_cache[contour_key],
            cmap=self._labels_custom_cmap(labels.max()),
            clim=(labels.min(), labels.max()),
            parent=self.view.scene,
        )
        self.contours_image.set_gl_state("additive", depth_test=False)
        self.contours_image.interactive = True
        self.contours_image.visible = False

    def _hash_labels(self, labels: np.ndarray) -> str:
        """Generate a unique hash for a given labels array."""
        return hashlib.sha256(labels.tobytes()).hexdigest()

    def _labels_custom_cmap(self, n_labels: int) -> Colormap:
        """Create a custom colormap for the labels."""
        colors = np.zeros((n_labels + 1, 4))
        colors[0] = [0, 0, 0, 1]  # Black for background (0)
        for i in range(1, n_labels + 1):
            colors[i] = [1, 1, 0, 1]  # yellow for all other labels
        return Colormap(colors)

    def _extract_label_contours(
        self, labels: np.ndarray, thickness: int = 1
    ) -> np.ndarray:
        """Extracts contours of the labels and increases thickness using NumPy."""
        contours = np.zeros_like(labels, dtype=np.uint8)

        for label in np.unique(labels):
            if label == 0:
                continue  # Skip background

            mask = labels == label  # Binary mask for the current label
            contour_list = find_contours(mask.astype(float), level=0.5)  # Find contours

            for contour in contour_list:
                contour = np.round(contour).astype(
                    int
                )  # Convert to integer coordinates

                # Expand the contour by setting a small square around each point
                for x, y in contour:
                    x_min, x_max = (
                        max(0, x - thickness),
                        min(labels.shape[0], x + thickness + 1),
                    )
                    y_min, y_max = (
                        max(0, y - thickness),
                        min(labels.shape[1], y + thickness + 1),
                    )
                    contours[x_min:x_max, y_min:y_max] = label  # Assign label value

        return contours

    def _on_mouse_move(self, event: SceneMouseEvent) -> None:
        """Update the pixel value when the mouse moves."""
        visual = self._canvas.visual_at(event.pos)
        image = self._find_image(visual)
        if image is None or self.labels_image is None:
            self._viewer._roi_number_le.setText("")
            return
        if image != self.labels_image:
            image = self.labels_image
        tform = image.get_transform("canvas", "visual")
        px, py, *_ = (int(x) for x in tform.map(event.pos))
        try:
            pixel_value = image._data[py, px]
        # necessary when the mouse is outside the image
        except IndexError:
            return
        pixel_value = "" if pixel_value == 0 else pixel_value
        self._viewer._roi_number_le.setText(f"{pixel_value}")

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
