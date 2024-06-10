from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget


class _GraphWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._fov: str = ""

        self._combo = QComboBox(self)
        # NOTE this is just a test, here we need to add proper names
        self._combo.addItems(["None", "test1", "test2", "test3", "test4"])
        self._combo.currentTextChanged.connect(self._on_combo_changed)

        # Create a figure and a canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a layout and add the canvas to it
        layout = QVBoxLayout(self)
        layout.addWidget(self._combo)
        layout.addWidget(self.canvas)

    @property
    def fov(self) -> str:
        return self._fov

    @fov.setter
    def fov(self, fov: str) -> None:
        self._fov = fov
        self._on_combo_changed(self._combo.currentText())

    def _on_combo_changed(self, test: str) -> None:
        if test == "None" or not self._fov:
            self._combo.setCurrentText("None")
            # clear the plot
            self.figure.clear()
            self.canvas.draw()
            return

        # NOTE this is just a test, here we need to add proper graphs
        x = np.linspace(0, 2 * np.pi, 100)
        if test == "test1":
            y = np.sin(x) + np.random.rand(100) * 0.1
        elif test == "test2":
            y = np.cos(x) + np.random.rand(100) * 0.1
        elif test == "test3":
            y = np.tan(x) + np.random.rand(100) * 10
        elif test == "test4":
            y = np.exp(x) + np.random.rand(100) * 10

        # clear the plot
        self.figure.clear()
        # maybe use  ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8]) instead of subplot
        ax = self.figure.add_subplot(111)
        # set title
        ax.set_title(f"{self._fov} - {test}")
        ax.plot(x, y)
        # Draw the plot
        self.canvas.draw()
