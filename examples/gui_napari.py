from pymmcore_plus import CMMCorePlus
from qtpy.QtWidgets import QApplication

from micromanager_gui import MicroManagerGUI

app = QApplication([])
mmc = CMMCorePlus.instance()
gui = MicroManagerGUI(use_napari=True)
gui.show()
app.exec_()
