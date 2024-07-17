from qtpy.QtWidgets import QApplication

from micromanager_gui._plate_viewer._plate_map import PlateMap

app = QApplication([])

viewer = PlateMap()
viewer.show()

app.exec()
