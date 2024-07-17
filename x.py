from qtpy.QtWidgets import QApplication

from micromanager_gui._plate_viewer._plate_map import PlateMapWidget

app = QApplication([])

viewer = PlateMapWidget()
viewer.show()

app.exec()
