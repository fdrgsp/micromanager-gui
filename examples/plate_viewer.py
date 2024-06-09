from qtpy.QtWidgets import QApplication

from micromanager_gui._plate_viewer._plate_viewer import PlateViewer

app = QApplication([])
pl = PlateViewer()
pl.show()
app.exec()
