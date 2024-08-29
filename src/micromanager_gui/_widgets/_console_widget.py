from qtpy.QtWidgets import QWidget, QApplication, QVBoxLayout
from pyqtconsole.console import PythonConsole
from pymmcore_plus import CMMCorePlus


class _ConsoleWidget(QWidget):
    def __init__(self,
                 parent: QWidget | None = None,
                 mmcore: CMMCorePlus | None = None):
        super().__init__()
        self.console = PythonConsole()
        self.console.eval_in_thread()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.console)
        self.setLayout(self.layout)