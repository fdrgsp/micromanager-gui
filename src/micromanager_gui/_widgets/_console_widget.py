from pymmcore_plus import CMMCorePlus
from pyqtconsole.console import PythonConsole
from qtpy.QtWidgets import QVBoxLayout, QWidget


class _ConsoleWidget(QWidget):
    def __init__(
        self, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ):
        super().__init__()
        self.console = PythonConsole()
        self.console.eval_in_thread()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.console)
        self.setLayout(self.layout)
