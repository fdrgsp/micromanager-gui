"""A Micro-Manager GUI based on pymmcore-widgets and pymmcore-plus."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("micromanager-gui")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Federico Gasparoli"
__email__ = "federico.gasparoli@gmail.com"


from micromanager_gui._plate_viewer import PlateViewer

from ._batch_segmentation import CellposeBatchSegmentation
from ._main_window import MicroManagerGUI

__all__ = ["MicroManagerGUI", "PlateViewer", "CellposeBatchSegmentation"]
