from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymmcore_widgets import InstallWidget

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


class _InstallWidget(InstallWidget):
    def __init__(self, parent: QWidget | None = None, **kwargs: Any) -> None:
        super().__init__(parent)
        self.resize(800, 400)
