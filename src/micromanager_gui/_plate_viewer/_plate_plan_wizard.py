from __future__ import annotations

from typing import TYPE_CHECKING

from pymmcore_widgets.useq_widgets import WellPlateWidget
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QWizard, QWizardPage

if TYPE_CHECKING:
    import useq


class PlatePlanWizard(QWizard):
    """A wizard for creating a plate plan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plate Plan Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        self.setMinimumWidth(600)
        self.setMinimumHeight(600)
        self.resize(600, 600)

        self._plate_plan: useq.WellPlatePlan | None = None

        # Set the button layout to include Cancel button
        self.setButtonLayout(
            [
                QWizard.WizardButton.Stretch,
                QWizard.WizardButton.CancelButton,
                QWizard.WizardButton.NextButton,
            ]
        )

        first_page = _QuestionPage(self)
        self._well_selection_page = _WellSelectionPage(self)
        self.addPage(first_page)
        self.addPage(self._well_selection_page)

        if cancel_button := self.button(QWizard.WizardButton.CancelButton):
            cancel_button.clicked.connect(self._on_cancel_clicked)

        # Connect the Finish button to close the wizard
        if finish_button := self.button(QWizard.WizardButton.FinishButton):
            finish_button.clicked.connect(self._on_finish_clicked)

    def value(self) -> useq.WellPlatePlan | None:
        """Return the plate plan if it was created, otherwise None."""
        return self._plate_plan

    def _on_cancel_clicked(self) -> None:
        """Handle the cancel button click - always close the wizard."""
        self._plate_plan = None
        self.close()

    def _on_finish_clicked(self) -> None:
        """Handle the finish button click."""
        self._plate_plan = self._well_selection_page.value()
        self.close()


class _QuestionPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("Plate Plan Wizard")

        lbl = QLabel(
            "Did you use the `HCSWizard` to create a position list but manually "
            "modified it?\n\nIf you did, you can continue to the next step "
            "and select the wells you want to include in the plate plan."
        )
        layout = QVBoxLayout(self)
        layout.addWidget(lbl)

    def initializePage(self) -> None:
        """Initialize the page when it's shown."""
        super().initializePage()
        if wizard := self.wizard():
            wizard.setButtonText(QWizard.WizardButton.NextButton, "Yes")
            wizard.setButtonText(QWizard.WizardButton.CancelButton, "No")


class _WellSelectionPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Well Selection")

        self._well_plate_widget = WellPlateWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self._well_plate_widget)

    def initializePage(self) -> None:
        """Initialize the page when it's shown."""
        super().initializePage()
        if wizard := self.wizard():
            # Set the button layout for this page to only show Cancel and Finish
            wizard.setButtonLayout(
                [
                    QWizard.WizardButton.Stretch,
                    QWizard.WizardButton.CancelButton,
                    QWizard.WizardButton.FinishButton,
                ]
            )
            wizard.setButtonText(QWizard.WizardButton.FinishButton, "Finish")
            wizard.setButtonText(QWizard.WizardButton.CancelButton, "Cancel")

    def isComplete(self) -> bool:
        """Always return True so the Finish button is enabled."""
        return True

    def value(self) -> useq.WellPlatePlan:
        """Return the selected wells."""
        return self._well_plate_widget.value()  # type: ignore
