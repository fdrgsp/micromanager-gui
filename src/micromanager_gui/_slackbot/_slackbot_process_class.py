from __future__ import annotations

from qtpy.QtCore import QProcess, Signal, Slot


class SlackBotProcess(QProcess):
    """Process to run the SlackBot."""

    messageReceived = Signal(str)

    def __init__(self, slack_bot_token: str, slack_app_token: str) -> None:
        super().__init__()
        self._slack_bot_token = slack_bot_token
        self._slack_app_token = slack_app_token

        self.readyReadStandardOutput.connect(self.handle_message)

    def start(self) -> None:
        """Start the SlackBot process."""
        super().start(
            "python",
            [
                "src/micromanager_gui/_slackbot/_slackbot_process.py",
                self._slack_bot_token,
                self._slack_app_token,
            ],
        )

    @Slot()  # type: ignore
    def handle_message(self) -> None:
        """Handle the message and emit the signal containing the message."""
        message = self.readAllStandardOutput().data().decode()
        self.messageReceived.emit(message)
