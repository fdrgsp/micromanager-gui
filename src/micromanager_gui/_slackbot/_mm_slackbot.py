from __future__ import annotations

import json
import logging

from qtpy.QtCore import QObject, Signal
from rich.logging import RichHandler

from ._slackbot_process import SlackBotProcess

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)


class MMSlackBot(QObject):
    """SlackBot to send and receive messages from a Slack channel."""

    slackMessage = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        # start your app with the app token in a separate process
        self._slack_process = SlackBotProcess()
        self._slack_process.messageReceived.connect(self.handle_message_events)
        self._slack_process.start()

    def send_message(self, message: str) -> None:
        """Send a message to the Slack channel."""
        self._slack_process.send_message(message)

    def handle_message_events(self, message: str) -> None:
        """Handle all the message events."""
        message = json.loads(message)

        if isinstance(message, dict):
            event = message.get("event", {})
            text = event.get("text")
        else:
            # is a command
            text = message.replace("/", "")

        logging.info(f"MMSlackBot -> received: {text}")
        self.slackMessage.emit(text)

    def __del__(self) -> None:
        self._slack_process.stop()
