from __future__ import annotations

import json
import logging
from typing import cast

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
        message_dict = cast(dict, json.loads(message))
        event = message_dict.get("event", {})
        text = event.get("text")
        logging.info(f"MMSlackBot, message received: {text}")
        self.slackMessage.emit(text)
