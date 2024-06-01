from __future__ import annotations

import json
import logging
import warnings
from typing import Any

from qtpy.QtCore import QProcess, Signal, Slot
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)


ROBOT = ":robot:"
ALARM = ":rotating_light:"
MICROSCOPE = ":microscope:"


class SlackBotProcess(QProcess):
    """Process to run the SlackBot."""

    messageReceived = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.readyReadStandardError.connect(self.handle_error)

    def stop(self) -> None:
        """Stop the SlackBot process."""
        self.kill()
        self.waitForFinished()

    def start(self) -> None:
        """Start the SlackBot in a new process.

        The process is started with the 'python' interpreter and the path to the
        '_slackbot.py' script (which contains the SlackBot class).
        """
        super().start("python", ["src/micromanager_gui/_slackbot/_slackbot.py"])
        if not self.waitForStarted():  # Check if the process started correctly
            msg = f"SlackBotProcess -> {ALARM} Failed to start SlackBotProcess! {ALARM}"
            logging.error(msg)
            warnings.warn(msg, stacklevel=2)
        else:
            logging.info(f"SlackBotProcess -> {ROBOT} SlackBotProcess started! {ROBOT}")

        self.send_message(
            {
                "icon_emoji": MICROSCOPE,
                "text": "Hello from Eve, the MicroManager's SlackBot!\n",
                # "- `/run` -> Start the MDA Sequence\n"
                # "- `/cancel` -> Cancel the current MDA Sequence\n"
                # "- `/progress` -> Get the current MDA Sequence progress",
            }
        )

    def send_message(self, message: str | dict[str, Any]) -> None:
        """Send a message to the process.

        The message is written to the process's stdin so that it can be read by the
        process and sent to the Slack channel.
        """
        logging.info(f"SlackBotProcess -> received: '{message}'")

        if isinstance(message, dict):
            text = message.get("text", "")
            emoji = message.get("icon_emoji", "")
            message = json.dumps({"icon_emoji": emoji, "text": text})

        # send message to the process with a newline
        self.write((message + "\n").encode())
        # ensure the bytes are written
        if not self.waitForBytesWritten(1000):
            logging.error(
                f"SlackBotProcess -> Failed to write '{message}' to the process!"
            )
        else:
            logging.info(f"SlackBotProcess -> sent: '{message}'")

    @Slot()  # type: ignore [misc]
    def handle_error(self) -> None:
        """Handle the error sent by the SlackBot in the new process process.

        This method is called when the process sends an error to stderr.
        """
        error = self.readAllStandardError().data().decode()
        logging.error(f"SlackBotProcess -> error received: {error}")
