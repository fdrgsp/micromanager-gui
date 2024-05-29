from __future__ import annotations

import logging
import warnings

from qtpy.QtCore import QProcess, Signal, Slot
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)


class SlackBotProcess(QProcess):
    """Process to run the SlackBot."""

    messageReceived = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.readyReadStandardOutput.connect(self.handle_message)

    def start(self) -> None:
        """Start the SlackBot in a new process.

        The process is started with the 'python' interpreter and the path to the
        '_slackbot.py' script (which contains the SlackBot class).
        """
        super().start("python", ["src/micromanager_gui/_slackbot/_slackbot.py"])
        if not self.waitForStarted():  # Check if the process started correctly
            logging.error("🚨 Failed to start SlackBotProcess! 🚨")
            warnings.warn("Failed to start the SlackBot process.", stacklevel=2)
        else:
            logging.info("🤖 SlackBotProcess started! 🤖")

        self.send_message("🔬 Hello from Eve, the MicroManager's SlackBot! 🔬")

    def send_message(self, message: str) -> None:
        """Send a message to the process.

        The message is written to the process's stdin so that it can be read by the
        process and sent to the Slack channel.
        """
        logging.info(f"send_message: {message}")
        # send message to the process with a newline
        self.write((message + "\n").encode())
        # ensure the bytes are written
        if not self.waitForBytesWritten(1000):
            logging.error("Failed to write message to the process!")
        else:
            logging.info("Message sent successfully!")

    @Slot()  # type: ignore [misc]
    def handle_message(self) -> None:
        """Handle the message sent by the SlackBot in the new process process.

        This method is called when the process sends a message to stdout. Once received,
        the message is emitted as a signal to be connected to a slot in MicroManagerGUI.
        """
        logging.info("handle_message")
        message = self.readAllStandardOutput().data().decode()
        logging.info(f"message: {message}")
        self.messageReceived.emit(message)
