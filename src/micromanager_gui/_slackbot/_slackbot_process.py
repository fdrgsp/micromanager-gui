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


ROBOT = "\U0001f916"
ALARM = "\U0001f6a8"
MICROSCOPE = "\U0001f52c"


class SlackBotProcess(QProcess):
    """Process to run the SlackBot."""

    messageReceived = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.readyReadStandardError.connect(self.handle_error)

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
            f"{MICROSCOPE} Hello from Eve, the MicroManager's SlackBot! {MICROSCOPE}"
        )

    def send_message(self, message: str) -> None:
        """Send a message to the process.

        The message is written to the process's stdin so that it can be read by the
        process and sent to the Slack channel.
        """
        logging.info(f"SlackBotProcess -> received: '{message}'")
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
