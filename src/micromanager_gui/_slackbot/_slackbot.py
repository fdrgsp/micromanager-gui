from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from slack_bolt import App
from slack_sdk.errors import SlackApiError

CHANNEL_ID = "C074WAU4L3Z"  # calcium
RUN = "run"
STOP = "stop"
CANCEL = "cancel"
STATUS = "status"
ALLOWED_COMMANDS = {RUN, STOP, CANCEL, STATUS}


# To use the SlackBot you need to have your SLACK_BOT_TOKEN and SLACK_APP_TOKEN;
# you can follow this instruction to setup your Slack app:
# https://slack.dev/bolt-python/tutorial/getting-started.

# After that, you can either set the environment variables SLACK_BOT_TOKEN and
# SLACK_APP_TOKEN (e.g. in your terminal type `export SLACK_BOT_TOKEN=your_token` and
# `export SLACK_APP_TOKEN=your_token`) or create a `.env` file in the root of the
# project with both the tokens (e.g.SLACK_BOT_TOKEN=your_token and
# SLACK_APP_TOKEN=your_token).

logging.basicConfig(
    filename=Path(__file__).parent / "slackbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.info("Starting...")

# add environment variables from .env file
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    loaded = load_dotenv(ENV_PATH)
    if not loaded:
        env_error_msg = f"SlackBot -> Failed to load '.env' file at {ENV_PATH}!"
        logging.error(env_error_msg)
        raise FileNotFoundError(env_error_msg)

logging.info("SlackBot -> getting 'SLACK_BOT_TOKEN' from environment variables...")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
if SLACK_BOT_TOKEN is None:
    bot_token_error = "SlackBot -> 'SLACK_BOT_TOKEN' is not found!"
    logging.error(bot_token_error)
    raise ValueError(bot_token_error)
else:
    logging.info("SlackBot -> 'SLACK_BOT_TOKEN' set correctly!")

logging.info("SlackBot -> getting 'SLACK_APP_TOKEN' from environment variables...")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
if SLACK_APP_TOKEN is None:
    app_token_error = "SlackBot -> 'SLACK_APP_TOKEN' is not found!"
    logging.error(app_token_error)
    raise ValueError(app_token_error)
else:
    logging.info("SlackBot -> 'SLACK_APP_TOKEN' set correctly!")


class SlackBot:
    """SlackBot class to send messages to a Slack channel.

    The message event is then written to 'stdout' so that in the 'SlackBotProcess'
    class (_slackbot_process_class.py) the message can be read from the process and
    emitted as a signal (messageReceived) that is connected to a slot in
    'MicroManagerGUI'.
    """

    def __init__(self) -> None:
        logging.info("SlackBot -> initializing...")

        self._app = App(token=SLACK_BOT_TOKEN, ignoring_self_events_enabled=True)
        self._slack_client = self._app.client
        self._bot_id = self._slack_client.auth_test().data["user_id"]

        self.listen_thread = threading.Thread(target=self.listen_for_messages)
        self.listen_thread.start()

    def listen_for_messages(self) -> Generator[str, None, None]:
        """Listen for messages from stdin."""
        logging.info(
            f"SlackBot -> Listening thread started: {self.listen_thread.is_alive()}"
        )
        while True:
            if message := sys.stdin.readline().strip():
                logging.info(f"SlackBot -> (listening) message received: {message}")
                logging.info(f"SlackBot -> (listening) forewarding message: {message}")
                self.send_message(message)

    def send_message(self, message: str) -> None:
        """Send a message to a channel."""
        logging.info(f"SlackBot -> sending message: {message}")
        try:
            response = self._slack_client.chat_postMessage(
                channel=CHANNEL_ID, text=message
            )
            assert response["ok"]
        except SlackApiError as e:
            msg = f"Failed to send message: {e.response['error']}"
            logging.error(f"SlackBot -> {msg}")


if __name__ == "__main__":
    slackbot = SlackBot()
