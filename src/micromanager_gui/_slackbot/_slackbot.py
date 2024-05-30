from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Generator, cast

from dotenv import load_dotenv
from qtpy.QtCore import QObject
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
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

# add environment variables from .env file
ENV_PATH = Path(__file__).parent / ".env"
loaded = load_dotenv(ENV_PATH)
if not loaded:
    env_error_msg = f"SlackBot -> Failed to load '.env' file at {ENV_PATH}!"
    logging.error(env_error_msg)
    raise FileNotFoundError(env_error_msg)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
if SLACK_BOT_TOKEN is None:
    bot_token_error = "SlackBot -> 'SLACK_BOT_TOKEN' is not set!"
    logging.error(bot_token_error)
    raise ValueError(bot_token_error)
else:
    logging.info("SlackBot -> 'SLACK_BOT_TOKEN' set correctly!")

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
if SLACK_APP_TOKEN is None:
    app_token_error = "SlackBot -> 'SLACK_APP_TOKEN' is not set!"
    logging.error(app_token_error)
    raise ValueError(app_token_error)
else:
    logging.info("SlackBot -> 'SLACK_APP_TOKEN' set correctly!")


class SlackBot(QObject):
    """Class that will be called when the process is started.

    The 'handle_message_events' method with the '@app.event("message")' decorator
    will be called every time a message event is received in the slack channel.

    The message event is then written to 'stdout' so that in the 'SlackBotProcess'
    class (_slackbot_process_class.py) the message can be read from the process and
    emitted as a signal (messageReceived) that is connected to a slot in
    'MicroManagerGUI'.
    """

    def __init__(self) -> None:
        super().__init__()

        logging.info("SlackBot -> initializing...")

        self.listen_thread = threading.Thread(target=self.listen_for_messages)
        self.listen_thread.start()

        self._app = App(token=SLACK_BOT_TOKEN)
        self._slack_client = self._app.client
        self._bot_id = self._slack_client.auth_test().data["user_id"]

        @self._app.event("message")  # type: ignore [misc]
        def handle_message_events(body: dict, say: Callable) -> None:
            """Handle all the message events."""
            event = cast(dict, body.get("event", {}))
            user_id = event.get("user")
            text = event.get("text")

            logging.info(f"SlackBot -> forewarding message: {text}")

            sys.stdout.write(json.dumps(body))
            sys.stdout.flush()

            # ignore messages from the bot itself
            if user_id is None or user_id == self._bot_id:
                return

            text = cast(str, text)
            if text.lower() in ALLOWED_COMMANDS:
                return

            say(
                f"Sorry <@{user_id}>, only the following commands are allowed: "
                f"{', '.join(ALLOWED_COMMANDS)}."
            )

        handler = SocketModeHandler(self._app, SLACK_APP_TOKEN)
        handler.start()
        logging.info("SlackBot -> 'SocketModeHandler' started")

    def listen_for_messages(self) -> Generator[str, None, None]:
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
