from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from qtpy.QtCore import QObject
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

CHANNEL_ID = "C074WAU4L3Z"  # calcium
RUN = "run"
STOP = "stop"
CANCEL = "cancel"
STATUS = "status"
CLEAR = "clear"
ALLOWED_COMMANDS = {RUN, STOP, CANCEL, STATUS, CLEAR}


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
)


# add environment variables from .env file
ENV_PATH = Path(__file__).parent / ".env"
loaded = load_dotenv(ENV_PATH)
if not loaded:
    env_error_msg = f"SlackBot -> Failed to load .env file at {ENV_PATH}!"
    logging.error(env_error_msg)
    sys.stderr.write(env_error_msg)
    sys.stderr.flush()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
if SLACK_BOT_TOKEN is None:
    bot_token_error_msg = (
        "SlackBot -> SLACK_BOT_TOKEN is not set in the environment variables!"
    )
    logging.error(bot_token_error_msg)
    sys.stderr.write(bot_token_error_msg)
    sys.stderr.flush()
else:
    logging.info("SlackBot -> SLACK_BOT_TOKEN set correctly!")

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
if SLACK_APP_TOKEN is None:
    app_token_error_msg = (
        "SlackBot -> SLACK_APP_TOKEN is not set in the environment variables!"
    )
    logging.error(app_token_error_msg)
    sys.stderr.write(app_token_error_msg)
    sys.stderr.flush()
else:
    logging.info("SlackBot -> SLACK_APP_TOKEN set correctly!")


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

        try:
            self._slack_client = WebClient(token=SLACK_BOT_TOKEN)
            self._bot_id = self._slack_client.auth_test().data["user_id"]
        except Exception as e:
            web_client_error_msg = f"SlackBot -> Failed to create WebClient: {e}"
            logging.error(web_client_error_msg)

        try:
            app = App(token=SLACK_BOT_TOKEN)
        except Exception as e:
            app_error_msg = f"SlackBot -> Failed to create App: {e}"
            logging.error(app_error_msg)

        logging.info("SlackBot -> SlackBot initialized!")
        logging.info(f"SlackBot -> Bot ID: {self._bot_id}")

        self.running = True
        self.listen_thread = threading.Thread(target=self.listen_for_messages)
        self.listen_thread.start()
        logging.info(
            f"SlackBot -> Listening thread started: {self.listen_thread.is_alive()}"
        )

        @app.event("message")  # type: ignore [misc]
        def handle_message_events(body: dict) -> None:
            """Handle all the message events."""
            event = body.get("event", {})
            user_id = event.get("user")
            text = event.get("text")

            # ignore messages from the bot itself
            if user_id is None or user_id == self._bot_id:
                return

            logging.info(f"SlackBot -> message received: {text}")
            logging.info(f"SlackBot -> forewarding message: {text}")

            sys.stdout.write(json.dumps(body))
            sys.stdout.flush()

            if text not in ALLOWED_COMMANDS:
                self.send_message(
                    f"Sorry <@{user_id}>, only the following commands are allowed: "
                    f"{', '.join(ALLOWED_COMMANDS)}."
                )

        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        handler.start()

    def listen_for_messages(self) -> None:
        while self.running:
            if message := sys.stdin.readline().strip():
                logging.info(f"SlackBot -> message received: {message}")
                logging.info(f"SlackBot -> forewarding message: {message}")
                self.send_message(message)
            else:
                time.sleep(0.1)

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
    if SLACK_BOT_TOKEN is None or SLACK_APP_TOKEN is None:
        sys.exit()

    proc = SlackBot()
    sys.exit()
