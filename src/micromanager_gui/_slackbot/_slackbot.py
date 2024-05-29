from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from qtpy.QtCore import QObject, Signal
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ._slackbot_process_class import SlackBotProcess

RUN = "run"
STOP = "stop"
CANCEL = "cancel"
STATUS = "status"
CLEAR = "clear"
ALLOWED_COMMANDS = {RUN, STOP, CANCEL, STATUS, CLEAR}
SKIP = ["has joined the channel"]

CHANNEL_ID = "C074WAU4L3Z"  # calcium


class SlackBot(QObject):
    """SlackBot to send and receive messages from a Slack channel.

    To use the SlackBot you need to have your SLACK_BOT_TOKEN and SLACK_APP_TOKEN;
    you can follow this instruction to setup your Slack app:
    https://slack.dev/bolt-python/tutorial/getting-started.

    After that, you can either set the environment variables SLACK_BOT_TOKEN and
    SLACK_APP_TOKEN (e.g. in your terminal type `export SLACK_BOT_TOKEN=your_token` and
    `export SLACK_APP_TOKEN=your_token`) or create a `.env` file in the root of the
    project with both the tokens (e.g.SLACK_BOT_TOKEN=your_token and
    SLACK_APP_TOKEN=your_token).
    """

    slackBotSignal = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self._slack_client: WebClient | None = None
        self._bot_id: str | None = None

        # add slack token to environment variables
        ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
        loaded = load_dotenv(ENV_PATH)
        if not loaded:
            warnings.warn(f"Failed to load .env file at {ENV_PATH}", stacklevel=2)
            return
        SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
        if SLACK_BOT_TOKEN is None:
            warnings.warn(
                "SLACK_BOT_TOKEN is not set in the environment variables", stacklevel=2
            )
            return
        SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
        if SLACK_APP_TOKEN is None:
            warnings.warn(
                "SLACK_APP_TOKEN is not set in the environment variables", stacklevel=2
            )
            return

        # initializes your app with your bot token
        try:
            self._slack_client = WebClient(token=SLACK_BOT_TOKEN)
            self._bot_id = self._slack_client.auth_test()["user_id"]
        except Exception as e:
            self._slack_client = None
            warnings.warn(f"Failed to initialize SlackBot: {e}", stacklevel=2)
            return

        # start your app with the app token in a separate process
        self._slack_process = SlackBotProcess(SLACK_BOT_TOKEN, SLACK_APP_TOKEN)
        self._slack_process.messageReceived.connect(self.handle_message_events)
        self._slack_process.start()

    @property
    def slack_client(self) -> WebClient | None:
        """Return the slack client."""
        return self._slack_client

    def handle_message_events(self, body: str) -> None:
        """Handle all the message events."""
        body_dict = cast(dict, json.loads(body))
        event = body_dict.get("event", {})
        user_id = event.get("user")

        if user_id is None or user_id == self._bot_id:
            return

        text = event.get("text")
        if text in ALLOWED_COMMANDS:
            # clear the chet from the messages sent by the bot
            if text == CLEAR:
                self.clear_chat()
                return
            self.slackBotSignal.emit(text)
        else:
            self.send_message(
                f"Sorry <@{user_id}>, only the following commands are allowed: "
                f"{', '.join(ALLOWED_COMMANDS)}."
            )

    def send_message(self, text: str) -> None:
        """Send a message to a Slack channel."""
        if self._slack_client is None:
            return

        try:
            response = self._slack_client.chat_postMessage(
                channel=CHANNEL_ID, text=text
            )
            assert response["ok"]
        except SlackApiError as e:
            warnings.warn(f"Failed to send message: {e}", stacklevel=2)

    def clear_chat(self) -> None:
        """Clear the chat in the Slack channel.

        NOTE: only messages sent by the bot will be deleted.
        """
        if self._slack_client is None:
            return
        try:
            # fetch the history of the channel
            response = self._slack_client.conversations_history(channel=CHANNEL_ID)
            # check if the history was fetched successfully
            if not response["ok"]:
                return
            # iterate over every message
            for message in response["messages"]:
                # check if the message was sent by the bot or if should be skipped
                if message.get("user") != self._bot_id or any(
                    text in message.get("text") for text in SKIP
                ):
                    continue
                # delete each message
                self._slack_client.chat_delete(channel=CHANNEL_ID, ts=message["ts"])
                # add a delay between each API call
                time.sleep(0.1)
        except SlackApiError as e:
            warnings.warn(
                f"Failed to clear chat in slack: {e.response['error']}",
                stacklevel=2,
            )
