from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from qtpy.QtCore import QObject, QProcess, Signal
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

RUN = "run"
STOP = "stop"
CANCEL = "cancel"
STATUS = "status"
CLEAR = "clear"
ALLOWED_COMMANDS = {RUN, STOP, CANCEL, STATUS, CLEAR}
SKIP = ["has joined the channel"]

CHANNEL_ID = "C074WAU4L3Z"  # calcium


class SlackBotWorker(QProcess):
    """Worker to run the SlackBot in a separate thread."""

    def __init__(self, app: App, slack_app_token: str) -> None:
        super().__init__()
        self._app = app
        self._slack_app_token = slack_app_token

    def run(self) -> None:
        handler = SocketModeHandler(self._app, self._slack_app_token)
        handler.start()


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

        # add slack token to environment variables
        ENV_PATH = Path(__file__).parent.parent.parent / ".env"
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
            self._app = App(token=SLACK_BOT_TOKEN)
            self._slack_client = WebClient(token=SLACK_BOT_TOKEN)
        except Exception as e:
            self._slack_client = None
            warnings.warn(f"Failed to initialize SlackBot: {e}", stacklevel=2)
            return

        @self._app.event("message")  # type: ignore [misc]
        def handle_message_events(body: dict, say: Callable[[str], None]) -> None:
            """Handle all the message events."""
            # say() sends a message to the channel where the event was triggered
            event = body.get("event", {})
            user_id = event.get("user")

            if user_id is None:
                return

            text = event.get("text")
            if text in ALLOWED_COMMANDS:
                # clear the chet from the messages sent by the bot
                if text == CLEAR:
                    self.clear_chat()
                    return
                # say(f"Hey there <@{user_id}>, you said {text}!")
                self.slackBotSignal.emit(text)
            else:
                say(
                    f"Sorry <@{user_id}>, only the following commands are allowed: "
                    f"{', '.join(ALLOWED_COMMANDS)}."
                )

        # start your app with the app token in a separate thread
        self._slack_worker = SlackBotWorker(self._app, SLACK_APP_TOKEN)
        self._slack_worker.start()

    @property
    def slack_client(self) -> WebClient | None:
        """Return the slack client."""
        return self._slack_client

    def _run_app(self) -> None:
        """Run the app."""
        SocketModeHandler(self._app, os.getenv("SLACK_APP_TOKEN")).start()

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

        bot_id = self._slack_client.auth_test()["user_id"]
        try:
            # fetch the history of the channel
            response = self._slack_client.conversations_history(channel=CHANNEL_ID)
            # check if the history was fetched successfully
            if not response["ok"]:
                return
            # iterate over every message
            for message in response["messages"]:
                # check if the message was sent by the bot or if should be skipped
                if message.get("user") != bot_id or any(
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
