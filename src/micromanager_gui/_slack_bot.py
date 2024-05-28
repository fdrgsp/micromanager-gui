from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pyngrok import conf, ngrok
from qtpy.QtCore import QObject, Signal
from rich import print
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

CHANNEL_ID = "#calcium"
PORT = 3000


class SlackBot(QObject):
    slackBotSignal = Signal(str)

    """Slack bot for sending messages to a channel."""

    def __init__(self) -> None:
        super().__init__()
        # add slack token to environment variables
        ENV_PATH = Path(__file__).parent.parent.parent / ".env"
        loaded = load_dotenv(ENV_PATH)
        if not loaded:
            warnings.warn(f"Failed to load .env file at {ENV_PATH}", stacklevel=2)
        SLACK_TOKEN = os.getenv("SLACK_TOKEN")
        if SLACK_TOKEN is None:
            warnings.warn(
                "SLACK_TOKEN is not set in the environment variables", stacklevel=2
            )
        # add ngrok authtoken to environment variables
        NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
        if NGROK_AUTHTOKEN is None:
            warnings.warn(
                "NGROK_AUTHTOKEN is not set in the environment variables", stacklevel=2
            )

        # slack client
        self._slack_client = WebClient(token=SLACK_TOKEN) if SLACK_TOKEN else None
        # slack channel
        self._channel_id = CHANNEL_ID

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.add_url_rule(
            "/slack/events", view_func=self.slack_events, methods=["POST"]
        )

        if NGROK_AUTHTOKEN is not None:
            # Configure ngrok and start tunnel
            conf.get_default().auth_token = NGROK_AUTHTOKEN
            self.public_url = ngrok.connect(PORT)
            print(f" * ngrok tunnel \"{self.public_url}\" -> 'http://127.0.0.1:{PORT}'")
            # NOTE: everytime ngrok is started, a new public URL is generated therefore
            # we need to update the Request URL in the Slack API Event Subscriptions:
            # Request URL: https://<public_url>/slack/events (https://api.slack.com/).

            # Run Flask server in a separate thread
            threading.Thread(target=self.run_flask_app).start()

    @property
    def slack_client(self) -> WebClient | None:
        return self._slack_client

    @property
    def channel_id(self) -> str:
        return self._channel_id

    def send_message(self, text: str) -> None:
        """Send a message to the Slack channel."""
        if self._slack_client is None:
            return
        try:
            response = self._slack_client.chat_postMessage(
                channel=self._channel_id, text=text
            )
            assert response["ok"]
        except SlackApiError as e:
            warnings.warn(
                f"Failed to send message to slack: {e.response['error']}",
                stacklevel=2,
            )

    def handle_event(self, event: dict[str, Any]) -> None:
        """Handle the event from the Slack API."""
        # print('\nSlackBot Event:', event)

        event_type = event.get("type")
        if event_type == "message":
            text = event.get("text")
            self.slackBotSignal.emit(text)

    def slack_events(self) -> Any:
        """Handle the Slack events."""
        data = request.json

        if "challenge" in data:
            return jsonify({"challenge": data["challenge"]})

        if "event" in data:
            event = data["event"]
            self.handle_event(event)

        return "", 200

    def run_flask_app(self) -> None:
        self.app.run(port=PORT)
