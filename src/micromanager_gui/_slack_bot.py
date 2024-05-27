from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

CHANNEL_ID = "#calcium"


class SlackBot:
    """Slack bot for sending messages to a channel."""

    def __init__(self) -> None:
        # add slack tocken to environment variables
        ENV_PATH = Path(__file__).parent.parent.parent / ".env"
        loaded = load_dotenv(ENV_PATH)
        if not loaded:
            warnings.warn(f"Failed to load .env file at {ENV_PATH}", stacklevel=2)
        SLACK_TOKEN = os.getenv("SLACK_TOKEN")
        if SLACK_TOKEN is None:
            warnings.warn(
                "SLACK_TOKEN is not set in the environment variables", stacklevel=2
            )

        # slack client
        self.SLACK_CLIENT = WebClient(token=SLACK_TOKEN) if SLACK_TOKEN else None
        # slack channel
        self.CHANNEL_ID = CHANNEL_ID

    @property
    def slack_client(self) -> WebClient | None:
        return self.SLACK_CLIENT

    @property
    def channel_id(self) -> str:
        return self.CHANNEL_ID

    def send_message(self, text: str) -> None:
        if self.slack_client is None:
            return
        try:
            response = self.slack_client.chat_postMessage(
                channel=self.channel_id, text=text
            )
            assert response["ok"]
        except SlackApiError as e:
            warnings.warn(
                f"Failed to send message to slack: {e.response['error']}",
                stacklevel=2,
            )
