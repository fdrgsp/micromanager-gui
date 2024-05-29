import json
import logging
import sys

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

logging.basicConfig(
    filename="/Users/fdrgsp/Desktop/process.log",
    level=logging.INFO,
)

# this is the main function that will be called when the process is started.
# the 'handle_message_events' method with the '@app.event("message")' decorator
# will be called every time a message event is received in the slack channel.
# the message event is then written to stdout so that in the 'SlackBotProcess'
# class, the message can be read from the process and emitted as a signal that
# can be connected to a slot in 'MicroManagerGUI'.


def main() -> None:
    slack_bot_token = sys.argv[1]
    slack_app_token = sys.argv[2]

    logging.info(f"slack_bot_token: {slack_bot_token}")

    app = App(token=slack_bot_token)

    @app.event("message")  # type: ignore [misc]
    def handle_message_events(body: dict) -> None:
        """Handle all the message events."""
        # write body to stdout
        sys.stdout.write(json.dumps(body))
        sys.stdout.flush()
        logging.info(f"message: {body}")

    handler = SocketModeHandler(app, slack_app_token)
    handler.start()


if __name__ == "__main__":
    main()
