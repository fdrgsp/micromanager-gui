# micromanager-gui [WIP]

[![License](https://img.shields.io/pypi/l/micromanager-gui.svg?color=green)](https://github.com/fdrgsp/micromanager-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/micromanager-gui.svg?color=green)](https://pypi.org/project/micromanager-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/micromanager-gui.svg?color=green)](https://python.org)
[![CI](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/micromanager-gui/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/micromanager-gui)

A Micro-Manager GUI based on [pymmcore-widgets](https://pymmcore-plus.github.io/pymmcore-widgets/) and [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/).

It has been designed to record calcium imaging experimets with or without optical stimulation using [Arduino](https://www.arduino.cc) and [Thorlabs](https://www.thorlabs.com) Components (paper in press...).

<img width="2672" alt="Screenshot 2025-05-30 at 12 45 31 PM" src="https://github.com/user-attachments/assets/9e6c95a0-5a61-4fbf-b408-4221b8e6092f" />

## Table of Contents

- [Installation](#installation)
- [Run the Micro-Manger GUI](#run-the-micro-manger-gui)
- [Run the Micro-Manger GUI with SlackBot](#run-the-micro-manger-gui-with-slackbot)
  - [SlackBot App Manifest example](#slackbot-app-manifest-example)
- [Run the Plate Viewer GUI](#run-the-plate-viewer-gui)

## Installation

Create a virtual environment and install the package using `pip` or `uv pip` (if you are using [uv](https://docs.astral.sh/uv/)).

```bash
pip install git+https://github.com/fdrgsp/micromanager-gui@micromanager-gui-calcium
```

Note: this is also installing the [PyQt6](https://pypi.org/project/PyQt6/) library for the GUI.

### Installing Micro-Manager

You also need to install the `Micro-Manager` device adapters and C++ core provided by [mmCoreAndDevices](https://github.com/micro-manager/mmCoreAndDevices#mmcoreanddevices). This can be done by following the steps described in the `pymmcore-plus` [documentation page](https://pymmcore-plus.github.io/pymmcore-plus/install/#installing-micro-manager-device-adapters).

## Run the Micro-Manger GUI

```bash
mmgui
```

By passing the `-c` or `-config` flag, you can specify the path of a micromanager configuration file you want to load. For example:

```bash
mmgui -c path/to/config.cfg
```

## Run the Micro-Manger GUI with SlackBot

By passing the `-s` or `-slack` boolean flag, you will be able to use a `SlackBot` to control the microscope. In particular, you will be able to start and stop the acquisition and to get the progress of the acquisition.

For example:

```bash
mmgui -c path/to/config.cfg -s True
```

To enable the `SlackBot`, you first need to follow the instructions in the [Slack Bolt documentation](https://slack.dev/bolt-python/tutorial/getting-started) to create your `Slack App` and get your `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`. In particular, go through the `Create an app`, `Tokens and installing apps` and `Setting up your project` sections (NOTE: you can use this [App Manifest](#slackbot-app-manifest-example) to create your `Slack App`).

The `OAuth & Permissions` Scope required are:

- `channels:history`
- `channels:read`
- `chat:write`
- `commands`

Since this `SlackBot` comunicates with the Micro-Manager through a set of `Slack commands`, you also need to set up the following command in your Slack App `Slash Commands` section:
- `/run`: Start the MDA Sequence
- `/cancel`: Cancel the current MDA Sequence
- `/progress`: Get the current MDA Sequence progress
- `/clear`: Clear the chat from the SlackBot messages
- `/mda`: Get the current MDASequence

Now that you have your `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`, you can either create a `.env` file in the root of this project containing the `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` variables (e.g. `SLACK_BOT_TOKEN=xoxb-...` and `SLACK_APP_TOKEN=xapp...`) or set them as global environment variables (e.g. `export SLACK_BOT_TOKEN=xoxb-...` and
`export SLACK_APP_TOKEN=xapp...`).

The last step is to grant access to the desired `Slack channel` to the `Slack App`. This can be done by inviting the `Slack App` to the desired `Slack channel`: right-click on the channel name, select `View channel details`, select the `Integrations` tab and `Add Apps`. You now need to add to the `.env` file (or as global environment) a variable named `CHANNEL_ID` containing the `Slack channel` ID.

After that, you can run the GUI with the `-s` or `-slack` flag set to `True` and start using the `Slack commands` to interact with the microscope.

#### SlackBot App Manifest example

```yaml
display_information:
  name: Eve
  description: A SlackBot for MicroManager & pymmcore-plus
  background_color: "#737373"
features:
  bot_user:
    display_name: Eve
    always_online: false
  slash_commands:
    - command: /run
      description: Run the MDA Acquisition
      should_escape: true
    - command: /cancel
      description: Cancel the ongoing MDA Acquisition
      should_escape: true
    - command: /progress
      description: Return the progress of the ongoing MDA Acquisition
      should_escape: true
    - command: /clear
      description: Clear the chat form the bot messages
      should_escape: true
    - command: /mda
      description: Return the current MDASequence.
      should_escape: true
oauth_config:
  scopes:
    bot:
      - chat:write
      - channels:read
      - channels:history
      - commands
settings:
  interactivity:
    is_enabled: true
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
```

## Run the Plate Viewer GUI

```bash
pv
```

<img width="1728" alt="Screenshot 2024-07-17 at 10 21 24 PM" src="https://github.com/user-attachments/assets/b81d0ad3-a6d4-4ada-97b5-b4734c7d8eea">
