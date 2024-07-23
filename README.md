# micromanager-gui [WIP]

[![License](https://img.shields.io/pypi/l/micromanager-gui.svg?color=green)](https://github.com/fdrgsp/micromanager-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/micromanager-gui.svg?color=green)](https://pypi.org/project/micromanager-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/micromanager-gui.svg?color=green)](https://python.org)
[![CI](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/micromanager-gui/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/micromanager-gui)

A Micro-Manager minimal GUI based on [pymmcore-widgets](https://pymmcore-plus.github.io/pymmcore-widgets/) and [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/).


<img width="1840" alt="Screenshot 2024-06-03 at 11 51 08 PM" src="https://github.com/fdrgsp/micromanager-gui/assets/70725613/7c648147-fb18-4b3f-802c-5a074f57e8b0">

## Python version

The package is tested on Python 3.10 and 3.11.

## Installation

```bash
pip install git+https://github.com/fdrgsp/micromanager-gui@calcium
```

### Installing PyQt

Since `micromanager-gui` relies on the [PyQt](https://riverbankcomputing.com/software/pyqt/) library, you also **need** to install one of these packages. You can use any of the available versions of [PyQt6](https://pypi.org/project/PyQt6/) or [PyQt5](https://pypi.org/project/PyQt5/). For example, to install [PyQt6](https://riverbankcomputing.com/software/pyqt/download), you can use:

```sh
pip install PyQt6
```

Note: tests are running on [PyQt6](https://pypi.org/project/PyQt6/) and [PyQt5](https://pypi.org/project/PyQt5/).

### Installing Micro-Manager

You also need to install the `Micro-Manager` device adapters and C++ core provided by [mmCoreAndDevices](https://github.com/micro-manager/mmCoreAndDevices#mmcoreanddevices). This can be done by following the steps described in the `pymmcore-plus` [documentation page](https://pymmcore-plus.github.io/pymmcore-plus/install/#installing-micro-manager-device-adapters).

## To run the Micro-Manger GUI

```bash
mmgui
```

By passing the `-c` or `-config` flag, you can specify the path of a micromanager configuration file you want to load. For example:

```bash
mmgui -c path/to/config.cfg
```

## To run the Micro-Manger GUI with SlackBot

By passing the `-s` or `-slack` boolean flag, you will be able to use a `SlackBot` to control the microscope. In particular, you will be able to start and stop the acquisition and to get the progress of the acquisition.

For example:

```bash
mmgui -c path/to/config.cfg -s True
```

To enable the `SlackBot`, you first need to follow the instructions in the [Slack Bolt documentation](https://slack.dev/bolt-python/tutorial/getting-started) to create your `Slack App` and get your `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`. In particular, go through the `Create an app`, `Tokens and installing apps` and `Setting up your project` sections.

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

Now that you have your `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`, you can either create a `.env` file in the root of this project containing the `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` variables (e.g. `SLACK_BOT_TOKEN=xoxb-...` and `SLACK_APP_TOKEN=xapp...`) or set them as global environment variables (e.g. `export SLACK_BOT_TOKEN=xoxb-...` and
`export SLACK_APP_TOKEN=xapp...`).

The last step is to grant access to the desired `Slack channel` to the `Slack App`. This can be done by inviting the `Slack App` to the desired `Slack channel`: right-click on the channel name, select `View channel details`, select the `Integrations` tab and `Add Apps`. You now need to add to the `.env` file (or as global environment) a variable named `CHANNEL_ID` containing the `Slack channel` ID.

After that, you can run the GUI with the `-s` or `-slack` flag set to `True` and start using the `Slack commands` to interact with the microscope.
