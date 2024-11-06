# micromanager-gui [WIP]

[![License](https://img.shields.io/pypi/l/micromanager-gui.svg?color=green)](https://github.com/fdrgsp/micromanager-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/micromanager-gui.svg?color=green)](https://pypi.org/project/micromanager-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/micromanager-gui.svg?color=green)](https://python.org)
[![CI](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/micromanager-gui/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/micromanager-gui)

A Micro-Manager GUI based on [pymmcore-widgets](https://pymmcore-plus.github.io/pymmcore-widgets/) and [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/).


<img width="1829" alt="Screenshot 2024-11-05 at 9 49 58 PM" src="https://github.com/user-attachments/assets/20245c3c-4d31-42b8-b859-32d6ea4c0d87">



## Python version

The package is tested on Python 3.10 and 3.11.

## Installation

```bash
pip install git+https://github.com/fdrgsp/micromanager-gui
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

