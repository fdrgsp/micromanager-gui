# micromanager-gui

[![License](https://img.shields.io/pypi/l/micromanager-gui.svg?color=green)](https://github.com/fdrgsp/micromanager-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/micromanager-gui.svg?color=green)](https://pypi.org/project/micromanager-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/micromanager-gui.svg?color=green)](https://python.org)
[![CI](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/micromanager-gui/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/micromanager-gui/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/micromanager-gui)

A Micro-Manager GUI based on [pymmcore-widgets](https://pymmcore-plus.github.io/pymmcore-widgets/) and [pymmcore-plus](https://pymmcore-plus.github.io/pymmcore-plus/).

#

<img width="1728" alt="Screenshot 2024-03-19 at 2 42 00 PM" src="https://github.com/fdrgsp/micromanager-gui/assets/70725613/57224b3f-0e84-4f1c-a604-734a05b7547a">

&nbsp;

<figure>
  <img width="1728" alt="Screenshot 2024-03-23 at 8 21 40 PM" src="https://github.com/fdrgsp/micromanager-gui/assets/70725613/3d501382-c601-42ea-97d0-c71a97d7690b">
  <figcaption>This version uses napari as a viewer (set the -n flag to True).</figcaption>
</figure>

#

## Installation

```bash
pip install git+https://github.com/fdrgsp/micromanager-gui
```

### Installing PyQt or PySide

Since `micromanager-gui` relies on either the [PyQt](https://riverbankcomputing.com/software/pyqt/) or [PySide](https://www.qt.io/qt-for-python) libraries, you also **need** to install one of these packages. You can use any of the available versions of these libraries: [PyQt5](https://pypi.org/project/PyQt5/), [PyQt6](https://pypi.org/project/PyQt6/), [PySide2](https://pypi.org/project/PySide2/) or [PySide6](https://pypi.org/project/PySide6/). For example, to install [PyQt6](https://riverbankcomputing.com/software/pyqt/download), you can use:

```sh
pip install PyQt6
```

### Installing Micro-Manager

You also need to install the `Micro-Manager` device adapters and C++ core provided by [mmCoreAndDevices](https://github.com/micro-manager/mmCoreAndDevices#mmcoreanddevices). This can be done by following the steps described in the `pymmcore-plus` [documentation page](https://pymmcore-plus.github.io/pymmcore-plus/install/#installing-micro-manager-device-adapters).

### Installing napari

To use `napari` as the viewer, also run:

```bash
pip install napari
```

## To run the GUI

```bash
python -m micromanager_gui
```

- set the `-c` flag to the path of the `Micro-Manager` configuration file to directly load the configuration.
- set the `-n` flag to `True` to use `napari` as the viewer (e.g., `python -m micromanager_gui -n True`)
