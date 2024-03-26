"""Run micromanager-gui as a script with `python -m micromanager_gui`.

set the `-c` flag to the path of the `Micro-Manager` configuration file to directly
load the configuration.

set the `-n` flag to `True` to use `napari` as the viewer
(e.g., `python -mmicromanager_gui -n True`)
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from qtpy.QtWidgets import QApplication

from micromanager_gui import MicroManagerGUI


def main(args: Sequence[str] | None = None) -> None:
    """Run the Micro-Manager GUI."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Enter string")
    parser.add_argument(
        "-n",
        "--napari",
        type=bool,
        default=False,
        help="Use napari as the viewer",
        nargs="?",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config file to load",
        nargs="?",
    )
    parsed_args = parser.parse_args(args)

    app = QApplication([])
    win = MicroManagerGUI(config=parsed_args.config, use_napari=parsed_args.napari)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
