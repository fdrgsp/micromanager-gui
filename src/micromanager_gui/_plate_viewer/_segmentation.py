from pathlib import Path
from typing import Generator, NamedTuple

import numpy as np
import tifffile
from cellpose import models
from qtpy.QtWidgets import QWidget
from superqt.utils import create_worker
from tqdm import tqdm

from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader


class CellposeOut(NamedTuple):
    masks: np.ndarray
    flows: np.ndarray
    styles: np.ndarray
    diams: np.ndarray


class _CellposeSegmentation(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
    ) -> None:
        super().__init__(parent)

        self._data = data

        self._labels: dict[str, CellposeOut] = {}

        # TODO:
        # - add combo to select model. If custom, add lineedit and browse button
        # - add a channel selector
        # - add other options
        # - add use gpu checkbox
        # - add a button to segment the data
        # - add a Qt progress bar
        # - add a way to plot the results

    @property
    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader | None) -> None:
        self._data = data

    @property
    def labels(self) -> dict[str, CellposeOut]:
        return self._labels

    def _segment(self) -> Generator[str, None, None]:
        if self.data is None:
            return

        model = models.Cellpose(gpu=True, model_type="cyto")
        channel = [0, 0]
        diameter = 0
        path = "/Users/fdrgsp/Desktop/labels"

        pos = self.data.sequence.sizes["p"]  # type: ignore
        progress_bar = tqdm(range(pos))
        for p in progress_bar:
            progress_bar.set_description(f"Segmenting position {p+1} of {pos}")
            # get the data
            data, meta = self.data.isel(p=p, metadata=True)
            # max projection
            data_max = data.max(axis=0)
            # perform cellpose on each time point
            cyto_frame = data_max
            masks, flows, styles, diams = model.eval(
                cyto_frame, diameter=diameter, channels=channel
            )
            # get position name from metadata
            pos_name = meta[0].get("Event", {}).get("pos_name", f"pos_{p}")
            self._labels[f"{pos_name}_p{p}"] = CellposeOut(masks, flows, styles, diams)
            # save to disk
            tifffile.imsave(Path(path) / f"{pos_name}_p{p}.tif", masks)

            yield f"Segmented position {p+1} of {pos} (well {pos_name})"

    def segment(self) -> None:
        create_worker(
            self._segment,
            _start_thread=True,
            _connect={"yielded": self._print_state},
        )

    def _print_state(self, state: str) -> None:
        print(state)
