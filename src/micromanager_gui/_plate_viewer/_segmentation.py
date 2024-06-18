from typing import TYPE_CHECKING

from cellpose import models
from qtpy.QtWidgets import QWidget
from tqdm import tqdm

from micromanager_gui._readers._ome_zarr_reader import OMEZarrReader
from micromanager_gui._readers._tensorstore_zarr_reader import TensorstoreZarrReader

if TYPE_CHECKING:
    import numpy as np


class _CellposeSegmentation(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        data: TensorstoreZarrReader | OMEZarrReader | None = None,
    ) -> None:
        super().__init__(parent)

        self._data = data

        self._labels: dict[str, np.ndarray] = {}

        # TODO:
        # - add combo to select model. If custom, add lineedit and browse button
        # - add a channel selector
        # - add other options
        # - add use gpu checkbox
        # - add a button to segment the data

    @property
    def data(self) -> TensorstoreZarrReader | OMEZarrReader | None:
        return self._data

    @data.setter
    def data(self, data: TensorstoreZarrReader | OMEZarrReader | None) -> None:
        self._data = data

    def segment(self) -> None:
        if self.data is None:
            return

        model = models.Cellpose(gpu=True, model_type="cyto")
        channel = [0, 0]
        diameter = 0
        # model.device = "mps"
        # print('__________________', model.device)

        # pos = self.data.sequence.sizes["p"]
        pos = 0

        for p in tqdm(range(pos)):
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
            pos_name = meta[p].get("Events", {}).get("pos_name", f"pos_{p}")
            self._labels[pos_name] = masks
            # TODO: save to disk
