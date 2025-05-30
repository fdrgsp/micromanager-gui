from __future__ import annotations

from pathlib import Path

import pytest
from cellpose import models

CUSTOM_MODEL_PATH = (
    Path(__file__).parent
    / "src"
    / "micromanager_gui"
    / "_cellpose"
    / "cellpose_models"
    / "cp3_img8_epoch7000_py"
)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_load_cellpose_models() -> None:
    model = models.CellposeModel(pretrained_model=CUSTOM_MODEL_PATH, gpu=False)
    assert model is not None, "Failed to load custom Cellpose model"
