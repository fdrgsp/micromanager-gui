import json
import re
from pathlib import Path
from typing import cast

import numpy as np
from tqdm import tqdm

from micromanager_gui._plate_viewer._util import GENOTYPE_MAP, TREATMENT_MAP, ROIData
from micromanager_gui.readers import TensorstoreZarrReader


def _filter_data(path_list: list[Path]) -> list[Path]:
    filtered_paths: list[Path] = []

    # the json file names should be in the form A1_0000.json
    for f in path_list:
        if f.name in {GENOTYPE_MAP, TREATMENT_MAP}:
            continue
        # skip hidden files
        if f.name.startswith("."):
            continue

        name_no_suffix = f.name.removesuffix(f.suffix)  # A1_0000 or A1_0000_p0
        split_name = name_no_suffix.split("_")  # ["A1","0000"]or["A1","0000","p0"]

        if len(split_name) == 2:
            well, fov = split_name
        elif len(split_name) == 3:
            well, fov, pos = split_name
        else:
            continue

        # validate well format, only letters and numbers
        if not re.match(r"^[a-zA-Z0-9]+$", well):
            continue

        # validate fov format, only numbers
        if len(split_name) == 3:
            if not fov.isdigit():
                continue
            if not pos[1:].isdigit():
                continue

        filtered_paths.append(f)

        return filtered_paths


def _load_and_set_data_from_json(path: Path) -> dict[str, dict[str, ROIData]]:
    """Load the analysis data from the given JSON file."""
    _pv_analysis_data = {}
    json_files = _filter_data(list(path.glob("*.json")))
    # loop over the files in the directory
    for f in tqdm(json_files, desc="Loading Analysis Data"):
        # get the name of the file without the extensions
        well = f.name.removesuffix(f.suffix)
        # create the dict for the well
        _pv_analysis_data[well] = {}
        # open the data for the well
        with open(f) as file:
            try:
                data = cast(dict, json.load(file))
            except json.JSONDecodeError as e:
                print(f"Error loading {f}: {e}")
                _pv_analysis_data = data
            # if the data is empty, continue
            if not data:
                continue
            # loop over the rois
            for roi in data.keys():
                if not roi.isdigit():
                    # this is the case of global data
                    # (e.g. cubic or linear global connectivity)
                    _pv_analysis_data[roi] = data[roi]
                    continue
                # get the data for the roi
                fov_data = cast(dict, data[roi])
                # remove any key that is not in ROIData
                for key in list(fov_data.keys()):
                    if key not in ROIData.__annotations__:
                        fov_data.pop(key)
                # convert to a ROIData object and add store it in _analysis_data
                _pv_analysis_data[well][roi] = ROIData(**fov_data)
    return _pv_analysis_data


datastore_path = "/Volumes/T7 Shield/neurons/SSADH_Fam005_CC240711_IG_NC_PlateC_240925_GCaMP6s/SSADH_Fam005_CC240711_IG_NC_PlateC_240925_GCaMP6s.tensorstore.zarr"
analysis_path = "/Volumes/T7 Shield/neurons/SSADH_Fam005_CC240711_IG_NC_PlateC_240925_GCaMP6s/SSADH_Fam005_CC240711_IG_NC_PlateC_240925_GCaMP6s_analysis"

data = TensorstoreZarrReader(datastore_path)

analysis_data = _load_and_set_data_from_json(Path(analysis_path))

# create a numpy array that contains the traces
# for well, fov_data in analysis_data.items():
#     for _, roi_data in fov_data.items():
#         roi_data = cast(ROIData, roi_data)
#         print(roi_data.dec_dff)


traces: list[float] = []
fov_data = analysis_data[analysis_data.keys()[0]]
for _, roi_data in fov_data.items():
    roi_data = cast(ROIData, roi_data)
    print(roi_data.dec_dff)
    traces.append(roi_data.dec_dff)

traces_array = np.array(traces)
print(traces_array.shape)
