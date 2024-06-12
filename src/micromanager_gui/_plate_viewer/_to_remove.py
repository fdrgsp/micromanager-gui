# in case of old files that did not have the metadata in the tensorstore, we can use
# the following code to add the metadata to the tensorstore.
# to use it you need to have:
# - the path to the tensorstore to which we want to add metadata
# - the path to the MDASequence that we want to add to the tensorstore. NOTE: the
#   MDASequence should contain the HCS metadata.
# - the number of timepoints of the timelapse (if not known, we can get it from the
#   MDASequence).
# By running this code we add the metadata to the tensorstore. After that, if we open
# the tensorstore with the reader we should be able to get data and metadata through the
# isel method (and this use the file in the plate_viewer).

import json

import tensorstore as ts
import useq

# ____________________________VARIABLES________________________________________________
TIMEPOINTS = 350  # number of timepoints of the timelapse (we can also get this from the
# MDASequence if not known)

# the path to the tensorstore to which we want to add metadata
TENSORSTORE_PATH = (
    r"/Volumes/T7 Shield/NC240509_240523_Chronic/nc240509_240523_chronic.zarr"
)

# the path to the MDASequence that we want to add to the tensorstore
MDA_PATH = r"/Volumes/T7 Shield/NC240509_240523_Chronic/mda_updated.json"
# ______________________________________________________________________________________

# open the MDASequence
with open(MDA_PATH) as f:
    mda = json.load(f)
mda = useq.MDASequence(**mda)

# generate Frame metadatas (for the plate_viewer we only need an 'Event' key with the
# index and pos_name)
frame_metadatas: list[dict] = []
for idx, pos in enumerate(mda.stage_positions):
    name = pos.name
    frame_metadatas.extend(
        {"Event": {"index": {"p": idx, "t": t, "c": 0}, "pos_name": name}}
        for t in range(TIMEPOINTS)
    )
# create the metadata that we want to add to the tensorstore
metadata = {
    "useq_MDASequence": mda.model_dump_json(exclude_defaults=True),
    "frame_metadatas": frame_metadatas,
}

# open the tensorstore and write the metadata
spec = {
    "driver": "zarr",
    "kvstore": {"driver": "file", "path": TENSORSTORE_PATH},
}
store = ts.open(spec)
store.kvstore.write(".zattrs", json.dumps(metadata))
