import random
from pathlib import Path

import tifffile

# from _plate_viewer._plate_viewer import PlateViewer
from micromanager_gui.readers import TensorstoreZarrReader

path = r"/Volumes/T7 Shield/LAM77_NC240503_384_CBD_20240927/NC240503_240927_Treated_CBDanalogs.tensorstore.zarr"

img_path = Path(path).stem

ts = TensorstoreZarrReader(path)

total_pos = int(ts.isel().shape[0])
print(total_pos)

pos = random.randint(0, total_pos - 1)
data = ts.isel(p=pos)

data_half_to_end = data[data.shape[0] // 2 :, :, :]
cyto_frame = data_half_to_end.max(axis=0)

train_save_path = (
    "/Users/annayang26/Desktop/CellposeTraining/" + f"{img_path}_{pos}.tif"
)
tifffile.imwrite(train_save_path, cyto_frame, imagej=True)

img_save_path = (
    "/Users/annayang26/Desktop/CellposeTraining/image/" + f"{img_path}_{pos}.tif"
)
tifffile.imwrite(img_save_path, data_half_to_end, imagej=True)
