import random
from pathlib import Path

import tifffile

from micromanager_gui.readers import TensorstoreZarrReader
from ._plate_viewer import PlateViewer

path = r'/Volumes/T7 Shield/LAM77_NC240503_384_CBD_20240927/NC240503_240927_Treated_CBDanalogs.tensorstore.zarr'

img_path = Path(path).stem

save_path = '/Users/annayang26/Desktop/CellposeTraining/' + f'{img_path}.tif'

ts = TensorstoreZarrReader(path)

pos = random.randint(0, 10)
data = ts.isel(p=pos)

data_half_to_end = data[data.shape[0] // 2 :, :, :]

tifffile.imwrite(save_path, data_half_to_end, imagej=True)
