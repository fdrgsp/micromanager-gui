import os

import matplotlib.pyplot as plt
import torch
from cellpose import io, models

img_folder = r"/Users/annayang26/Desktop/Cellpose_train_test/test_set"
files = io.get_image_files(img_folder, "_cp_masks")
images = [io.imread(f) for f in files]

model_path = (
    r"/Users/annayang26/Desktop/Cellpose_train_test/model/models/cp_img8_epoch7000_py"
)
model = models.CellposeModel(
    gpu=False, pretrained_model=model_path, device=torch.device("cpu")
)
print(f"Use GPU: {model.gpu}, device: {model.device}")

diameter = model.net.diam_labels.item()
flow_threshold = 0.4
cellprob_threshold = 0

# run model on test images
masks, flows, styles = model.eval(
    images,
    channels=[0, 0],
    diameter=diameter,
    flow_threshold=flow_threshold,
    cellprob_threshold=cellprob_threshold,
)
print(f"finished segmentation, masks: {masks[0].shape}")

io.save_masks(
    images,
    masks,
    flows,
    files,
    channels=[0, 0],
    png=True,  # save masks as ONGs and save example image
    tif=True,  # save masks as TIFFs
    save_txt=True,  # save txt outlines for ImageJ
    save_flows=False,  # save flows as TIFFs
    save_outlines=False,  # save outlines as TIFFs
    save_mpl=True,  # make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
)
f = files[0]
plt.figure(figsize=(12, 4), dpi=300)
plt.imshow(io.imread(os.path.splitext(f)[0] + "_cp_masks.png"))
plt.axis("off")
plt.show()
