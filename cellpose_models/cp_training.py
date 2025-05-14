# import numpy as np
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cellpose import io, models, train


def train_cp_model(
    n_epoch: int, save_every: int, train_dir: str, test_dir: str, saving_path: str
):
    """Train Cellpose3 model."""
    io.logger_setup()

    output = io.load_train_test_data(
        train_dir,
        test_dir,
        image_filter="_img",
        mask_filter="_cp_masks",
        look_one_level_down=False,
    )

    images, labels, image_names, test_images, test_labels, image_names_test = output

    # images = [image[:, :, np.newaxis] for image in images]
    # print(f"shape of images: {images[0].shape}")

    # e.g. retrain a Cellpose model
    model = models.CellposeModel(
        model_type="cyto3", gpu=False, device=torch.device("cpu")
    )

    model_path, train_performance = train.train_seg(
        model.net,
        save_path=saving_path,
        train_data=images,
        train_labels=labels,
        channels=[0, 0],
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=1e-4,
        SGD=True,
        learning_rate=0.1,
        save_every=save_every,
        n_epochs=n_epoch,
    )

    return train_performance


def _save_performance(train_performance, save_path):
    with open(save_path, "w") as file:
        json.dump(train_performance, file, indent=4)


def plot_losses(n_epoch: int, plot_every: int, json_file_path: str, img_name: str):
    """Plot losses from json file."""
    train_losses: list[float] = []
    test_losses: list[float] = []
    epoch_num: list[int] = []
    lrs: list[float] = []

    with open(json_file_path) as file:
        data = json.load(file)
        for i in range(n_epoch):
            if i % plot_every == 0 and str(i) in data:
                train_msg = data[str(i)].split(",")
                train_loss = float(train_msg[0].split("=")[-1])
                test_loss = float(train_msg[1].split("=")[-1])
                lr = float(train_msg[2].split("=")[-1])

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                lrs.append(lr)
                epoch_num.append(i)

    fig, ax = plt.subplots()
    plt.plot(epoch_num, train_losses, "-b", label="train loss")
    plt.plot(epoch_num, test_losses, "-r", label="test loss")
    plt.xlabel("epoch")
    plt.legend(loc="upper right")

    save_path = Path(json_file_path).parent / f"{img_name}.png"
    plt.savefig(save_path)

    return train_losses, test_losses, epoch_num


def plot_losses_from_txt(txt_path: str, save_every: int, img_name: str):
    """Plot losses from txt file."""
    train_losses: list[float] = []
    test_losses: list[float] = []
    epoch_num: list[int] = []
    lrs: list[float] = []

    with open(txt_path) as file:
        for line in file:
            # line = file.readline()
            elements = line.split(" ")
            epoch = int(elements[3].split(",")[0])
            if epoch > 0 and epoch % save_every == 0:
                epoch_num.append(epoch)
                train_loss_phrase = elements[4]
                train_loss = float(train_loss_phrase.split("=")[1][:-1])
                train_losses.append(train_loss)
                # print(f"train loss: {train_loss}")

                test_loss_phrase = elements[5]
                test_loss = float(test_loss_phrase.split("=")[1][:-1])
                test_losses.append(test_loss)
                # print(f"test loss: {test_loss}")

                lr_phrase = elements[6]
                lr = float(lr_phrase.split("=")[1][:-1])
                lrs.append(lr)
            # print(f"lr: {lr}")

    # print(f"length of epoch: {len(epoch_num)}")
    # print(f"length of test loss: {len(test_losses)}")

    fig, ax = plt.subplots()
    plt.plot(epoch_num, train_losses, "-b", label="train loss")
    plt.plot(epoch_num, test_losses, "-r", label="test loss")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")

    save_path = Path(txt_path).parent / f"{img_name}.png"
    print(save_path)
    plt.savefig(save_path)

    return train_losses, test_losses, epoch_num


if __name__ == "__main__":
    train_dir = r"/Users/annayang26/Desktop/Cellpose_train_test/training_round3"
    test_dir = r"/Users/annayang26/Desktop/Cellpose_train_test/test_set"
    saving_path = r"/Users/annayang26/Desktop/Cellpose_train_test/model"

    n_epoch = 10000
    save_every = 1000
    # train_performance = train_cp_model(n_epoch, save_every, train_dir, test_dir,
    # saving_path)

    # json_save_path = f"{saving_path}/Img10_Epoch_{n_epoch}_train_performance.json"
    # json_save_path = r'/Users/annayang26/Desktop/Cellpose_train_test/model/Epoch_10000_train_performance.json'  # noqa: E501
    # _save_performance(train_performance, json_save_path)
    img_name = "img8_training_performance"
    # img6_train_losses, img6_test_losses, img6_epoch = plot_losses(n_epoch, save_every,
    # json_save_path, img_name)
    txt_path = r"/Users/annayang26/Desktop/Cellpose_train_test/model/iimg8_train_performance.txt"  # noqa: E501
    img8_train_losses, img8_test_losses, img8_epoch = plot_losses_from_txt(
        txt_path, save_every, img_name
    )
