import json
import argparse
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from model.cnn import CNN
from loss import CircleLoss
from util import CircleParams, plot_losses, plot_mean_iou, load_config, thresholded_iou
from img_dataset import ImageData, ImageDataset

from typing import List, Tuple

torch.manual_seed(42)


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--config_file", required=True)
    return parser.parse_args()


def load_datasets(dataset_json: str) -> Tuple[Dataset, float, float]:
    """
    Returns a dataset consisting of the images specified in the JSON file at 'dataset_json', the image size and the max. radius found in the images
    """
    with open(dataset_json, "r") as f:
        parameters = json.load(f)

    converter = ToTensor()
    dataset_list: List[ImageData] = []
    max_radius = 0

    dataset_dir = os.path.dirname(dataset_json)

    for img_name, params in parameters.items():
        img_path = f"{dataset_dir}/{img_name}"
        img = Image.open(img_path).convert('L')
        arr = np.asarray(img).copy()
        params = CircleParams(**params)
        if params.radius > max_radius:
            max_radius = params.radius

        dataset_list.append(ImageData(converter(arr), torch.Tensor(
            [params.row, params.col, params.radius])))

    dataset = ImageDataset(dataset_list)
    return dataset, dataset_list[0].img.size(1), max_radius


@torch.no_grad()
def eval_model(model: CNN, loss_fct: nn.modules.loss._Loss, test_loader: DataLoader, device: torch.device) -> Tuple[float]:
    model.eval()
    total_loss = 0
    total_acc = 0
    total_mean_iou = 0
    num_batches = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss: torch.Tensor = loss_fct(out, labels)
        total_loss += loss.item()
        accuracy, mean_iou = thresholded_iou(
            out.cpu(), labels.cpu(), threshold=0.7)
        total_acc += accuracy
        total_mean_iou += mean_iou
        num_batches += 1
    return total_loss / len(test_loader), total_acc / num_batches, total_mean_iou / num_batches


def train(train_dataset_json: str, test_dataset_json: str, config_json: str, args: argparse.Namespace):
    train_dataset, img_size, max_radius = load_datasets(train_dataset_json)
    test_dataset, _, _ = load_datasets(test_dataset_json)
    config = load_config(config_json)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] GPU not available, using CPU instead.")

    batch_size = config["batch_size"]
    lr = config["lr"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    kernel_size = config["kernel_size"]
    cnn = CNN(in_channels=1, kernel_size=kernel_size,
              img_size=img_size, radius_scale=max_radius)
    cnn.to(device)

    model_parameters = filter(lambda p: p.requires_grad, cnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"[INFO] Created CNN Model with {params} trainable parameters!")

    if args.checkpoint:
        cnn.load_state_dict(torch.load(args.checkpoint))
        print("[INFO] Loaded Model from checkpoint")

    criterion = CircleLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40], gamma=0.1)

    num_epochs = config["num_epochs"]

    train_loss_arr = []
    val_loss_arr = []
    train_iou = []
    val_iou = []

    save_every = config["save_every"]

    for i in range(num_epochs):
        loss_epoch = 0
        acc_epoch = 0
        nr_batches = 0
        cnn.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            out = cnn(images)
            loss = criterion(out, labels)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy, mean_iou = thresholded_iou(
                out.cpu(), labels.cpu(), threshold=0.7)
            acc_epoch += accuracy
            nr_batches += 1

        avg_loss = loss_epoch / len(train_loader)
        avg_acc = acc_epoch / nr_batches
        val_loss, val_acc, val_mean_iou = eval_model(
            cnn, criterion, test_loader, device)

        train_loss_arr.append(avg_loss)
        val_loss_arr.append(val_loss)
        train_iou.append(mean_iou)
        val_iou.append(val_mean_iou)

        print(f"[{i+1}/{num_epochs}]\tTrain Loss: {avg_loss}\tTrain Acc: ({round(avg_acc, 2)}, {round(mean_iou, 2)}) \tVal Loss: {val_loss}\tVal Acc: ({round(val_acc, 2)}, {round(val_mean_iou, 2)})")

        scheduler.step()

        if (i + 1) % save_every == 0:
            torch.save(cnn.state_dict(), f"model/checkpoint_{i+1}_weights.pth")

    plot_losses(train_loss_arr, val_loss_arr, save_path="plots/loss.pdf")
    plot_mean_iou(train_iou, val_iou, save_path="plots/iou.pdf")
    torch.save(cnn.state_dict(), "model/final_weights.pth")


if __name__ == "__main__":
    args = parse_cmd_line()
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        raise Warning(
            f"Specified checkpoint at {args.checkpoint} does not exist!")

    train_dataset_json = f"{args.data_path}/train/dataset.json"
    test_dataset_json = f"{args.data_path}/test/dataset.json"
    config_json = args.config_file

    train(train_dataset_json, test_dataset_json, config_json, args)
