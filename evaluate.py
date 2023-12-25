import argparse
import os

import torch
from torch.utils.data import DataLoader

from model.cnn import CNN
from util import load_config, thresholded_iou
from train import load_datasets
from loss import CircleLoss


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-p", "--parameters", required=True)
    parser.add_argument("-i", "--images_path", required=True)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.parameters)
    dataset, img_size, max_radius = load_datasets(args.images_path)

    kernel_size = config["kernel_size"]
    batch_size = config["batch_size"]
    cnn = CNN(in_channels=1, kernel_size=kernel_size,
              img_size=img_size, radius_scale=max_radius)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    cnn.load_state_dict(torch.load(args.checkpoint))
    print("[INFO] Loaded Model from checkpoint")

    loss_fct = CircleLoss()

    cnn.eval()
    total_loss = 0
    total_acc_05 = 0
    total_acc_06 = 0
    total_acc_07 = 0
    total_mean_iou = 0
    num_batches = 0
    for images, labels in data_loader:
        out = cnn(images)
        loss: torch.Tensor = loss_fct(out, labels)
        total_loss += loss.item()
        accuracy_05, mean_iou = thresholded_iou(
            out, labels, threshold=0.5)
        accuracy_06, _ = thresholded_iou(
            out, labels, threshold=0.6)
        accuracy_07, _ = thresholded_iou(
            out, labels, threshold=0.7)
        total_acc_05 += accuracy_05
        total_acc_06 += accuracy_06
        total_acc_07 += accuracy_07
        total_mean_iou += mean_iou
        num_batches += 1

    print("=== Results ===")
    print(f"Loss:\t{total_loss / len(data_loader)}")
    print(
        f"Accuracy at Threshold:\t0.5: {total_acc_05 / num_batches}\t0.6: {total_acc_06 / num_batches}\t0.7: {total_acc_07 / num_batches}")
    print(f"Mean IoU:\t{total_mean_iou / num_batches}")


if __name__ == "__main__":
    args = parse_cmd_line()

    if not os.path.isfile(args.checkpoint):
        raise Warning(
            f"Specified checkpoint at {args.checkpoint} does not exist!")

    evaluate(args)
