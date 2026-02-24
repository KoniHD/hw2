import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def visualize_batch(model: nn.Module, batch: dict) -> None:
    device = model.device
    images, keypoints = batch["image"].to(device), batch["keypoints"].to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model.forward(images)

    outputs = outputs.view(-1, 68, 2).cpu()
    images_cpu = images.cpu()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        _, h, w = images_cpu[i].shape
        ax.imshow(images_cpu[i].numpy().transpose(1, 2, 0), cmap="gray")
        ax.scatter(
            outputs[i, :, 0] * (w / 2) + (w / 2),
            outputs[i, :, 1] * (h / 2) + (h / 2),
            c="r",
            s=10,
        )
        ax.scatter(
            keypoints[i, :, 0].cpu() * (w / 2) + (w / 2),
            keypoints[i, :, 1].cpu() * (h / 2) + (h / 2),
            c="g",
            s=10,
        )
        ax.axis("off")
    plt.suptitle("Red=Predicted, Green=Ground Truth")
    plt.show()


def visualize_loss_curve(logs: str, title: str) -> None:
    metrics = pd.read_csv(logs)

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics[["epoch", "train_loss"]].dropna().plot(x="epoch", ax=ax, label="Train Loss")
    metrics[["epoch", "val_loss"]].dropna().plot(x="epoch", ax=ax, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    plt.tight_layout()
    plt.show()
