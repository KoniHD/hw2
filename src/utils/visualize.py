import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def visualize_batch(model: nn.Module, batch: dict) -> None:
    device = next(model.parameters()).device
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


def visualize_heatmaps(model, batch):
    device = next(model.parameters()).device
    images, heatmaps = batch["image"].to(device), batch["heatmaps"].to(device)

    model.eval()
    with torch.inference_mode():
        outputs = model(images)

    # We will visualize the first image in the batch
    idx = 0
    img = images[idx].cpu().numpy().transpose(1, 2, 0)

    # Get heatmaps
    heatmaps_target = heatmaps[idx].cpu()
    pred_hm = outputs[idx].cpu()  # [68, 64, 64]
    c, h, w = pred_hm.shape

    # Extract argmax from predicted heatmaps
    pred_hm_flat = pred_hm.view(c, -1)
    argmax_idx = pred_hm_flat.argmax(dim=-1)

    # Convert 1D indices back to 2D (y, x)
    pred_y = argmax_idx // w
    pred_x = argmax_idx % w

    # Resize coordinates back to original image size (224x224)
    scale_y = img.shape[0] / h
    scale_x = img.shape[1] / w
    pred_y = pred_y.float() * scale_y
    pred_x = pred_x.float() * scale_x

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Image with Predicted Points
    ax1.imshow(img, cmap="gray")
    ax1.scatter(pred_x, pred_y, c="r", s=10, label="Predicted")
    ax1.set_title("Predicted Keypoints (Argmax)")
    ax1.axis("off")
    ax1.legend()

    # Visualize Heatmap overlay
    ax2.imshow(img, cmap="gray")
    # Overlay the max heatmap across all keypoints
    max_hm, _ = torch.max(pred_hm, dim=0)
    ax2.imshow(max_hm, cmap="jet", alpha=0.5, extent=[0, img.shape[1], img.shape[0], 0])
    ax2.set_title("Predicted Heatmaps Overlay")
    ax2.axis("off")

    ax3.imshow(img, cmap="gray")
    max_target_hm, _ = torch.max(heatmaps_target, dim=0)
    ax3.imshow(
        max_target_hm, cmap="jet", alpha=0.5, extent=[0, img.shape[1], img.shape[0], 0]
    )
    ax3.set_title("Ground Truth Heatmaps")
    ax3.axis("off")

    plt.show()
