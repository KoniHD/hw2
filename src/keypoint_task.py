import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class KeypointDetection(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 4e-3,
        criterion: str = "mse",
        optimizer: str = "adam",
        pretrained_backbone: bool = False,  # only used for tracking hparams
        use_scheduler: bool = False,  # enable ReduceLROnPlateau
        patience: int = 5,  # only used for tracking hparams
        activation: str = "relu",  # only used for tracking hparams
        dropout: float = 0.3,  # only used for tracking hparams
        batch_norm: bool = True,  # only used for tracking hparams
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        if criterion == "mse":
            self.criterion = nn.MSELoss()
        elif criterion == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.SmoothL1Loss()
        self.lr = lr
        self.example_input_array = torch.zeros(1, 1, 224, 224)

        if hasattr(self.model, "backbone"):
            self.backbone = self.model.backbone

    def forward(self, input):
        return self.model(input)

    def _shared_step(self, batch, batch_idx: int, stage: str):
        inputs = batch["image"]
        outputs = self(inputs)
        if outputs.ndim == 2:  # Direct Coordinate Regression
            targets = batch["keypoints"]
            loss = self.criterion(outputs, targets.view(targets.size(0), -1))
        elif outputs.ndim == 4:  # 2D Heatmap Regression
            targets = batch["heatmaps"]
            loss = self.criterion(outputs, targets)
        else:
            raise ValueError("No valid output shape")
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True,
        )

        if (
            batch_idx == 0
            and self.current_epoch % 5 == 0
            and (stage == "val" or stage == "test")
        ):
            self._log_predicted_images(inputs, outputs, targets)

        return loss

    def _log_predicted_images(self, images, outputs, targets, n=4):
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        for i in range(n):
            img = images[i, 0].cpu().numpy()  # (H, W)
            out = outputs[i].detach().cpu().numpy().reshape(68, 2)
            tar = targets[i].cpu().numpy().reshape(68, 2)
            h, w = img.shape

            ax = axes[i]
            ax.imshow(img, cmap="gray")
            ax.scatter(
                out[:, 0] * (w / 2) + (w / 2),
                out[:, 1] * (h / 2) + (h / 2),
                c="r",
                s=10,
                label="pred",
            )
            ax.scatter(
                tar[:, 0] * (w / 2) + (w / 2),
                tar[:, 1] * (h / 2) + (h / 2),
                c="g",
                s=10,
                label="gt",
            )
            ax.axis("off")

        axes[0].legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_arr = np.asarray(buf)[..., :3]  # drop alpha channel → (H, W, 3)
        plt.close(fig)

        # TensorBoard expects (N, H, W, C) or (H, W, C)
        self.logger.experiment.add_image(
            "val/predictions",
            img_arr,
            global_step=self.current_epoch,
            dataformats="HWC",
        )

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, input):
        return self(input)

    def configure_optimizers(self):
        # Detect backbone/head dynamically from model structure.
        # ResNet exposes both; UNet and SimpleCNN do not — avoids AttributeError.
        if hasattr(self.model, "backbone") and hasattr(self.model, "head"):
            parameters = self.model.head.parameters()
        else:
            parameters = self.parameters()

        self.optimizer = (
            optim.Adam(parameters, lr=self.lr)
            if self.hparams.optimizer == "adam"
            else optim.SGD(parameters, lr=self.lr)
        )

        if not self.hparams.use_scheduler:
            return self.optimizer

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,  # halve LR on plateau
            patience=3,  # wait 3 epochs with no improvement
            threshold=1e-3,  # matches EarlyStopping min_delta
            threshold_mode="abs",  # absolute, not relative — consistent with EarlyStopping
            min_lr=1e-6,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
