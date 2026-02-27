import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


class KeypointDetection(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 4e-3,
        criterion: str = "mse",
        optimizer: str = "adam",
        pretrained_backbone: bool = False,
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

        if pretrained_backbone:
            self.backbone = self.model.backbone

    def forward(self, input):
        return self.model(input)

    def _shared_step(self, batch, stage: str):
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
        return loss

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch):
        return self._shared_step(batch, "val")

    def test_step(self, batch):
        return self._shared_step(batch, "test")

    def predict_step(self, input):
        return self(input)

    def configure_optimizers(self):
        if self.hparams.pretrained_backbone:
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
