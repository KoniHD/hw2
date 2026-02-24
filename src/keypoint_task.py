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
        patience: int = 5,
        optimizer: str = "adam",
        activation: str = "relu",  # only used for tracking hparams
        droput: float = 0.3,  # only used for tracking hparams
        batch_norm: bool = True,  # only used for tracking hparams
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.criterion = nn.MSELoss() if criterion == "mse" else nn.SmoothL1Loss()
        self.lr = lr
        self.patience = patience
        self.example_input_array = torch.zeros(1, 1, 224, 224)

    def forward(self, input):
        return self.model(input)

    def _shared_step(self, batch, stage: str):
        inputs, targets = batch["image"], batch["keypoints"]
        outputs = self(inputs)
        loss = self.criterion(outputs, targets.view(targets.size(0), -1))
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
        self.optimizer = (
            optim.Adam(self.parameters(), lr=self.lr)
            if self.hparams.optimizer == "adam"
            else optim.SGD(self.parameters(), lr=self.lr)
        )
        return self.optimizer
