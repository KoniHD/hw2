import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


class KeypointDetection(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 4e-3, criterion: str = "mse"):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.criterion = nn.MSELoss() if criterion == "mse" else nn.SmoothL1Loss()

    def forward(self, input):
        self.model.eval()
        with torch.no_grad():
            return self(input)

    def _shared_step(self, batch, stage: str):
        inputs, targets = batch["image"], batch["keypoints"]
        outputs = self(inputs)
        loss = self.criterion(outputs, targets.view(targets.size(0), -1))
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
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
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
