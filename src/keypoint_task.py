import lightning as L
import torch
import torch.optim as optim


class KeypointDetection(L.LightningModule):
    def __init__(self, model, criterion, lr=4e-3):
        super().__init__()
        self.model = torch.compile(model)
        self.save_hyperparameters(ignore=["out_dim", "model"])

    def training_step(self, batch):
        inputs, targets = batch["image"], batch["keypoints"]
        outputs = self.model(inputs)
        loss = self.hparams.criterion(outputs, targets.view(targets.size(0), -1))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch["image"], batch["keypoints"]
        outputs = self.model(inputs)
        loss = self.hparams.criterion(outputs, targets.view(targets.size(0), -1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch):
        inputs, targets = batch["image"], batch["keypoints"]
        outputs = self.model(inputs)
        loss = self.hparams.criterion(outputs, targets.view(targets.size(0), -1))
        self.log("test_loss", loss, logger=True)
        return loss

    def predict_step(self, input):
        return self.model(input)

    def forward(self, input):
        self.model.eval()
        with torch.no_grad():
            return self.model(input)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.hparams.lr)
