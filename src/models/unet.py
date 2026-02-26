import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


class Encoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "relu"
    ) -> None:
        assert activation in ACTIVATION_MAP, (
            f"activation must be one of {list(ACTIVATION_MAP.keys())}"
        )

        super().__init__()
        self.act = ACTIVATION_MAP[activation]

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.act(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.act(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, input) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(input)
        return self.pool(skip), skip


class Decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "relu"
    ) -> None:
        assert activation in ACTIVATION_MAP, (
            f"activation must be one of {list(ACTIVATION_MAP.keys())}"
        )
        super().__init__()
        self.act = ACTIVATION_MAP[activation]

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.block = nn.Sequential(
            # input is cat([upsample, skip]) → 2 * out_channels channels
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.act(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.act(),
        )

    def forward(self, input, skip) -> torch.Tensor:
        up = self.upsample(input)
        up = F.interpolate(up, size=skip.shape[2:], mode="nearest-exact")
        x = torch.cat([up, skip], dim=1)
        return self.block(x)


class UNetKeypointDetector(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    pipeline_tag="keypoint-detection",
    repo_url="https://github.com/KoniHD/hw2.git",
):
    def __init__(
        self,
        num_predictions: int = 68,
        activation: str = "relu",
        base_features: int = 64,
        output_size: int = 64,
    ) -> None:
        assert num_predictions % 2 == 0, "num_predictions has to be divisible by 2"
        assert activation in ACTIVATION_MAP, (
            f"activation must be one of {list(ACTIVATION_MAP.keys())}"
        )
        super().__init__()
        self.output_size = output_size

        f = base_features
        act = ACTIVATION_MAP[activation]

        self.enc1 = Encoder(1, f, activation)
        self.enc2 = Encoder(f, f * 2, activation)
        self.enc3 = Encoder(f * 2, f * 4, activation)
        self.enc4 = Encoder(f * 4, f * 8, activation)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 16),
            act(),
            nn.Conv2d(f * 16, f * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 16),
            act(),
        )

        self.dec4 = Decoder(f * 16, f * 8, activation)
        self.dec3 = Decoder(f * 8, f * 4, activation)
        self.dec2 = Decoder(f * 4, f * 2, activation)
        self.dec1 = Decoder(f * 2, f, activation)

        self.head = nn.Conv2d(f, num_predictions, kernel_size=1)

    def forward(self, input) -> torch.Tensor:
        # [B, 1, 224, 224]
        x, skip1 = self.enc1(
            input
        )  # x: [B, f,    112, 112]  skip1: [B, f,    224, 224]
        x, skip2 = self.enc2(x)  # x: [B, f*2,   56,  56]  skip2: [B, f*2,  112, 112]
        x, skip3 = self.enc3(x)  # x: [B, f*4,   28,  28]  skip3: [B, f*4,   56,  56]
        x, skip4 = self.enc4(x)  # x: [B, f*8,   14,  14]  skip4: [B, f*8,   28,  28]
        x = self.bottleneck(x)  # x: [B, f*16,  14,  14]
        x = self.dec4(x, skip4)  # x: [B, f*8,   28,  28]
        x = self.dec3(x, skip3)  # x: [B, f*4,   56,  56]
        x = self.dec2(x, skip2)  # x: [B, f*2,  112, 112]
        x = self.dec1(x, skip1)  # x: [B, f,    224, 224]
        x = self.head(x)  # x: [B, num_predictions, 224, 224]
        return F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="nearest-exact",
        )  # [B, num_predictions, output_size, output_size]
