from collections import OrderedDict

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class Simple_CNN(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    pipeline_tag="keypoint-detection",
    repo_url="https://github.com/KoniHD/hw2.git",
):
    def __init__(
        self,
        out_dim: int = 136,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ) -> None:
        """
        Args:
            out_dim: Defines the number of key points the are being detected. Must be divisible by 2. out_dim = 2 * num_keypoints
            activation: Modular way to swap activation function.
            dropout: Dropout probability. Must be in [0, 1).
        """
        assert out_dim % 2 == 0, "out_dim must be divisible by 2"
        assert activation in ACTIVATION_MAP, (
            f"activation must be one of {list(ACTIVATION_MAP.keys())}"
        )
        assert 0.0 <= dropout < 1.0, "drouput must be in [0, 1)"

        super().__init__()

        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        act = ACTIVATION_MAP[activation]

        self.conv_block1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(1, 8, kernel_size=5, padding=1),
                    ),  # [B, 8, 222, 222]
                    ("bn", nn.BatchNorm2d(8) if self.batch_norm else nn.Identity()),
                    ("act", act()),
                    ("pool", nn.MaxPool2d(2, stride=2)),  # [B, 8, 111, 111]
                ]
            )
        )

        self.conv_block2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(8, 16, kernel_size=4, padding=1),
                    ),  # [B, 16, 110, 110]
                    ("bn", nn.BatchNorm2d(16) if self.batch_norm else nn.Identity()),
                    ("act", act()),
                    ("pool", nn.MaxPool2d(2, stride=2)),  # [B, 16, 55, 55]
                ]
            )
        )

        self.conv_block3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(16, 16, kernel_size=4, padding=1),
                    ),  # [B, 16, 54, 54]
                    ("bn", nn.BatchNorm2d(16) if self.batch_norm else nn.Identity()),
                    ("act", act()),
                    ("pool", nn.MaxPool2d(3, stride=3)),  # [B, 16, 18, 18]
                ]
            )
        )

        self.fc_head = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),  # [B, 16 * 18 * 18] = [B, 5184]
                    ("fc1", nn.Linear(16 * 18 * 18, 1024)),
                    ("bn1", nn.BatchNorm1d(1024) if self.batch_norm else nn.Identity()),
                    ("act1", act()),
                    ("dropout1", nn.Dropout(p=dropout)),
                    ("fc2", nn.Linear(1024, out_dim)),
                    ("tanh", nn.Tanh()),  # [B, out_dim]
                ]
            )
        )

    def forward(self, input):

        # Add dimensions to be able to run a single input
        if input.dim() == 3:
            input = input.unsqueeze(0)

        x = self.conv_block1(input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.fc_head(x)
