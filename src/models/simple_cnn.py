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
        self, out_dim: int = 136, activation: str = "relu", dropout: float = 0.0
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

        act = ACTIVATION_MAP[activation]

        self.net = nn.Sequential(
            # Input: [B, 1, 224, 224]
            # Block 1
            nn.Conv2d(1, 8, kernel_size=5, padding=1),  # [B, 8, 222, 222]
            nn.BatchNorm2d(8),
            act(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(2, stride=2),  # [B, 8, 111, 111]
            # Block 2
            nn.Conv2d(8, 16, kernel_size=4, padding=1),  # [B, 16, 110, 110]
            nn.BatchNorm2d(16),
            act(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(2, stride=2),  # [B, 16, 55, 55]
            # Block 3
            nn.Conv2d(16, 16, kernel_size=4, padding=1),  # [B, 16, 54, 54]
            nn.BatchNorm2d(16),
            act(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(3, stride=3),  # [B, 16, 18, 18]
            nn.Flatten(),  # [B, 16 * 18 * 18] = [B, 5184]
            # FC Layers with Dropout
            nn.Linear(16 * 18 * 18, 1024),  # [B, 1024]
            nn.BatchNorm1d(1024),
            act(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, out_dim),  # [B, 136]
            nn.Tanh(),
            # Output: [B, 136] -> reshaped to [B, 68, 2] by caller if needed
        )

    def forward(self, input):

        # Add dimensions to be able to run a single input
        if input.dim() == 3:
            input = input.unsqueeze(0)

        return self.net.forward(input)
