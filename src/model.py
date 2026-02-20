import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class Simple_CNN(nn.Module,
                 PyTorchModelHubMixin,
                 library_name="pytorch",
                 pipeline_tag=["keypoint-detection"],
                 repo_url="https://github.com/KoniHD/hw2.git"):
    
    def __init__(self, out_dim: int = 136, hparams=None, activation: nn.Module = nn.ReLU) -> None:
        r"""
        out_dim (`int`, *optional*, defaults to 136):
            Defines the number of key points the are being detected. Must be divisible by 2. out_dim = 2 * num_keypoints
        hparams (`hparams`, *optional*, defautls to None):
            List of possible hyperparameters to be given to the network.
        activation (`nn.Module`, *optional*, defaults to nn.ReLU):
            Modular way to swap activation function.
        """
        assert(out_dim % 2 == 0)
        super().__init__()
        self.hparams = hparams

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=1),     # 4*4*1*8 + 8 = 136
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=4, padding=1),     # 3*3*8*16 + 16 = 1168
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, kernel_size=4, padding=1),    # 3*3*16*16 + 16 = 2320
            activation(),
            nn.MaxPool2d(3, stride=3),
            nn.Flatten(),
            nn.Linear(16 * 18 * 18, 1024),                  # 16*18*18*1024 = 5_308_416
            activation(),
            nn.Linear(1024, out_dim),                        # 1024 * 136 = 139_264
            nn.Tanh()
        )

    def forward(self, input):

        # Add dimensions to be able to run a single input
        if input.dim() == 3:
            input = input.unsqueeze(0)

        return self.net.forward(input)