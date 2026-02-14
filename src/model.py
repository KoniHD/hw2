import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class Simple_CNN(nn.Module,
                 PyTorchModelHubMixin):
    
    def __init__():
        self.net = nn.Sequential(
            nn.Conv2D()
        )

    def forward(input):
        net(input)