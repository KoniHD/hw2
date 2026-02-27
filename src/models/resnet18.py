import torch.nn as nn
import torchvision.models as models
from huggingface_hub import PyTorchModelHubMixin


class ResNetKeypointDetector(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    pipeline_tag="keypoint-detection",
    repo_url="https://github.com/KoniHD/hw2.git",
):
    def __init__(
        self,
        out_dim: int = 136,
        grayScale: bool = True,
    ) -> None:
        assert out_dim % 2 == 0, "out_dim must be divisible by 2"
        super().__init__()

        backbone = models.resnet18()

        if grayScale:
            original_weight = backbone.conv1.weight.data
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            backbone.conv1.weight.data = original_weight.mean(dim=1, keepdim=True)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input):
        features = self.backbone(input)
        return self.head(features)
