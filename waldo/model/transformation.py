import torch

from PIL import Image
from torchvision import transforms


class Transformation(torch.nn.Module):
    def __init__(self, device: str | torch.device | None = None):
        super().__init__()

        self.device = device

    def forward(self, x: Image) -> torch.Tensor:
        size = (224, 224)
        x = x.convert('RGB')

        transformation = transforms.Compose([
            transforms.Resize(
                size,
                antialias=True
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                hue=0.2,
                saturation=0.2
            ),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomInvert(p=0.5),
            transforms.RandomAdjustSharpness(
                p=0.5,
                sharpness_factor=1.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transformation(x)
