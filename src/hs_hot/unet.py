"""A U-Net on a torchvision ResNet-50 encoder (needs the `train` extra; import explicitly).

The encoder is meant to be initialised from a torchvision ImageNet checkpoint (`resnet50-*.pth`; see `export.build_unet`);
the decoder is new. The model takes an RGB image in [0, 1] of any height and width and returns one road logit per pixel
at that size: ImageNet normalisation happens inside, so training and serving cannot disagree about it.
"""

import torch
import torch.nn.functional as functional
import torchvision
from torch import nn

from .params import IMAGE_MEAN, IMAGE_STD

ENCODER_CHANNELS = (
    64,
    256,
    512,
    1024,
    2048,
)  # stem (stride 2), layer1 (4), layer2 (8), layer3 (16), layer4 (32)
DECODER_CHANNELS = (256, 128, 64, 32, 16)


class ConvBlock(nn.Sequential):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Upsample to the size of the skip features (or to `size`), join them, and convolve."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(
        self, features: torch.Tensor, skip: torch.Tensor | None = None, size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        target = skip.shape[-2:] if skip is not None else size
        features = functional.interpolate(features, size=target, mode="bilinear", align_corners=False)
        if skip is not None:
            features = torch.cat([features, skip], dim=1)
        return self.convolutions(features)


class ResNetEncoder(nn.Module):
    """The ResNet-50 stem and four stages, returning the features of each for the decoder's skip connections."""

    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        stem = self.relu(self.bn1(self.conv1(image)))
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return [stem, layer1, layer2, layer3, layer4]


class HighwayUNet(nn.Module):
    """RGB in [0, 1], `(B, 3, H, W)` -> road logits `(B, 1, H, W)`; any `H` and `W`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1), persistent=False)
        self.encoder = ResNetEncoder()
        deepest, *shallower = reversed(ENCODER_CHANNELS)
        in_channels = [deepest, *DECODER_CHANNELS[:-1]]
        skip_channels = [*shallower, 0]
        self.up_blocks = nn.ModuleList(
            UpBlock(inp, skip, out) for inp, skip, out in zip(in_channels, skip_channels, DECODER_CHANNELS)
        )
        self.head = nn.Conv2d(DECODER_CHANNELS[-1], 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        *skips, features = self.encoder((image - self.mean) / self.std)
        for up_block, skip in zip(self.up_blocks[:-1], reversed(skips)):
            features = up_block(features, skip)
        features = self.up_blocks[-1](features, size=image.shape[-2:])
        return self.head(features)
