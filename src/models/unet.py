import torch
import torch.nn as nn
from typing import List, Optional


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Squeezes each channel to a scalar via global average pooling, learns
    per-channel importance with two 1x1 convolutions, and reweights the input.
    Typically worth +0.5-2 Dice on small crack datasets at negligible cost.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class DoubleConv(nn.Module):
    """
    Back to back instances of 3x3 convolution layers,
    batch normalization, and ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, se: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SELayer(out_channels) if se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.se(x) if self.se is not None else x


class UNet(nn.Module):
    """
    Custom implementation of U-Net architecture for high-res semantic
    segmentation.

    Optional upgrades (all default OFF so existing checkpoints load unchanged):
      - ``se``:               squeeze-and-excitation after every conv block
      - ``dropout``:          Dropout2d on the bottleneck (regularization)
      - ``deep_supervision``: auxiliary 1x1 heads on decoder stages, used
                              only during training via ``forward_deep``
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        se: bool = False,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contraction
        in_ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(in_ch, f, se=se))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, se=se)
        self.bottleneck_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Expansion
        self.up_transposes = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.up_transposes.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f, se=se))

        # Pixel-level Output Map
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Deep supervision heads: one per decoder stage except the last
        if deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv2d(f, out_channels, kernel_size=1) for f in list(reversed(features))[:-1]
            ])
        else:
            self.deep_heads = nn.ModuleList()

    def _forward(self, x: torch.Tensor) -> tuple:
        skips = []

        # save the encoded maps for context during the expansion half
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.bottleneck_dropout(x)
        skips = skips[::-1]

        aux = []
        for idx in range(len(self.up_transposes)):
            # 'upscale'
            x = self.up_transposes[idx](x)
            skip_connection = skips[idx]

            # if upscaled image doesn't match dimesions with
            # skip connection, then make it match
            if x.shape[2:] != skip_connection.shape[2:]:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            # concatenate high-res encoder map (skip connection) and convolute
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)

            if idx < len(self.deep_heads):
                aux.append(self.deep_heads[idx](x))

        logits = self.final_conv(x)
        if self.deep_heads:
            aux = [
                nn.functional.interpolate(a, size=logits.shape[2:], mode="bilinear", align_corners=False)
                for a in aux
            ]
        return logits, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main prediction head (used for inference and eval)."""
        return self._forward(x)[0]

    def forward_deep(self, x: torch.Tensor) -> tuple:
        """(logits, aux_logits) for deep-supervision training."""
        return self._forward(x)
