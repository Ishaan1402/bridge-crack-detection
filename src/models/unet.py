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
      - ``upsample_mode``:    ``"conv_transpose"`` (default, checkpoint-
                              compatible) or ``"interpolate"`` (bilinear
                              upsample + 1x1 conv, avoids checkerboard
                              artifacts on thin cracks)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        se: bool = False,
        dropout: float = 0.0,
        deep_supervision: bool = False,
        upsample_mode: str = "conv_transpose",
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]
        if upsample_mode not in ("conv_transpose", "interpolate"):
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")
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
        self.upsample_mode = upsample_mode
        self.decoder = nn.ModuleList()
        if upsample_mode == "conv_transpose":
            self.up_transposes = nn.ModuleList()
            for f in reversed(features):
                self.up_transposes.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
                self.decoder.append(DoubleConv(f * 2, f, se=se))
        else:
            self.up_convs = nn.ModuleList()
            for f in reversed(features):
                self.up_convs.append(nn.Conv2d(f * 2, f, kernel_size=1))
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

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Kaiming init for convs, unit-gain BN — helps from-scratch training."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

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
        n_ups = len(self.up_transposes) if self.upsample_mode == "conv_transpose" else len(self.up_convs)
        for idx in range(n_ups):
            # 'upscale'
            if self.upsample_mode == "conv_transpose":
                x = self.up_transposes[idx](x)
            else:
                x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = self.up_convs[idx](x)
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
