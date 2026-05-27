import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Back to back instances of 3x3 convolution layers,
    batch normalization, and ReLU activation
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Custom implementation of U-Net architecture for
    high-res semantic segmentation.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list = [64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contraction
        in_ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Expansion
        self.up_transposes = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.up_transposes.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f))

        # Pixel-level Output Map
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        skips = []

        # save the encoded maps for context during the expansion half
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

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

        return self.final_conv(x)
