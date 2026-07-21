import torch
import torch.nn as nn

import config


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT):
        super().__init__()

        # Encoder (6 levels: 64 -> 128 -> 256 -> 512 -> 512 -> bottleneck 1024)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128,  dropout=dropout * 0.25)
        self.enc3 = DoubleConv(128, 256, dropout=dropout * 0.5)
        self.enc4 = DoubleConv(256, 512, dropout=dropout)
        self.enc5 = DoubleConv(512, 512, dropout=dropout)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout=dropout)

        # Decoder (mirrors encoder)
        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = DoubleConv(1024, 512, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout=dropout * 0.5)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout=dropout * 0.25)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        b = self.bottleneck(self.pool(e5))

        d5 = self.dec5(torch.cat([self.up5(b),  e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


class MultiBinaryUNet(nn.Module):
    """Shared encoder-decoder backbone with 3 independent sigmoid binary heads."""
    def __init__(self, in_channels=1, num_binary=3, dropout=config.DROPOUT):
        super().__init__()
        # Reuse full UNet but stop before final classification
        self.backbone = UNet(in_channels=in_channels, num_classes=64, dropout=dropout)
        self.backbone.out_conv = nn.Identity()
        self.heads = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(num_binary)])

    def forward(self, x):
        features = self.backbone(x)                          # (B, 64, H, W)
        return torch.cat([h(features) for h in self.heads], dim=1)  # (B, 3, H, W)


def new_model():
    if config.MODE == "multibinary":
        return MultiBinaryUNet().to(config.DEVICE)
    return UNet().to(config.DEVICE)
