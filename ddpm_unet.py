import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class LightweightUNet(nn.Module):
    def __init__(self, time_emb_dim=32, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.time_mlp = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim))

        self.conv_down1 = DoubleConv(1, 32)
        self.conv_down2 = DoubleConv(32, 64)
        self.conv_down3 = DoubleConv(64, 128)

        self.bottleneck = DoubleConv(128, 256)

        self.conv_up3 = DoubleConv(256 + 128, 128)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        self.conv_up1 = DoubleConv(64 + 32 + time_emb_dim, 32)

        self.conv_last = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t = t.float() / self.timesteps
        t_emb = self.time_mlp(t.unsqueeze(-1))

        # Downsampling
        conv1 = self.conv_down1(x)
        x = F.max_pool2d(conv1, 2)
        conv2 = self.conv_down2(x)
        x = F.max_pool2d(conv2, 2)
        conv3 = self.conv_down3(x)
        x = F.max_pool2d(conv3, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        x = F.interpolate(x, size=conv3.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)

        x = F.interpolate(x, size=conv2.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)

        x = F.interpolate(x, size=conv1.shape[2:], mode="bilinear", align_corners=True)
        t_emb_expanded = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, conv1, t_emb_expanded], dim=1)
        x = self.conv_up1(x)

        return self.conv_last(x)
