"""
U-Net architecture for optic disc and cup segmentation.

A simple 3-level U-Net with skip connections suitable for medical image segmentation.
"""
import torch
import torch.nn as nn
from typing import Tuple


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with ReLU activation.

    This is the basic building block of the U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net model for binary segmentation (optic disc/cup).

    Architecture:
        - Encoder: 3 levels with max pooling (3→64→128→256)
        - Decoder: 2 upsampling levels with skip connections (256→128→64)
        - Output: Sigmoid activation for binary segmentation

    Input shape: (B, 3, H, W)
    Output shape: (B, 1, H, W) with values in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # Encoder (downsampling path)
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bridge (bottleneck)
        self.bridge = DoubleConv(128, 256)

        # Decoder (upsampling path)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 128)  # 256 because of skip connection

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(128, 64)  # 128 because of skip connection

        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Output tensor of shape (B, 1, H, W) with sigmoid activation
        """
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        # Bridge
        b = self.bridge(p2)

        # Decoder with skip connections
        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)  # Skip connection from encoder
        u2 = self.upconv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)  # Skip connection from encoder
        u1 = self.upconv1(u1)

        # Final output
        out = self.final(u1)
        return self.sigmoid(out)

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = self.count_parameters()
        return f"UNet model with {total_params:,} trainable parameters"


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for segmentation tasks.

    Dice coefficient measures overlap between predicted and target masks.
    Dice loss = 1 - Dice coefficient.

    Args:
        pred: Predicted mask (B, 1, H, W) with values in [0, 1]
        target: Target mask (B, 1, H, W) with values in [0, 1]
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Dice loss value (scalar tensor)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
