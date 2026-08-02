"""Compact NAFNet-style baseline for image restoration."""
import torch
from torch import nn
import torch.nn.functional as F  # <--- Add this line

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SimpleGate(nn.Module):
    """Split channels in half and multiply them elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class NAFBlock(nn.Module):
    """Nonlinear activation free residual block used by the baseline."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = residual + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """Small encoder-free NAFNet baseline with PixelShuffle upsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        num_blocks: int = 8,
        scale_factor: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        if width <= 0:
            raise ValueError("width must be > 0")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")

        self.scale_factor = scale_factor
        self.out_channels = out_channels
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[NAFBlock(width, dropout=dropout) for _ in range(num_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(
                width,
                out_channels * scale_factor * scale_factor,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale_factor),
        )
        self.skip = nn.Upsample(
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.skip_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(self.intro(x))
        restored = self.upsample(features)
        skip = self.skip_projection(self.skip(x))
        return torch.clamp(restored + skip, 0.0, 1.0)


class NAFNetUNet(nn.Module):
    """Encoder-decoder NAFNet for full image restoration."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        enc_blocks: list[int] = [1, 1, 1, 1],
        mid_blocks: int = 1,
        dec_blocks: list[int] = [1, 1, 1, 1],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, dropout=dropout) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2
            
        self.middle = nn.Sequential(*[NAFBlock(chan, dropout=dropout) for _ in range(mid_blocks)])
        
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for num in dec_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, dropout=dropout) for _ in range(num)])
            )
            
        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
        
        self.skip_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

# Inside NAFNetUNet class
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. Calculate required padding (multiple of 2^levels)
    h, w = x.shape[-2:]
    mod = 2**len(self.encoders)
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    
    # 2. Apply padding
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    # 3. Existing logic using x_padded
    features = self.intro(x_padded)
    # ... (encoder/middle/decoder blocks)
    restored = self.ending(features)
    
    # 4. Crop back to original size
    restored = restored[:, :, :h, :w]
    
    skip_x = self.skip_projection(x)
    return torch.clamp(restored + skip_x, 0.0, 1.0)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from the repository YAML config structure."""
    model_cfg = config.get("model", config)
    name = str(model_cfg.get("name", "NAFNet")).lower()
    
    if name == "nafnet":
        return NAFNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            width=int(model_cfg.get("width", 32)),
            num_blocks=int(model_cfg.get("num_blocks", 8)),
            scale_factor=int(model_cfg.get("scale_factor", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif name == "nafnetunet":
        enc_blocks = model_cfg.get("enc_blocks", [1, 1, 1, 1])
        dec_blocks = model_cfg.get("dec_blocks", [1, 1, 1, 1])
        return NAFNetUNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            width=int(model_cfg.get("width", 32)),
            enc_blocks=enc_blocks,
            mid_blocks=int(model_cfg.get("mid_blocks", 1)),
            dec_blocks=dec_blocks,
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported model: {model_cfg.get('name')}")
