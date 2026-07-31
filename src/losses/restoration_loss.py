"""Weighted restoration losses for denoising, deblurring, and SR."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class EdgeLoss(nn.Module):
    """L1 difference between Sobel edge magnitudes."""

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel_x = self.kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


class FFTLoss(nn.Module):
    """L1 difference between log FFT magnitudes."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        return F.l1_loss(pred_mag, target_mag)


class MultiScaleSSIMLoss(nn.Module):
    """Small dependency-free multi-scale SSIM loss."""

    def __init__(self, scales: int = 3) -> None:
        super().__init__()
        self.scales = scales

    @staticmethod
    def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x.square()
        sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y.square()
        sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
        return torch.clamp((numerator / denominator).mean(), 0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values = []
        x = pred
        y = target
        for _ in range(self.scales):
            values.append(self._ssim(x, y))
            if min(x.shape[-2:]) < 8:
                break
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
        return 1.0 - torch.stack(values).mean()


class LPIPSLoss(nn.Module):
    """Optional LPIPS wrapper, loaded only when configured with nonzero weight."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install lpips or set loss.weights.lpips to 0.0") from exc
        self.model = lpips.LPIPS(net="alex")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        return self.model(pred_scaled, target_scaled).mean()


class RestorationLoss(nn.Module):
    """Weighted combination of restoration losses from config."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        weights_cfg = (config or {}).get("weights", config or {})
        self.weights = {
            "l1": float(weights_cfg.get("l1", 1.0)),
            "ms_ssim": float(weights_cfg.get("ms_ssim", 0.0)),
            "lpips": float(weights_cfg.get("lpips", 0.0)),
            "edge": float(weights_cfg.get("edge", 0.0)),
            "fft": float(weights_cfg.get("fft", 0.0)),
        }
        self.ms_ssim = MultiScaleSSIMLoss()
        self.edge = EdgeLoss()
        self.fft = FFTLoss()
        self.lpips = LPIPSLoss() if self.weights["lpips"] > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape {pred.shape} must match target {target.shape}")

        parts: dict[str, torch.Tensor] = {}
        total = pred.new_tensor(0.0)

        if self.weights["l1"] > 0:
            parts["l1"] = F.l1_loss(pred, target)
            total = total + parts["l1"] * self.weights["l1"]
        if self.weights["ms_ssim"] > 0:
            parts["ms_ssim"] = self.ms_ssim(pred, target)
            total = total + parts["ms_ssim"] * self.weights["ms_ssim"]
        if self.weights["edge"] > 0:
            parts["edge"] = self.edge(pred, target)
            total = total + parts["edge"] * self.weights["edge"]
        if self.weights["fft"] > 0:
            parts["fft"] = self.fft(pred, target)
            total = total + parts["fft"] * self.weights["fft"]
        if self.weights["lpips"] > 0:
            if self.lpips is None:
                raise RuntimeError("LPIPS loss was not initialized")
            parts["lpips"] = self.lpips(pred, target)
            total = total + parts["lpips"] * self.weights["lpips"]

        parts["total"] = total
        return total, parts
