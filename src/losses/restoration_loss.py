import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, Tuple

class MultiScaleSSIMLoss(nn.Module):
    """L1-based Multi-Scale SSIM approximation."""
    def __init__(self, scales: int = 3):
        super().__init__()
        self.scales = scales

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1, c2 = 0.01**2, 0.03**2
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        sigma1sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1**2
        sigma2sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2**2
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1sq + sigma2sq + c2))
        return ssim_map.mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for _ in range(self.scales):
            loss += (1.0 - self._ssim(x, y))
            if x.shape[-1] < 8: break
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        return loss / self.scales

class RestorationLoss(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        w = config.get("weights", config)
        self.weights = {
            "l1": float(w.get("l1", 1.0)),
            "ms_ssim": float(w.get("ms_ssim", 0.0)),
            "edge": float(w.get("edge", 0.0)),
            "fft": float(w.get("fft", 0.0))
        }
        self.ms_ssim_fn = MultiScaleSSIMLoss() if self.weights["ms_ssim"] > 0 else None
        # ... (other functions: edge_fn, fft_fn)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts = {}
        total = 0.0
        if self.weights["l1"] > 0:
            parts["l1"] = F.l1_loss(pred, target)
            total += parts["l1"] * self.weights["l1"]
        if self.ms_ssim_fn:
            parts["ms_ssim"] = self.ms_ssim_fn(pred, target)
            total += parts["ms_ssim"] * self.weights["ms_ssim"]
        # ... (apply edge and fft similarly)
        parts["total"] = total
        return total, parts