"""Metrics for image restoration (PSNR, SSIM, LPIPS)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class RestorationMetrics(nn.Module):
    """Computes standard image restoration metrics: PSNR, SSIM, and optionally LPIPS."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        compute_lpips: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        
        self.compute_lpips = compute_lpips
        self.lpips_model = None
        if compute_lpips:
            if not LPIPS_AVAILABLE:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("LPIPS requested but not installed. Skipping LPIPS.")
                self.compute_lpips = False
            else:
                self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
                self.lpips_model.eval()
                for param in self.lpips_model.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute metrics for a batch.

        Args:
            pred: Predicted images (N, C, H, W) in [0, 1].
            target: Ground truth images (N, C, H, W) in [0, 1].

        Returns:
            Dictionary containing 'psnr', 'ssim', and optionally 'lpips'.
        """
        # Ensure tensors are on the right device and cast to float32 for metric computation
        pred = pred.to(self.device, dtype=torch.float32)
        target = target.to(self.device, dtype=torch.float32)
        
        metrics = {
            "psnr": self.psnr(pred, target),
            "ssim": self.ssim(pred, target),
        }
        
        if self.compute_lpips and self.lpips_model is not None:
            # LPIPS expects inputs in [-1, 1]
            pred_scaled = pred * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            # Returns shape [N, 1, 1, 1], we take the mean over the batch
            lpips_val = self.lpips_model(pred_scaled, target_scaled).mean()
            metrics["lpips"] = lpips_val

        return metrics
