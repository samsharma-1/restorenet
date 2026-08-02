"""Training entry point for the image restoration pipeline."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.datasets.dataloader import build_dataloaders
from src.losses import RestorationLoss
from src.metrics.restoration_metrics import RestorationMetrics
from src.models import build_model
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train an image restoration model (NAFNet baseline).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["lr"].to(device, non_blocking=True), batch["hr"].to(device, non_blocking=True)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_cfg = config.get("scheduler", {})
    if str(scheduler_cfg.get("name", "cosine")).lower() != "cosine":
        return None

    epochs = int(config.get("training", {}).get("epochs", 1))
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
    base_lr = float(config.get("training", {}).get("learning_rate", 2e-4))
    eta_min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=min_lr,
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=max(eta_min_ratio, 1e-3),
        total_iters=warmup_epochs,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: float,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
        "history": history,
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float]]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        float(checkpoint.get("best_metric", float("inf"))),
        list(checkpoint.get("history", [])),
    )


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: RestorationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0

    for batch in loader:
        lr, hr = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(lr)
            loss, parts = criterion(pred, hr)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # Bug Fix: Use .item() to extract Python floats and release autograd graph
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + value.item()
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader produced no batches; reduce batch_size or disable drop_last")
    return {f"train_{name}": value / batches for name, value in totals.items()}


def _run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: RestorationLoss,
    device: torch.device,
    use_amp: bool,
    metrics_calc: RestorationMetrics,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0

    with torch.no_grad():
        for batch in loader:
            lr, hr = _move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(lr)
                _, parts = criterion(pred, hr)

            img_metrics = metrics_calc(pred.float(), hr.float())

            # Bug Fix: Use .item() to extract scalar values safely
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value.item()
            for name, value in img_metrics.items():
                totals[name] = totals.get(name, 0.0) + value.item()
            batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"val_{name}": value / batches for name, value in totals.items()}


def train(config: dict[str, Any], resume_path: str | None = None) -> dict[str, Any]:
    """Run the training loop and save checkpoints."""
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config)
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = RestorationLoss(config.get("loss", {})).to(device)

    training_cfg = config.get("training", {})
    checkpoint_cfg = config.get("checkpoint", {})
    paths_cfg = config.get("paths", {})
    epochs = int(training_cfg.get("epochs", 1))
    learning_rate = float(training_cfg.get("learning_rate", 2e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    grad_clip_norm = training_cfg.get("grad_clip_norm", None)
    grad_clip = float(grad_clip_norm) if grad_clip_norm is not None else None
    use_amp = bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, config)
    
    # Bug Fix: Explicitly pass "cuda" string or disable scaler for CPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    log_dir = paths_cfg.get("log_dir", "outputs/logs")
    tb_writer = SummaryWriter(log_dir) if config.get("tensorboard", {}).get("enabled", True) else None

    monitor_metric = str(checkpoint_cfg.get("monitor_metric", "val_total"))
    monitor_mode = str(checkpoint_cfg.get("mode", "min")).lower()
    save_every = int(checkpoint_cfg.get("save_every_n_epochs", 5))
    save_best_only = bool(checkpoint_cfg.get("save_best_only", False))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 15))

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    history: list[dict[str, float]] = []

    metrics_calc = RestorationMetrics(device=device, compute_lpips=False)

    if resume_path is not None:
        start_epoch, best_metric, history = _load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, device
        )
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    logger.info(
        "Training %s on %s: train=%d val=%d params=%.2fM",
        model.__class__.__name__,
        device,
        len(train_loader.dataset),
        len(val_loader.dataset),
        sum(parameter.numel() for parameter in model.parameters()) / 1_000_000,
    )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip_norm=grad_clip,
        )
        val_metrics = _run_validation(model, val_loader, criterion, device, use_amp, metrics_calc)
        if scheduler is not None:
            scheduler.step()

        metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
        history.append(metrics)
        metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
        history.append(metrics)
        
        # --- ADD THIS: WandB Logging ---
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(metrics, step=epoch)
        # -------------------------------

        if tb_writer is not None:
            for k, v in train_metrics.items():
                tb_writer.add_scalar(k, v, epoch)
            for k, v in val_metrics.items():
                tb_writer.add_scalar(k, v, epoch)

        default_metric_val = float("inf") if monitor_mode == "min" else -float("inf")
        current = float(metrics.get(monitor_metric, default_metric_val))
        improved = current < best_metric if monitor_mode == "min" else current > best_metric
        
        if improved:
            best_metric = current
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                history,
                config,
            )
        else:
            epochs_without_improvement += 1

        should_save_epoch = save_every > 0 and epoch % save_every == 0
        if should_save_epoch and not save_best_only:
            _save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                history,
                config,
            )

        logger.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f best_%s=%.4f",
            epoch,
            epochs,
            metrics.get("train_total", 0.0),
            metrics.get("val_total", 0.0),
            monitor_metric,
            best_metric,
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if tb_writer is not None:
        tb_writer.close()

    _save_checkpoint(
        checkpoint_dir / "last_model.pth",
        model,
        optimizer,
        scheduler,
        epoch if 'epoch' in locals() else start_epoch,
        best_metric,
        history,
        config,
    )
    return {"history": history, "best_metric": best_metric, "checkpoint_dir": str(checkpoint_dir)}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()