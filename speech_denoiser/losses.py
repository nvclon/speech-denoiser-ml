from __future__ import annotations

import torch


def si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    Args:
        pred: predicted signal (batch_size, length) or (batch_size, 1, length)
        target: target signal (batch_size, length) or (batch_size, 1, length)
        eps: small value for numerical stability

    Returns:
        SI-SDR values in dB, shape (batch_size,)
    """
    if pred.ndim == 3:
        pred = pred.squeeze(1)
    if target.ndim == 3:
        target = target.squeeze(1)

    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(pred * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target * target, dim=-1, keepdim=True).clamp_min(eps)

    s_target = dot * target / target_energy
    e_noise = pred - s_target

    num = torch.sum(s_target**2, dim=-1)
    den = torch.sum(e_noise**2, dim=-1).clamp_min(eps)

    return 10 * torch.log10(num / den)


def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SI-SDR loss (negative SI-SDR)."""
    return -si_sdr(pred, target).mean()


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 (MAE) loss."""
    return torch.nn.functional.l1_loss(pred, target)


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 (MSE) loss."""
    return torch.nn.functional.mse_loss(pred, target)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "si_sdr_l1",
    alpha: float = 0.1,
) -> torch.Tensor:
    """Combined loss function.

    Args:
        pred: predicted signal
        target: target signal
        loss_type: type of loss ("si_sdr_l1" or "si_sdr_l2")
        alpha: weight for the auxiliary loss component

    Returns:
        combined loss value
    """
    si_sdr_component = si_sdr_loss(pred, target)

    if loss_type == "si_sdr_l1":
        aux_component = l1_loss(pred, target)
    elif loss_type == "si_sdr_l2":
        aux_component = l2_loss(pred, target)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}")

    return si_sdr_component + alpha * aux_component
