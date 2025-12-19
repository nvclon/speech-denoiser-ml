from __future__ import annotations

import torch


def si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute SI-SDR for tensors shaped [B, T] or [B, 1, T]."""

    if pred.ndim == 3:
        pred = pred.squeeze(1)
    if target.ndim == 3:
        target = target.squeeze(1)

    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # projection of pred onto target
    dot = torch.sum(pred * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target * target, dim=-1, keepdim=True).clamp_min(eps)
    s_target = dot * target / target_energy

    e_noise = pred - s_target

    ratio = torch.sum(s_target * s_target, dim=-1).clamp_min(eps) / torch.sum(
        e_noise * e_noise, dim=-1
    ).clamp_min(eps)
    return 10.0 * torch.log10(ratio)


def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -si_sdr(pred, target).mean()
