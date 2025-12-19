from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn


def _import_demucs_class():
    """Import Demucs model class from the `demucs` PyPI package.

    We intentionally depend on the external package instead of vendoring the upstream repo.
    """

    try:
        # Common in demucs package
        from demucs.demucs import Demucs  # type: ignore

        return Demucs
    except Exception:
        try:
            # Some versions may expose it differently
            from demucs.models.demucs import Demucs  # type: ignore

            return Demucs
        except Exception as e:
            raise ImportError(
                "Demucs is not installed. Install it with: `poetry install -E demucs` "
                "(or `poetry add demucs`)."
            ) from e


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class DemucsWrapper(nn.Module):
    """Thin wrapper around the external Demucs implementation.

    Goal: use Demucs as a waveform-to-waveform denoiser in our training loop.

    Input:  [B, 1, T] or [B, T]
    Output: [B, 1, T]
    """

    def __init__(
        self,
        channels: int = 32,
        depth: int = 4,
        kernel_size: int = 8,
    ) -> None:
        super().__init__()

        Demucs = _import_demucs_class()

        # Many demucs versions are source-separation models and return [B, S, C, T].
        # We set S=1 and treat it as a denoiser.
        base_kwargs: dict[str, Any] = {
            "sources": ["clean"],
            "audio_channels": 1,
            "channels": channels,
            "depth": depth,
            "kernel_size": kernel_size,
        }

        # Some Demucs constructors may not accept all keys; filter them.
        kwargs = _filter_kwargs(Demucs, base_kwargs)

        try:
            self.model = Demucs(**kwargs)
        except TypeError:
            # Fallback: sources could be positional in some APIs.
            sources = base_kwargs["sources"]
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "sources"}
            self.model = Demucs(sources, **fallback_kwargs)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)

        out = self.model(noisy)

        # demucs commonly returns [B, S, C, T]
        if out.ndim == 4:
            out = out[:, 0]  # [B, C, T]

        # [B, T] -> [B, 1, T]
        if out.ndim == 2:
            out = out.unsqueeze(1)

        # If multi-channel, convert to mono.
        if out.ndim == 3 and out.shape[1] > 1:
            out = out.mean(dim=1, keepdim=True)

        return out
