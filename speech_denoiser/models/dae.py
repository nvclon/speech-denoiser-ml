from __future__ import annotations

import torch
from torch import nn


class DAE(nn.Module):
    def __init__(self, num_layers: int = 4, kernel_size: int = 5, channels: int = 64) -> None:
        super().__init__()

        padding = kernel_size // 2
        encoder: list[nn.Module] = []
        in_ch = 1
        ch = channels
        for _ in range(num_layers):
            encoder.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=padding))
            encoder.append(nn.ReLU())
            in_ch = ch
            ch *= 2

        decoder: list[nn.Module] = []
        ch = in_ch
        for _ in range(num_layers - 1):
            decoder.append(nn.Conv1d(ch, ch // 2, kernel_size=kernel_size, padding=padding))
            decoder.append(nn.ReLU())
            ch = ch // 2
        decoder.append(nn.Conv1d(ch, 1, kernel_size=kernel_size, padding=padding))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        z = self.encoder(x)
        y = self.decoder(z)
        return y
