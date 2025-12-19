from __future__ import annotations

import pytorch_lightning as pl
import torch

from speech_denoiser.losses import si_sdr, si_sdr_loss
from speech_denoiser.models.dae import DAE
from speech_denoiser.models.demucs_wrapper import DemucsWrapper


class DenoiserLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        loss_function: str,
        optimizer: str,
        dae_num_layers: int = 4,
        dae_kernel_size: int = 5,
        demucs_channels: int = 32,
        demucs_depth: int = 4,
        demucs_kernel_size: int = 8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_name_lower = model_name.lower()
        if model_name_lower.startswith("dae"):
            self.model = DAE(num_layers=dae_num_layers, kernel_size=dae_kernel_size)
        elif model_name_lower.startswith("demucs"):
            demucs_channels = int(getattr(self.hparams, "demucs_channels", 32))
            demucs_depth = int(getattr(self.hparams, "demucs_depth", 4))
            demucs_kernel_size = int(getattr(self.hparams, "demucs_kernel_size", 8))
            self.model = DemucsWrapper(
                channels=demucs_channels,
                depth=demucs_depth,
                kernel_size=demucs_kernel_size,
            )
        else:
            raise ValueError(f"Unsupported model_name={model_name!r}")

        if loss_function != "si_sdr_loss":
            raise ValueError(f"Unsupported loss_function={loss_function!r}")

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        return self.model(noisy)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return si_sdr_loss(pred, target)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        noisy, clean = batch
        pred = self(noisy)
        loss = self._loss(pred, clean)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        noisy, clean = batch
        pred = self(noisy)
        loss = self._loss(pred, clean)
        metric = si_sdr(pred, clean).mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/si_sdr", metric, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = float(self.hparams.learning_rate)
        if self.hparams.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        raise ValueError(f"Unsupported optimizer={self.hparams.optimizer!r}")
