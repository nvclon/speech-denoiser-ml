from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from speech_denoiser.eval_test import eval_test
from speech_denoiser.export import export_onnx
from speech_denoiser.infer import infer
from speech_denoiser.train import train
from speech_denoiser.triton import prepare_triton_repo, triton_infer
from speech_denoiser.utils import try_gdown_folder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compose_cfg(overrides: list[str] | None = None) -> DictConfig:
    config_dir = _repo_root() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


class _CLI:
    def train(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        train(cfg)

    def infer(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        infer(cfg)

    def export_onnx(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        export_onnx(cfg)

    def eval_test(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        eval_test(cfg)

    def prepare_triton_repo(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        prepare_triton_repo(cfg)

    def triton_infer(self, *overrides: str) -> None:
        cfg = _compose_cfg(list(overrides))
        triton_infer(cfg)

    def download_data(self, url: str | None = None, output_dir: str | None = None) -> None:
        cfg = _compose_cfg([])
        repo_root = _repo_root()

        resolved_url = url or getattr(cfg.data, "gdrive_folder_url", None)
        if not resolved_url:
            raise ValueError("Google Drive folder URL is not set")

        data_dir = Path(output_dir) if output_dir else repo_root / str(cfg.data.data_dir)
        if not data_dir.is_absolute():
            data_dir = repo_root / data_dir

        try_gdown_folder(str(resolved_url), output_dir=data_dir)

    def setup_cuda(
        self,
        torch_version: str = "2.6.0+cu124",
        cuda_index_url: str = "https://download.pytorch.org/whl/cu124",
    ) -> None:
        """Install CUDA-enabled torch/torchaudio into the active Poetry env.

        Poetry may be unable to resolve from download.pytorch.org in some networks (HTTP 403).
        This command uses pip directly (inside the Poetry venv), which often works.
        """

        if sys.platform.startswith("win") and sys.version_info >= (3, 13):
            print(
                "[WARN] Windows + Python 3.13 often has no CUDA PyTorch wheels yet. "
                "If CUDA stays unavailable, create a Python 3.12 environment and reinstall."
            )

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            cuda_index_url,
            "--extra-index-url",
            "https://pypi.org/simple",
            f"torch=={torch_version}",
            f"torchaudio=={torch_version}",
        ]

        print("[INFO] Installing CUDA torch via pip:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(
            "[INFO] Done. Verify with: `poetry run python -c "
            '"import torch; print(torch.__version__, torch.cuda.is_available())"`'
        )

    def print_config(self, *overrides: str) -> dict[str, Any]:
        cfg = _compose_cfg(list(overrides))
        # Fire pretty-prints dicts nicely
        return {
            "cfg": cfg,
        }


def main() -> None:
    fire.Fire(_CLI)
