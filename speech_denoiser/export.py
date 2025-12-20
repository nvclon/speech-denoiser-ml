from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from speech_denoiser.lightning_module import DenoiserLightningModule
from speech_denoiser.utils import ensure_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_name(name: str) -> str:
    name = name.strip() or "model"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def export_onnx(cfg: DictConfig) -> Path:
    repo_root = _repo_root()

    ckpt_path = cfg.export.ckpt_path
    if ckpt_path is None:
        raise ValueError("export.ckpt_path is required")

    ckpt_path = Path(str(ckpt_path))
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path

    module = DenoiserLightningModule.load_from_checkpoint(str(ckpt_path))
    module.eval()

    onnx_dir = Path(str(cfg.paths.onnx_dir))
    if not onnx_dir.is_absolute():
        onnx_dir = repo_root / onnx_dir
    ensure_dir(onnx_dir)

    model_name = str(getattr(module.hparams, "model_name", ""))
    safe_model_name = _safe_name(model_name)

    # Dummy input length matters for some architectures.
    sample_rate = int(cfg.audio.sample_rate)
    segment_seconds = float(getattr(cfg.audio, "segment_length", 1.0))
    dummy_seconds = max(1.0, segment_seconds)
    dummy_len = int(round(dummy_seconds * sample_rate))
    dummy = torch.zeros(1, 1, dummy_len, dtype=torch.float32)

    out_model_dir = ensure_dir(onnx_dir / safe_model_name)
    out_path = out_model_dir / "denoiser.onnx"

    torch.onnx.export(
        module,
        dummy,
        str(out_path),
        input_names=["noisy"],
        output_names=["clean"],
        dynamic_axes={"noisy": {2: "time"}, "clean": {2: "time"}},
        opset_version=int(cfg.export.opset),
    )

    return out_path
