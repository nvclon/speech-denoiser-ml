from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig

from speech_denoiser.lightning_module import DenoiserLightningModule
from speech_denoiser.utils import ensure_dir, try_dvc_pull, try_gdown_folder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_pull_test_data(cfg: DictConfig) -> None:
    repo_root = _repo_root()
    data_dir = repo_root / str(cfg.data.data_dir)
    test_dir = data_dir / "test"
    if test_dir.exists():
        return

    gdrive_url = getattr(cfg.data, "gdrive_folder_url", None)
    if gdrive_url:
        try:
            print("[INFO] test data not found; downloading from Google Drive via gdown...")
            try_gdown_folder(str(gdrive_url), output_dir=data_dir)
        except Exception as e:
            print(f"[WARN] gdown download failed: {e}")

    if test_dir.exists():
        return

    try_dvc_pull(repo_root, targets=["data/test.dvc"])


def infer(cfg: DictConfig) -> None:
    repo_root = _repo_root()
    _maybe_pull_test_data(cfg)

    if cfg.infer.input_wav is None:
        raise ValueError("infer.input_wav is required (e.g. infer.input_wav=path/to/noisy.wav)")

    ckpt_path = cfg.infer.ckpt_path
    if ckpt_path is None:
        raise ValueError("ckpt_path is required (e.g. ckpt_path=artifacts/checkpoints/latest.ckpt)")

    ckpt_path = repo_root / str(ckpt_path)
    module = DenoiserLightningModule.load_from_checkpoint(str(ckpt_path))
    module.eval()

    input_wav = Path(str(cfg.infer.input_wav))
    if not input_wav.is_absolute():
        input_wav = repo_root / input_wav

    audio_np, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
    # [T, C] -> [C, T]
    wav = torch.from_numpy(np.asarray(audio_np)).transpose(0, 1)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    sample_rate = int(cfg.audio.sample_rate)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)

    with torch.no_grad():
        pred = module(wav.unsqueeze(0))  # [1, 1, T]
    pred = pred.squeeze(0)

    output_dir = Path(str(cfg.infer.output_dir))
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    ensure_dir(output_dir)

    out_path = output_dir / f"denoised_{input_wav.stem}.wav"
    # pred: [1, T] -> write as mono [T]
    pred_np = pred.squeeze(0).detach().cpu().numpy().astype("float32")
    sf.write(str(out_path), pred_np, samplerate=sample_rate)
