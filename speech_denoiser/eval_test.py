from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig

from speech_denoiser.lightning_module import DenoiserLightningModule
from speech_denoiser.losses import si_sdr
from speech_denoiser.utils import (
    dvc_pull_with_bootstrap,
    ensure_dir,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_mono(audio_2d: np.ndarray) -> torch.Tensor:
    # audio_2d: [T, C]
    wav = torch.from_numpy(np.asarray(audio_2d)).transpose(0, 1)  # [C, T]
    if wav.numel() == 0:
        return torch.zeros((1, 0), dtype=torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
    audio_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return _to_mono(audio_np), int(sr)


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)


def _maybe_pull_test_data(cfg: DictConfig) -> None:
    repo_root = _repo_root()
    data_dir = repo_root / str(cfg.dataset.data_dir)
    test_dir = data_dir / "test"
    if test_dir.exists():
        return

    store_url = getattr(getattr(cfg, "dvc", None), "store_url", None)
    store_dir_cfg = getattr(getattr(cfg, "dvc", None), "store_dir", "../dvcstore")
    remote_name = getattr(getattr(cfg, "dvc", None), "remote_name", "local_data")
    store_dir = (repo_root / str(store_dir_cfg)).resolve()
    dvc_pull_with_bootstrap(
        repo_root,
        store_url=store_url,
        store_dir=store_dir,
        remote_name=str(remote_name),
    )


def eval_test(cfg: DictConfig) -> None:
    repo_root = _repo_root()
    _maybe_pull_test_data(cfg)

    ckpt_path = getattr(cfg.eval, "ckpt_path", None)
    if ckpt_path is None:
        raise ValueError(
            "eval.ckpt_path is required (e.g. eval.ckpt_path=artifacts/checkpoints/latest.ckpt)"
        )

    ckpt_path = repo_root / str(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    noisy_dir = repo_root / str(cfg.dataset.test_noisy_path)
    clean_dir = repo_root / str(cfg.dataset.test_clean_path)
    if not noisy_dir.exists() or not clean_dir.exists():
        raise FileNotFoundError(
            f"Test folders not found. Expected {noisy_dir} and {clean_dir}. "
            "Restore with `poetry run speech-denoiser dvc_pull`."
        )

    save_wavs = bool(getattr(cfg.eval, "save_wavs", False))
    output_dir = Path(
        str(getattr(cfg.eval, "output_dir", repo_root / "artifacts/predictions/test"))
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    metrics_csv = Path(str(getattr(cfg.eval, "metrics_csv", repo_root / "plots/test_metrics.csv")))
    if not metrics_csv.is_absolute():
        metrics_csv = repo_root / metrics_csv

    ensure_dir(metrics_csv.parent)
    if save_wavs:
        ensure_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = DenoiserLightningModule.load_from_checkpoint(str(ckpt_path))
    module.eval()
    module.to(device)

    sample_rate = int(cfg.audio.sample_rate)

    rows: list[dict[str, float | str]] = []
    noisy_sisdrs: list[float] = []
    denoised_sisdrs: list[float] = []

    noisy_files = sorted([p for p in noisy_dir.glob("*.wav") if p.is_file()])
    if not noisy_files:
        raise FileNotFoundError(f"No wav files found in: {noisy_dir}")

    for noisy_path in noisy_files:
        clean_path = clean_dir / noisy_path.name
        if not clean_path.exists():
            print(f"[WARN] Missing clean pair for {noisy_path.name}, skipping")
            continue

        noisy, noisy_sr = _load_wav(noisy_path)
        clean, clean_sr = _load_wav(clean_path)

        noisy = _resample_if_needed(noisy, noisy_sr, sample_rate)
        clean = _resample_if_needed(clean, clean_sr, sample_rate)

        # Align lengths
        T = int(min(noisy.shape[-1], clean.shape[-1]))
        noisy = noisy[..., :T]
        clean = clean[..., :T]

        if T == 0:
            print(f"[WARN] Empty audio for {noisy_path.name}, skipping")
            continue

        with torch.no_grad():
            noisy_b = noisy.unsqueeze(0).to(device)  # [1, 1, T]
            pred_b = module(noisy_b)  # [1, 1, T']
            pred = pred_b.squeeze(0).to("cpu")

        # Align pred to clean length if needed
        T2 = int(min(pred.shape[-1], clean.shape[-1]))
        pred = pred[..., :T2]
        clean2 = clean[..., :T2]
        noisy2 = noisy[..., :T2]

        noisy_si = float(si_sdr(noisy2, clean2).mean().item())
        denoised_si = float(si_sdr(pred, clean2).mean().item())
        imp = denoised_si - noisy_si

        noisy_sisdrs.append(noisy_si)
        denoised_sisdrs.append(denoised_si)

        rows.append(
            {
                "file": noisy_path.name,
                "seconds": float(T2) / float(sample_rate),
                "noisy_si_sdr": noisy_si,
                "denoised_si_sdr": denoised_si,
                "improvement_db": imp,
            }
        )

        if save_wavs:
            out_path = output_dir / f"denoised_{noisy_path.stem}.wav"
            pred_np = pred.squeeze(0).numpy().astype("float32")
            sf.write(str(out_path), pred_np, samplerate=sample_rate)

    if not rows:
        raise RuntimeError("No test pairs were evaluated (missing clean files?)")

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "seconds", "noisy_si_sdr", "denoised_si_sdr", "improvement_db"]
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_noisy = float(np.mean(noisy_sisdrs))
    mean_denoised = float(np.mean(denoised_sisdrs))
    print(f"[OK] Evaluated {len(rows)} files")
    print(f"[OK] Mean SI-SDR noisy:    {mean_noisy:.3f} dB")
    print(f"[OK] Mean SI-SDR denoised: {mean_denoised:.3f} dB")
    print(f"[OK] Mean improvement:     {(mean_denoised - mean_noisy):.3f} dB")
    print(f"[OK] Wrote per-file metrics to: {metrics_csv}")
