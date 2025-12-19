from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig

from speech_denoiser.export import export_onnx
from speech_denoiser.utils import ensure_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _triton_config_pbtxt(model_name: str, max_batch_size: int = 8) -> str:
    # ONNX model has input/output: [B, 1, T] with dynamic T.
    # Triton uses dims without the batch dimension.
    # dims: [channels=1, time=-1]
    return (
        f'name: "{model_name}"\n'
        f'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: {max_batch_size}\n"
        "input [\n"
        "  {\n"
        '    name: "noisy"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1, -1 ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "clean"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1, -1 ]\n"
        "  }\n"
        "]\n"
    )


def prepare_triton_repo(cfg: DictConfig) -> None:
    """Prepare a Triton model repository for ONNXRuntime backend.

    Steps:
    1) Export ONNX (uses export.ckpt_path).
    2) Create model repo folder: <repo>/denoiser_onnx/1/model.onnx
    3) Write config.pbtxt.
    """

    repo_root = _repo_root()

    # Export ONNX to artifacts/onnx/denoiser.onnx
    export_onnx(cfg)

    onnx_dir = Path(str(cfg.paths.onnx_dir))
    if not onnx_dir.is_absolute():
        onnx_dir = repo_root / onnx_dir
    onnx_path = onnx_dir / "denoiser.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX was not created at: {onnx_path}")

    model_repo_dir = Path(str(cfg.triton.model_repo_dir))
    if not model_repo_dir.is_absolute():
        model_repo_dir = repo_root / model_repo_dir

    model_name = str(cfg.triton.model_name)
    model_version = int(cfg.triton.model_version)

    model_dir = ensure_dir(model_repo_dir / model_name / str(model_version))
    dst_onnx = model_dir / "model.onnx"
    dst_onnx.write_bytes(onnx_path.read_bytes())

    config_path = model_repo_dir / model_name / "config.pbtxt"
    _write_text(config_path, _triton_config_pbtxt(model_name=model_name, max_batch_size=8))

    # Helpful metadata for debugging / reproducibility
    meta_path = model_repo_dir / model_name / "meta.json"
    meta = {
        "source_onnx": str(onnx_path),
        "model_name": model_name,
        "model_version": model_version,
        "sample_rate": int(cfg.audio.sample_rate),
        "input": {"name": "noisy", "dtype": "FP32", "shape": ["B", 1, "T"]},
        "output": {"name": "clean", "dtype": "FP32", "shape": ["B", 1, "T"]},
    }
    _write_text(meta_path, json.dumps(meta, indent=2))

    print(f"[OK] Triton model repo prepared at: {model_repo_dir}")
    print(f"[OK] Model: {model_name}, version: {model_version}")


def _to_mono(audio_2d: np.ndarray) -> torch.Tensor:
    wav = torch.from_numpy(np.asarray(audio_2d)).transpose(0, 1)  # [C, T]
    if wav.numel() == 0:
        return torch.zeros((1, 0), dtype=torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def triton_infer(cfg: DictConfig) -> None:
    """Run inference via Triton HTTP server.

    Requires: `poetry install -E triton`

    Inputs are waveforms shaped [B, 1, T] FP32.
    """

    try:
        import tritonclient.http as httpclient  # type: ignore
    except Exception as e:
        raise ImportError(
            "Triton client is not installed. Install it with: `poetry install -E triton`."
        ) from e

    repo_root = _repo_root()

    input_wav = getattr(cfg.triton, "input_wav", None)
    if input_wav is None:
        raise ValueError("triton.input_wav is required")

    input_path = Path(str(input_wav))
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    audio_np, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    wav = _to_mono(audio_np)

    sample_rate = int(cfg.audio.sample_rate)
    if int(sr) != sample_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=sample_rate)

    # Triton expects numpy input
    noisy = wav.unsqueeze(0).cpu().numpy().astype(np.float32)  # [1, 1, T]

    model_name = str(cfg.triton.model_name)
    url = str(cfg.triton.url)

    client = httpclient.InferenceServerClient(url=url, verbose=False)

    inputs = [httpclient.InferInput("noisy", noisy.shape, "FP32")]
    inputs[0].set_data_from_numpy(noisy)
    outputs = [httpclient.InferRequestedOutput("clean")]

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    pred = result.as_numpy("clean")  # [1, 1, T]

    out_dir = Path(str(cfg.triton.output_dir))
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    ensure_dir(out_dir)

    out_path = out_dir / f"triton_denoised_{input_path.stem}.wav"
    pred_mono = np.asarray(pred[0, 0]).astype("float32")
    sf.write(str(out_path), pred_mono, samplerate=sample_rate)

    print(f"[OK] Wrote: {out_path}")
