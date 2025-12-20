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


def _triton_config_pbtxt(model_name: str, max_batch_size: int = 0) -> str:
    # By default we disable Triton batching (max_batch_size=0).
    # Our current ONNX export uses a fixed batch dimension of 1.
    # Therefore Triton must see the full tensor shape including that dimension.
    return (
        f'name: "{model_name}"\n'
        f'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: {max_batch_size}\n"
        "input [\n"
        "  {\n"
        '    name: "noisy"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1, 1, -1 ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "clean"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1, 1, -1 ]\n"
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

    # Export ONNX (path may be model-specific, e.g. artifacts/onnx/<model>/denoiser.onnx)
    onnx_path = export_onnx(cfg)
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

    # Torch ONNX export may use the "external data" format and store weights in a
    # sidecar file named like `<onnx_filename>.data` (e.g. `denoiser.onnx.data`).
    # ONNXRuntime will look for that exact file name at runtime.
    onnx_data_path = onnx_path.with_name(onnx_path.name + ".data")
    if onnx_data_path.exists():
        (model_dir / onnx_data_path.name).write_bytes(onnx_data_path.read_bytes())

    config_path = model_repo_dir / model_name / "config.pbtxt"
    _write_text(config_path, _triton_config_pbtxt(model_name=model_name, max_batch_size=0))

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

    safe_model_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)
    out_path = out_dir / f"triton_{safe_model_name}_denoised_{input_path.stem}.wav"
    pred_mono = np.asarray(pred[0, 0]).astype("float32")
    sf.write(str(out_path), pred_mono, samplerate=sample_rate)

    print(f"[OK] Wrote: {out_path}")
