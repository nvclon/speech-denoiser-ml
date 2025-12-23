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


def _triton_config_pbtxt(
    model_name: str, platform: str = "onnxruntime_onnx", max_batch_size: int = 0
) -> str:
    """Generate Triton config.pbtxt for ONNX or TensorRT backend.

    Args:
        model_name: Triton model name
        platform: "onnxruntime_onnx" for ONNX, "tensorrt_plan" for TensorRT
        max_batch_size: max batch size (0 = dynamic batching disabled)
    """
    return (
        f'name: "{model_name}"\n'
        f'platform: "{platform}"\n'
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


def prepare_triton_repo(cfg: DictConfig, backend: str | None = None) -> None:
    """Prepare Triton model repository.

    Creates structure:
        triton/<model_name>/<backend>/
            ├── 1/model.(onnx|plan)
            ├── config.pbtxt
            └── meta.json

    Args:
        cfg: Hydra config
        backend: "onnx" or "trt". If None, uses cfg.server.backend
    """
    if backend is None:
        backend = str(cfg.server.backend)
    repo_root = _repo_root()
    model_name = str(cfg.model.model_name)
    model_version = int(cfg.server.model_version)

    if backend == "onnx":
        # ONNX backend
        artifacts_onnx = repo_root / "artifacts" / "onnx" / model_name / "denoiser.onnx"
        onnx_path = artifacts_onnx if artifacts_onnx.exists() else export_onnx(cfg)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX export failed: {onnx_path}")

        triton_dir = repo_root / "triton" / model_name / "onnx"
        version_dir = ensure_dir(triton_dir / str(model_version))

        # Copy ONNX file
        dst_onnx = version_dir / "model.onnx"
        dst_onnx.write_bytes(onnx_path.read_bytes())

        # Copy external data if exists
        onnx_data_path = onnx_path.with_name(onnx_path.name + ".data")
        if onnx_data_path.exists():
            (version_dir / onnx_data_path.name).write_bytes(onnx_data_path.read_bytes())

        platform = "onnxruntime_onnx"
        backend_name = "onnxruntime"
        source_file = str(onnx_path)

    elif backend == "trt":
        # TensorRT backend
        trt_path = repo_root / "artifacts" / "trt" / model_name / "denoiser.plan"
        if not trt_path.exists():
            raise FileNotFoundError(
                f"TensorRT file not found: {trt_path}\n"
                f"Run: bash scripts/build_trt_engine.sh artifacts/onnx/{model_name}/denoiser.onnx"
            )

        triton_dir = repo_root / "triton" / model_name / "trt"
        version_dir = ensure_dir(triton_dir / str(model_version))

        # Copy TensorRT file
        dst_plan = version_dir / "model.plan"
        dst_plan.write_bytes(trt_path.read_bytes())

        platform = "tensorrt_plan"
        backend_name = "tensorrt"
        source_file = str(trt_path)

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'onnx' or 'trt'.")

    # Write config.pbtxt
    config_path = triton_dir / "config.pbtxt"
    _write_text(config_path, _triton_config_pbtxt(model_name, platform=platform))

    # Write metadata
    meta_path = triton_dir / "meta.json"
    meta = {
        "backend": backend_name,
        "model_name": model_name,
        "model_version": model_version,
        "source_file": source_file,
        "sample_rate": int(cfg.audio.sample_rate),
        "input": {"name": "noisy", "dtype": "FP32", "shape": ["B", 1, "T"]},
        "output": {"name": "clean", "dtype": "FP32", "shape": ["B", 1, "T"]},
    }
    _write_text(meta_path, json.dumps(meta, indent=2))

    print(f"[OK] Triton {backend.upper()} repo: {triton_dir}")
    print(f"[OK] Model: {model_name} v{model_version}")


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

    input_wav = getattr(cfg.server, "input_wav", None)
    if input_wav is None:
        raise ValueError("server.input_wav is required")

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

    model_name = str(cfg.server.model_name)
    url = str(cfg.server.url)

    client = httpclient.InferenceServerClient(url=url, verbose=False)

    inputs = [httpclient.InferInput("noisy", noisy.shape, "FP32")]
    inputs[0].set_data_from_numpy(noisy)
    outputs = [httpclient.InferRequestedOutput("clean")]

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    pred = result.as_numpy("clean")  # [1, 1, T]

    out_dir = Path(str(cfg.server.output_dir))
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    ensure_dir(out_dir)

    safe_model_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)
    out_path = out_dir / f"triton_{safe_model_name}_denoised_{input_path.stem}.wav"
    pred_mono = np.asarray(pred[0, 0]).astype("float32")
    sf.write(str(out_path), pred_mono, samplerate=sample_rate)

    print(f"[OK] Wrote: {out_path}")
