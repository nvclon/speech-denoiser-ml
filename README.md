# Speech Denoiser

A neural network-based speech enhancement (denoising) system to improve audio quality by removing background noise from noisy speech signals.

**Author:** Ivan Vetoshkin

## Problem Statement

This project aims to train a speech denoising model that transforms noisy audio signals into clean, enhanced speech.

**Applications:**

- Online calls (VoIP, conferencing)
- Voice assistants
- Speech recognition (Speech-to-Text)
- Audio processing pipelines

**Input:** WAV audio file with background noise
**Output:** WAV audio file with enhanced/denoised speech

## Metrics

The primary quality metric is **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio), measured in dB.

**Target:** ~5–6 dB SI-SDR (in line with published results for Demucs architecture)

## Data

The dataset consists of paired (noisy, clean) WAV audio samples.

**Data Split:**

- Train: 28 speakers (~6Gb overall)
- Validation: 15% of train speakers
- Test: seperate dataset

**Noise types:** 8 real (DEMAND dataset) + 2 synthetic
**SNR levels:** Train: {0, 5, 10, 15} dB | Test: {2.5, 7.5, 12.5, 17.5} dB

**Storage:** Data is managed via [DVC](https://dvc.org) with local storage backend.

## Models

### DAE (Denoising Autoencoder) - Baseline

A lightweight convolutional autoencoder using 1D convolutions and transposed convolutions.

**Architecture:**

- 4 encoder layers + 4 decoder layers
- Kernel size: 5
- Parameter count: ~1.7M

### Demucs v3 (Tiny)

A simplified version of Facebook's Demucs model, optimized for faster training/inference.

**Configuration:**

- 32 channels, depth 4, kernel size 8
- Parameter count: ~1.6M

Both models use PyTorch Lightning for training and support multiple loss functions (SI-SDR, SI-SDR + L1, SI-SDR + L2).

## Quick Start

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd speech-denoiser-ml
   ```

2. **Create and activate a Python environment** (Python ≥3.10, <3.15)

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies with Poetry**

   ```bash
   pip install poetry
   poetry install --with dev
   ```

4. **Optional: Install model-specific extras**

   ```bash
   # For Demucs model support
   poetry install --with dev -E demucs

   # For Triton inference server support (Linux only)
   poetry install --with dev -E triton
   ```

5. **Install pre-commit hooks** (for code quality)

   ```bash
   pre-commit install
   ```

6. **Verify installation**
   ```bash
   poetry run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```

### Data Preparation

Data is automatically pulled from DVC storage when needed. To manually restore:

```bash
poetry run speech-denoiser dvc_pull
```

This will run `dvc pull` and, on a clean machine, download+extract `dvcstore.tar.gz` (from `dvc.store_url`) into `../dvcstore` and retry.

## Training

### Train the DAE baseline model

```bash
# Default config (demucs model, see configs/config.yaml)
poetry run speech-denoiser train

# Or explicitly use DAE
poetry run speech-denoiser train model=dae

# With custom hyperparameters
poetry run speech-denoiser train model=dae trainer.max_epochs=20 dataset.batch_size=32 model.learning_rate=0.0005
```

### Train Demucs v3 Tiny

```bash
# Use demucs config
poetry run speech-denoiser train model=demucs

# With SI-SDR + L1 loss (alpha=0.15)
poetry run speech-denoiser train model=demucs model.loss_function=si_sdr_l1 model.loss_alpha=0.15
```

### Training Output

Logs and artifacts are saved in:

- **Checkpoints:** `artifacts/checkpoints/` (latest checkpoint: `latest_<model_name>.ckpt`)
- **Metrics:** `plots/<model_name>/metrics.csv`
- **Plots:** `plots/<model_name>/{train_loss, val_loss, val_si_sdr}.png`
- **MLflow:** http://127.0.0.1:8080 (if server is running)

### Loss Functions

The training framework supports multiple loss combinations:

- `si_sdr_loss`: Pure SI-SDR loss (default)
- `si_sdr_l1`: SI-SDR + α×L1 (Mean Absolute Error)
- `si_sdr_l2`: SI-SDR + α×MSE

Adjust in config:

```yaml
loss_function: "si_sdr_l1"
loss_alpha: 0.1
```

## Inference

### Quick inference on a single audio file

```bash
# Hydra overrides
poetry run speech-denoiser infer model=demucs \
   infer.input_wav=/path/to/noisy_audio.wav

# Optionally override checkpoint/output
poetry run speech-denoiser infer model=demucs \
   infer.input_wav=/path/to/noisy_audio.wav \
   infer.ckpt_path=artifacts/checkpoints/latest_demucs_v3_tiny.ckpt \
   infer.output_dir=artifacts/predictions
```

**Output:** Enhanced audio saved to `artifacts/predictions/denoised_<input_stem>.wav`

### Evaluate on test set

```bash
poetry run speech-denoiser eval_test model=demucs

poetry run speech-denoiser eval_test model=demucs eval.save_wavs=true # Strongly not recommended!!
```

**Output:** Test metrics saved to `plots/<model_name>/test_metrics.csv`

## Tests

### Test-set evaluation (`eval_test`)

Reproduce:

```bash
poetry run speech-denoiser eval_test model=dae
poetry run speech-denoiser eval_test model=demucs
```

Results (run on 23 Dec 2025):

| Model          | Mean SI-SDR (noisy), dB | Mean SI-SDR (denoised), dB | Mean improvement, dB |
| -------------- | ----------------------: | -------------------------: | -------------------: |
| DAE_baseline   |                   8.444 |                     10.463 |                2.019 |
| demucs_v3_tiny |                   8.444 |                     18.460 |               10.016 |

Per-file CSV outputs:

- `plots/DAE_baseline/test_metrics.csv`
- `plots/demucs_v3_tiny/test_metrics.csv`

### Audio example (5 seconds)

Source: `data/test/noisy_testset_wav/p232_001.wav`

![Noisy](assets/readme/p232_001_noisy_5s.mp4)

![DAE_baseline denoised](assets/readme/p232_001_DAE_baseline_denoised_5s.mp4)

![demucs_v3_tiny denoised](assets/readme/p232_001_demucs_v3_tiny_denoised_5s.mp4)

## Production Deployment

### 1. Export Model to ONNX

ONNX format enables deployment to various backends (CPU/GPU inference engines).

```bash
# Export the latest checkpoint to ONNX
poetry run speech-denoiser export_onnx model=demucs export.ckpt_path=artifacts/checkpoints/latest_demucs_v3_tiny.ckpt

# Outputs: artifacts/onnx/demucs_v3_tiny/denoiser.onnx
```

### 2. (Optional) Convert to TensorRT

TensorRT optimization for faster GPU inference (NVIDIA GPUs only), this sadly doesn't work with Demucs model:

```bash
bash scripts/build_trt_engine.sh artifacts/onnx/DAE_baseline/denoiser.onnx
```

**Requirements:**

- CUDA-capable GPU
- TensorRT installed
- Sample audio for calibration (included in `artifacts/onnx/`)

**Output:** `artifacts/trt/DAE_baseline/denoiser.plan`

### 3. Prepare Triton Model Repository

For serving via [Triton Inference Server](https://developer.nvidia.com/triton-inference-server):

```bash
# onnx
poetry run speech-denoiser prepare_triton_repo model=dae server.backend=onnx
poetry run speech-denoiser prepare_triton_repo model=demucs server.backend=onnx

# TensorRT
poetry run speech-denoiser prepare_triton_repo model=dae server.backend=trt
```

Creates directory structure:

```
triton/
├── DAE_baseline/
│   ├── onnx/{config.pbtxt, meta.json, 1/model.onnx}
│   └── trt/{config.pbtxt, meta.json, 1/model.plan}
└── demucs_v3_tiny/
   └── onnx/{config.pbtxt, meta.json, 1/model.onnx}
```

## Inference Server (Triton)

### Launch Triton Inference Server

The server auto-discovers all models of the specified backend.

```bash
# Serve ONNX models (auto-loads all available ONNX models)
bash scripts/run_triton.sh --backend onnx

# Serve TensorRT models (auto-loads all available TRT models)
bash scripts/run_triton.sh --backend trt
```

In order to run docker container in the background:

```bash
bash scripts/run_triton.sh --backend onnx --detach
```

**Backend Notes:**

- **ONNX (Recommended)**: Full support for both models with variable-length audio
- **TensorRT**: DAE_baseline only. Demucs fails due to dynamic shape incompatibility with TensorRT

Triton will start on:

- **8000** - HTTP inference endpoint
- **8001** - gRPC endpoint
- **8002** - Metrics endpoint

### Send Inference Requests

```bash
# Test with Hydra config override
poetry run speech-denoiser triton_infer model=dae server.input_wav=/path/to/audio.wav
```

## Configuration & Hyperparameters

All hyperparameters are managed via [Hydra](https://hydra.cc/).

**Config location:** `configs/`

### Main config files:

- `config.yaml` – Main configuration, paths
- `model/dae.yaml` – DAE-specific hyperparameters
- `model/demucs.yaml` – Demucs-specific hyperparameters
- `audio/audio.yaml` – Audio preprocessing (sample rate, segment length)
- `dataset/dataset.yaml` – Data paths and batch settings
- `mlflow/tracking.yaml` – MLflow server URI and experiment naming
- `trainer/trainer.yaml` – Trainer hyperparameters
- `server/triton.yaml` – Configuration file for Triton Server

### Hierarchical structure:

```
configs/
├── config.yaml              # Root config
├── trainer
│   └── trainer.yaml
├── server
│   └── triton.yaml
├── model/
│   ├── dae.yaml            # DAE baseline
│   └── demucs.yaml         # Demucs model
├── audio/
│   └── audio.yaml          # Audio parameters
├── dataset/
│   └── dataset.yaml        # Data paths
└── mlflow/
    └── tracking.yaml       # MLflow settings
```

## Logging & Monitoring

### MLflow

Training runs are automatically logged to MLflow if available:

```bash
mlflow ui
```

### CSV & Plots

During training, metrics are also saved to CSV:

```
plots/<model_name>/
├── metrics.csv              # Full training metrics
├── metrics_last_row.csv     # Summary of final epoch
├── train_loss.png           # Loss curve
├── val_loss.png             # Validation loss curve
└── val_si_sdr.png           # SI-SDR metric curve
```

## Code Quality

### Pre-commit hooks

Automatic code quality checks before committing:

```bash
# Already installed with: pre-commit install

# Manual run on all files
pre-commit run -a
```

**Tools used:**

- **Black** – Code formatter
- **isort** – Import organizer
- **Flake8** – Linter
- **Prettier** – Non-Python files (YAML, Markdown, etc.)

## Project Structure

```
speech-denoiser-ml/
├── README.md                    # This file
├── pyproject.toml              # Python dependencies (Poetry)
├── dvc.yaml                     # DVC pipeline (data + artifacts)
├── dvc.lock                     # Locked DVC state
├── .pre-commit-config.yaml     # Pre-commit hooks config
├── .gitignore                  # Git ignore rules
│
├── speech_denoiser/            # Main package
│   ├── __init__.py
│   ├── commands.py             # CLI entry point
│   ├── train.py                # Training pipeline
│   ├── infer.py                # Inference script
│   ├── eval_test.py            # Test evaluation
│   ├── export.py               # ONNX export
│   ├── triton.py               # Triton server prep
│   ├── data.py                 # Data loading (PyTorch Lightning)
│   ├── losses.py               # Loss functions
│   ├── lightning_module.py     # PyTorch Lightning module
│   ├── utils.py                # Utility functions
│   └── models/
│       ├── dae.py              # DAE baseline model
│       └── demucs_wrapper.py   # Demucs model wrapper
│
├── configs/                    # Hydra configuration files
│   ├── config.yaml             # Root config
│   ├── model/{dae,demucs}.yaml
│   ├── audio/audio.yaml
│   ├── dataset/dataset.yaml
│   └── mlflow/tracking.yaml
│
├── scripts/                    # Shell/helper scripts
│   ├── build_trt_engine.sh     # TensorRT conversion
│   ├── run_triton.sh           # Launch Triton server
│   └── ...
│
├── data/                       # Datasets (managed by DVC)
│   ├── train/
│   └── test/
│
├── artifacts/                  # Training outputs
│   ├── checkpoints/            # Model checkpoints (.ckpt)
│   ├── onnx/                   # Exported ONNX models (DVC)
│   └── predictions/            # Inference results
│
├── triton/                     # Triton model repositories
│   ├── <model_name>/
│   │   ├── onnx/{config.pbtxt, meta.json, 1/model.onnx}
│   │   └── trt/{config.pbtxt, meta.json, 1/model.plan}
│
├── plots/                      # Training logs and plots
│   ├── <model_name>/
│   │   ├── metrics.csv
│   │   ├── metrics_last_row.csv
│   │   ├── train_loss.png
│   │   ├── val_loss.png
│   │   ├── val_si_sdr.png
│   │   └── lightning_logs/
│   └── README.md
│
└── .dvc/                       # DVC configuration
    └── config                  # Remote storage settings
```

## Reproducibility

For reproducible results:

1. **Fixed Seeds:** `seed: 42` in `config.yaml`
2. **Deterministic Data Split:** Speaker-based split with fixed seed
3. **Git Tracking:** Commit ID is logged to MLflow
4. **DVC Versioning:** Data and model artifacts are version-controlled

To reproduce training:

```bash
poetry run speech-denoiser train model=demucs
```

## Dependencies

All dependencies are specified in `pyproject.toml`:

**Core:**

- PyTorch ≥2.6.0 with torchaudio
- PyTorch Lightning ≥2.6.0
- Hydra-core ≥1.3.2
- DVC ≥3.64.2

**Optional:**

- `demucs` – For Demucs model (install with `-E demucs`)
- `tritonclient` – For Triton client (Linux only, install with `-E triton`)

For development:

```bash
poetry install --with dev
```

## Troubleshooting

### MLflow server not reachable

If training starts but MLflow logging fails, the training continues without logging. Start the MLflow server:

```bash
mlflow ui --host 127.0.0.1 --port 8080
```

### DVC pull fails

Ensure DVC remote is configured and accessible:

```bash
dvc remote list
dvc pull
```

### CUDA/GPU issues

Check PyTorch CUDA setup:

```bash
poetry run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

If CUDA is not available, PyTorch will automatically use CPU.

### Pre-commit hook failures

Some hooks may auto-fix files. Re-stage and commit:

```bash
pre-commit run -a
git add .
git commit -m "Fix code style"
```

## Contributing

1. Create a feature branch
2. Make changes and test locally
3. Run `pre-commit run -a` before committing
4. Push and create a pull request

## License

See [LICENSE](LICENSE) file for details.

## References

- Demucs: https://github.com/facebookresearch/demucs
- SI-SDR metric: https://arxiv.org/abs/1902.07891
- PyTorch Lightning: https://www.pytorchlightning.ai
- Hydra: https://hydra.cc
- DVC: https://dvc.org
- Triton Inference Server: https://github.com/triton-inference-server/server
