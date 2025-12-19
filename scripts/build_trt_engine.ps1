param(
  [string]$OnnxPath = "artifacts/onnx/denoiser.onnx",
  [string]$TritonRepo = "artifacts/triton",
  [string]$ModelName = "denoiser_trt",
  [int]$ModelVersion = 1,
  [string]$Image = "nvcr.io/nvidia/tensorrt:24.09-py3"
)

# Optional step: build TensorRT engine from ONNX via trtexec (inside NVIDIA container).
# Requires Docker + NVIDIA Container Toolkit.

$onnx = Resolve-Path $OnnxPath
$repo = Resolve-Path $TritonRepo

$engineDir = Join-Path $repo "$ModelName/$ModelVersion"
New-Item -ItemType Directory -Force -Path $engineDir | Out-Null

Write-Host "[INFO] ONNX: $onnx"
Write-Host "[INFO] Engine output: $engineDir/model.plan"

# Build engine (FP16 recommended on RTX GPUs)
docker run --rm --gpus=all `
  -v "${onnx}:/onnx/denoiser.onnx" `
  -v "${repo}:/models" `
  $Image `
  trtexec --onnx=/onnx/denoiser.onnx --saveEngine=/models/$ModelName/$ModelVersion/model.plan --fp16

# You must also create a matching config.pbtxt under /models/$ModelName.
Write-Host "[INFO] Done. Create config.pbtxt for TensorRT backend (platform: tensorrt_plan)."
