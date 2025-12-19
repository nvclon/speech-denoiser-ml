param(
  [string]$ModelRepo = "artifacts/triton",
  [string]$Image = "nvcr.io/nvidia/tritonserver:24.09-py3"
)

$repo = Resolve-Path $ModelRepo

Write-Host "[INFO] Using model repo: $repo"
Write-Host "[INFO] Starting Triton on ports 8000/8001/8002"

# Requires Docker + NVIDIA Container Toolkit (or WSL2 with GPU support)
# Exposes:
# - HTTP: 8000
# - GRPC: 8001
# - Metrics: 8002

docker run --rm --gpus=all `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${repo}:/models" `
  $Image `
  tritonserver --model-repository=/models
