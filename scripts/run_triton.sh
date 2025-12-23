#!/bin/bash
# Start Triton Inference Server with auto-discovery of models.
#
# Usage:
#   bash scripts/run_triton.sh --backend onnx
#   bash scripts/run_triton.sh --backend trt --detach
#
# The script will find all models in triton/<model>/<backend>/ and serve them.

set -euo pipefail

BACKEND="${1:-onnx}"
DETACH=0
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.09-py3}"
CONTAINER_NAME="${TRITON_CONTAINER_NAME:-triton_speech_denoiser}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend|-b)
      BACKEND="$2"
      shift 2
      ;;
    --detach|-d)
      DETACH=1
      shift
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRITON_BASE="$PROJECT_ROOT/triton"

echo "=================================================="
echo "Triton Inference Server"
echo "=================================================="
echo "Backend: $BACKEND"
echo "Base directory: $TRITON_BASE"
echo ""

# Create temporary serving directory that combines all models of the specified backend
SERVING_DIR="/tmp/triton_models_${BACKEND}"
rm -rf "$SERVING_DIR"
mkdir -p "$SERVING_DIR"

echo "Looking for models with backend: $BACKEND..."
FOUND_MODELS=0

# Find all model directories: triton/<model>/<backend>/
if [[ -d "$TRITON_BASE" ]]; then
  for model_dir in "$TRITON_BASE"/*; do
    if [[ -d "$model_dir" ]]; then
      model_name="$(basename "$model_dir")"
      backend_dir="$model_dir/$BACKEND"

      if [[ -d "$backend_dir" ]]; then
        echo "  ✓ Found: $model_name/$BACKEND"
        # Triton expects: /models/<model_name>/{config.pbtxt, <version>/model.*}
        model_serving_dir="$SERVING_DIR/$model_name"
        mkdir -p "$model_serving_dir"
        cp -r "$backend_dir"/* "$model_serving_dir/" 2>/dev/null || true
        FOUND_MODELS=$((FOUND_MODELS + 1))
      fi
    fi
  done
fi

if [[ $FOUND_MODELS -eq 0 ]]; then
  echo ""
  echo "ERROR: No models found for backend '$BACKEND'"
  echo ""
  echo "Available models:"
  echo "  poetry run speech-denoiser prepare_triton_repo model=dae server.backend=$BACKEND"
  echo "  poetry run speech-denoiser prepare_triton_repo model=demucs server.backend=$BACKEND"
  exit 1
fi

echo ""
echo "Found $FOUND_MODELS model(s). Models will be served from:"
ls -d "$SERVING_DIR"/*/ | sed 's|^|  |'

echo ""
echo "=================================================="
echo "Starting Triton Server"
echo "=================================================="
echo "Ports: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)"
echo ""

if [[ "$DETACH" == "1" ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "Starting in detached mode..."
fi

DOCKER_ARGS=(
  --rm --gpus=all
  -p 8000:8000 -p 8001:8001 -p 8002:8002
  -v "$SERVING_DIR:/models"
  "$IMAGE"
  tritonserver --model-repository=/models
)

if [[ "$DETACH" == "1" ]]; then
  docker run -d --name "$CONTAINER_NAME" "${DOCKER_ARGS[@]}" >/dev/null 2>&1
  echo "[OK] Container started: $CONTAINER_NAME"
  echo ""
  echo "Next steps:"
  echo "  • View logs: docker logs -f $CONTAINER_NAME"
  echo "  • Stop: docker stop $CONTAINER_NAME"
  echo "  • Test: poetry run speech-denoiser triton_infer model=DAE_baseline input_wav=<audio.wav>"
else
  docker run "${DOCKER_ARGS[@]}"
fi
