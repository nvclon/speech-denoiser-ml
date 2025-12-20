#!/bin/bash
set -euo pipefail

DETACH=0
if [[ "${1:-}" == "--detach" || "${1:-}" == "-d" ]]; then
  DETACH=1
  shift
fi

MODEL_REPO=${1:-"artifacts/triton"}
IMAGE=${2:-"nvcr.io/nvidia/tritonserver:24.09-py3"}
CONTAINER_NAME=${TRITON_CONTAINER_NAME:-"triton_speech_denoiser"}

REPO_ABS=$(realpath "$MODEL_REPO")

echo "[INFO] Using model repo: $REPO_ABS"
echo "[INFO] Starting Triton on ports 8000/8001/8002"

if [[ "$DETACH" == "1" ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

DOCKER_ARGS=(--rm --gpus=all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$REPO_ABS:/models" \
  "$IMAGE" \
  tritonserver --model-repository=/models)

if [[ "$DETACH" == "1" ]]; then
  echo "[INFO] Detach mode. Container name: $CONTAINER_NAME"
  docker run -d --name "$CONTAINER_NAME" "${DOCKER_ARGS[@]}" >/dev/null
  echo "[OK] Triton is starting in background. Stop with: docker stop $CONTAINER_NAME"
else
  docker run "${DOCKER_ARGS[@]}"
fi
