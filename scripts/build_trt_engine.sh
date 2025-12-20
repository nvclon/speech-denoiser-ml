#!/bin/bash
ONNX_PATH=${1:-"artifacts/onnx/DAE_baseline/denoiser.onnx"}
TRITON_REPO=${2:-"artifacts/triton"}
MODEL_NAME=${3:-"denoiser_trt"}
MODEL_VERSION=${4:-1}
IMAGE=${5:-"nvcr.io/nvidia/tensorrt:24.09-py3"}

ONNX_ABS=$(realpath "$ONNX_PATH")
REPO_ABS=$(realpath "$TRITON_REPO")
ENGINE_DIR="$REPO_ABS/$MODEL_NAME/$MODEL_VERSION"

mkdir -p "$ENGINE_DIR"

echo "[INFO] ONNX: $ONNX_ABS"
echo "[INFO] Engine output: $ENGINE_DIR/model.plan"

docker run --rm --gpus=all \
  -v "$ONNX_ABS:/onnx/denoiser.onnx" \
  -v "$REPO_ABS:/models" \
  "$IMAGE" \
  trtexec --onnx=/onnx/denoiser.onnx --saveEngine=/models/$MODEL_NAME/$MODEL_VERSION/model.plan --fp16

echo "[INFO] Done. Create config.pbtxt for TensorRT backend (platform: tensorrt_plan)."
