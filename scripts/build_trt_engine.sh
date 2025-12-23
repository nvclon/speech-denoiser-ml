#!/bin/bash
# build_trt_engine.sh

ONNX_PATH=$1  # например: artifacts/onnx/DAE_baseline/denoiser.onnx

if [ -z "$ONNX_PATH" ]; then
  echo "Usage: $0 <path_to_onnx>"
  exit 1
fi

# Извлекаем модель из пути (DAE_baseline или demucs_v3_tiny)
MODEL_DIR=$(basename $(dirname "$ONNX_PATH"))

# Проверка: TensorRT не поддерживает Demucs
if [[ "$MODEL_DIR" == *"demucs"* ]]; then
  echo ""
  echo "======================================================================"
  echo "ERROR: TensorRT backend not supported for $MODEL_DIR"
  echo "======================================================================"
  echo ""
  echo "Reason:"
  echo "  Demucs uses dynamic shapes (-1 time dimension) for variable-length"
  echo "  audio input. TensorRT requires fixed tensor shapes and cannot"
  echo "  reshape tensors with incompatible dimensions."
  echo ""
  echo "Solutions:"
  echo "  1. Use ONNX backend instead (recommended):"
  echo "     poetry run speech-denoiser prepare_triton_repo model=demucs triton.backend=onnx"
  echo ""
  echo "  2. Use DAE_baseline for TensorRT (fully compatible):"
  echo "     bash scripts/build_trt_engine.sh artifacts/onnx/DAE_baseline/denoiser.onnx"
  echo ""
  echo "======================================================================"
  echo ""
  exit 1
fi

ONNX_ABS=$(realpath "$ONNX_PATH")
OUTPUT_DIR="artifacts/trt/$MODEL_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Converting: $ONNX_ABS"
echo "[INFO] Output: $OUTPUT_DIR/denoiser.plan"

docker run --rm --gpus=all \
  --user "$(id -u):$(id -g)" \
  -v "$(dirname "$ONNX_ABS"):/onnx" \
  -v "$(realpath "$OUTPUT_DIR"):/output" \
  nvcr.io/nvidia/tensorrt:24.09-py3 \
  trtexec --onnx=/onnx/denoiser.onnx \
          --saveEngine=/output/denoiser.plan \
          --fp16 \
          --minShapes=noisy:1x1x16000 \
          --optShapes=noisy:1x1x48000 \
          --maxShapes=noisy:1x1x160000

echo "[INFO] Done: $OUTPUT_DIR/denoiser.plan"
