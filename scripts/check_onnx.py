import sys
from pathlib import Path

import onnx


def _pick_default_onnx_path() -> Path:
    candidates = list(Path("artifacts/onnx").glob("*/denoiser.onnx"))
    if not candidates:
        raise FileNotFoundError(
            "No ONNX exports found under artifacts/onnx/*/denoiser.onnx. "
            "Run: poetry run speech-denoiser export_onnx ..."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _pick_default_onnx_path()
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"[OK] ONNX model {onnx_path} is valid.")
        print(f"Graph inputs: {[node.name for node in model.graph.input]}")
        print(f"Graph outputs: {[node.name for node in model.graph.output]}")
        return 0
    except Exception as e:
        print(f"[FAIL] ONNX model check failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
