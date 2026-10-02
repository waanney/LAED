#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/sherpa-onnx/python${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$ROOT_DIR/scripts/voice_chat.py" "$@"
