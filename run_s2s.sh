#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHERPA_DIR="$SCRIPT_DIR/sherpa-onnx"

echo "================================================================="
echo "   LAED: Real-Time Streaming Speech-to-Speech Engine             "
echo "   Audio / VAD / ASR / TTS: sherpa-onnx                          "
echo "   LLM Brain: Edge0 / llama-server                               "
echo "================================================================="

exec nix develop "$SHERPA_DIR" --command bash -c "
    export PYTHONPATH=\"$SHERPA_DIR/sherpa-onnx/python:\${PYTHONPATH:-}\"
    python3 \"$SHERPA_DIR/scripts/voice_chat.py\" \"\$@\"
" -- "$@"
