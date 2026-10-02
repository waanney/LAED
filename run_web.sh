#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHERPA_DIR="$SCRIPT_DIR/sherpa-onnx"

echo "================================================================="
echo "   LAED: Speech-to-Speech Web Studio (Interactive Web Demo)      "
echo "   Opening on: http://127.0.0.1:7860                             "
echo "================================================================="

exec nix develop "$SHERPA_DIR" --command bash -c "
    export PYTHONPATH=\"$SHERPA_DIR/sherpa-onnx/python:\$SHERPA_DIR/scripts:\${PYTHONPATH:-}\"
    python3 \"$SCRIPT_DIR/web/server.py\" \"\$@\"
" -- "$@"
