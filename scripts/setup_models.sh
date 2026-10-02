#!/usr/bin/env bash
set -euo pipefail

# Download required lightweight speech models (ASR, VAD, TTS)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

cd "$MODELS_DIR"

echo "=== 1. Downloading Silero VAD ==="
if [ ! -f "silero_vad.onnx" ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
else
    echo "silero_vad.onnx already exists."
fi

echo "=== 2. Downloading Whisper Base English ASR (High Accuracy) ==="
if [ ! -d "sherpa-onnx-whisper-base.en" ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2
    tar -xjf sherpa-onnx-whisper-base.en.tar.bz2
    rm -f sherpa-onnx-whisper-base.en.tar.bz2
else
    echo "Whisper Base ASR already exists."
fi

echo "=== 3. Downloading Kokoro-TTS (English, 24kHz High-Fidelity) ==="
if [ ! -d "kokoro-en-v0_19" ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
    tar -xjf kokoro-en-v0_19.tar.bz2
    rm -f kokoro-en-v0_19.tar.bz2
else
    echo "Kokoro-TTS already exists."
fi

echo "=== Models setup complete! ==="
