# Rust model backends

The Rust core owns orchestration, context, cancellation, resampling and WAV output. Flutter owns
only the platform UI, microphone stream and playback surface.

Real inference should be added inside `rust/src/engines.rs` behind the `native-models` feature:

1. Bind pinned `whisper.cpp`, `llama.cpp` and `sherpa-onnx` revisions through their C APIs.
2. Implement `AsrEngine`, `LlmEngine` and `TtsEngine`; keep model contexts alive for warm latency.
3. Poll the supplied `is_cancelled` callback from each runtime's abort callback/token loop.
4. Apply Qwen's official chat template with thinking disabled, enforce 2,048 context tokens and
   96 output tokens using the real tokenizer, not a character estimate.
5. Return TTS audio at its model-native sample rate. The Rust core creates a valid WAV container.
6. Build Android `arm64-v8a` and iOS device/simulator libraries through flutter_rust_bridge.

Expected application-support layout:

```text
models/
├── whisper/ggml-tiny.bin
├── qwen3/model.gguf
└── tts-vais1000/
    ├── model.onnx
    ├── tokens.txt
    └── ...voice-specific files
```

Do not silently fall back in production. The current explicit `demoMode: true` keeps this source
scaffold usable until adapters exist; switch it off only after verified model installation.

