# LAED

Offline speech-to-speech MVP with a Flutter UI and a shared Rust core, derived from
`Speech_to_Speech_Mobile_Plan.md`.

```text
Flutter microphone (PCM16) → Rust resample → ASR → bounded context → LLM → TTS
                                                                ↓
Flutter playback (WAV bytes) ←───────────────────────────────────┘
```

## Architecture

- Flutter owns presentation, microphone permission/capture, app lifecycle and playback.
- Rust owns input validation, 15-second limit, resampling, the pipeline state sequence, monotonic
  `turn_id`, stale-result rejection, bounded conversation history, prompt limits and WAV creation.
- `flutter_rust_bridge` generates typed asynchronous bindings. The same Rust API can later generate
  Swift bindings for iOS rather than duplicating pipeline logic.
- A deterministic demo backend exercises every stage without claiming to perform real AI. The UI
  labels demo mode explicitly.

The real model adapters remain deliberately isolated behind Rust traits in `rust/src/engines.rs`.
They are not faked: production mode refuses to start until pinned whisper.cpp, llama.cpp and
sherpa-onnx adapters and model files are present.

## Source map

```text
lib/
├── main.dart                              Flutter entry point
└── src/presentation/                      Push-to-talk UI and platform audio
rust/src/
├── api/speech.rs                          Cross-platform public pipeline API
├── audio.rs                               PCM validation/resampling/WAV
├── engines.rs                             ASR/LLM/TTS traits and demo engines
└── memory.rs                              Bounded complete exchanges
docs/
├── MODEL_BACKENDS.md                      Real runtime integration contract
└── PLATFORM_SETUP.md                      Generated runners, bindings, permissions
```

## Before the first build

This repository contains authored source only. Generate standard Flutter Android/iOS runners and
the bridge glue using the pinned workflow in `docs/PLATFORM_SETUP.md`. Generated bridge files and
model weights are intentionally ignored by Git.

No audio or transcripts are persisted. The example model manifest uses required placeholders for
revisions, checksums and licenses; fill and verify them before distributing any weights.

## Real-Time Speech-to-Speech Pipeline (Edge0 × sherpa-onnx)

Integrated real-time streaming speech-to-speech assistant powered by:
- **ASR**: `sherpa-onnx` Streaming Zipformer Transducer (16kHz offline speech-to-text)
- **VAD**: Silero VAD (real-time voice activity detection)
- **LLM**: `Edge0-8B` MoE with OpenAI-compatible SSE streaming endpoint
- **TTS**: `hexgrad/Kokoro-82M` (`af_heart`) high-fidelity 24kHz neural speech
- **Role**: AI English Teacher (Teacher Sarah) for conversational English practice

### Architecture

```text
Microphone (16kHz) → Silero VAD → Zipformer ASR → Edge0-8B (Token Stream)
                                                       ↓ (Sentence Chunker)
Speaker Output (24kHz) ← Kokoro TTS (af_heart) ← Sentence Queue
```

### Quick Start

1. **Interactive Web Studio** (Browser UI with Audio Reactive Visualizer):
   ```bash
   ./run_web.sh
   # Open http://127.0.0.1:7860
   ```

2. **Terminal Voice Chat**:
   ```bash
   ./run_s2s.sh
   ```

3. **Connecting Remote Backend (e.g. Vast.ai GPU)**:
   ```bash
   LLM_URL=https://your-vast-ai-tunnel.trycloudflare.com ./run_s2s.sh
   ```

4. **GitHub Pages Deployment**:
   The `web/` directory is standalone and can be deployed directly to GitHub Pages. Use the in-app **Settings** modal to connect to your remote backend.

