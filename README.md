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

## MVP status

- Implemented in source: push-to-talk PCM streaming, six UI phases, cancellation, session clearing,
  15-second cap, context/output budgets, audio resampling and playback.
- Awaiting native backend integration: whisper.cpp, llama.cpp/Qwen3 non-thinking template and
  sherpa-onnx Vietnamese TTS.
- Later phases: verified model downloader, stage benchmark report, sentence-level LLM/TTS overlap,
  VAD and barge-in.

