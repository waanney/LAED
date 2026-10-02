#!/usr/bin/env python3
"""End-to-End Pipeline Dry Run Test (WAV -> ASR -> Streaming LLM -> Sentence Chunking -> TTS)."""

import os
from pathlib import Path
import sys
import time
import wave
import numpy as np

# Ensure sherpa_onnx is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sherpa-onnx/python"))
sys.path.insert(0, str(ROOT / "scripts"))

from voice_chat import (
    create_recognizer,
    create_tts,
    recognize,
    LocalLlmClient,
    SentenceChunker,
    LLM_URL,
    LLM_PORT,
)
import sherpa_onnx

def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16000, f"Expected 16kHz, got {wf.getframerate()}"
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

def main():
    print("=== S2S End-to-End Test ===")
    test_wav = ROOT / "sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav"
    assert test_wav.exists(), f"Test wav missing at {test_wav}"
    
    # 1. Load ASR
    asr = create_recognizer()
    samples = load_wav(test_wav)
    print(f"Loaded test WAV: {len(samples)} samples ({len(samples)/16000:.2f}s)")
    
    # 2. Transcribe
    t0 = time.perf_counter()
    transcript = recognize(asr, samples)
    asr_time = (time.perf_counter() - t0) * 1000
    print(f"ASR Transcript ({asr_time:.0f}ms): '{transcript}'")
    assert transcript, "ASR transcript was empty!"
    
    # 3. LLM Server Test
    llm = LocalLlmClient(LLM_URL)
    llm.start()
    
    # 4. TTS Test
    tts = create_tts()
    gen_cfg = sherpa_onnx.GenerationConfig()
    gen_cfg.speed = 1.0
    
    # 5. Stream LLM -> Sentence Chunk -> Synthesize
    print("\n--- Starting Streaming Generation & Parallel Synthesis ---")
    sentences = []
    tokens = []
    first_token_time = None
    first_audio_time = None
    llm_start = time.perf_counter()
    
    def on_token(t: str):
        tokens.append(t)
        print(t, end="", flush=True)

    def on_sentence(s: str):
        nonlocal first_audio_time
        sentences.append(s)
        t_gen_start = time.perf_counter()
        audio = tts.generate(s, gen_cfg)
        gen_duration = (time.perf_counter() - t_gen_start) * 1000
        if first_audio_time is None:
            first_audio_time = time.perf_counter()
        print(f"\n   [TTS synthesized sentence in {gen_duration:.0f}ms: '{s}' ({len(audio.samples)} samples)]")

    print("[Assistant]: ", end="", flush=True)
    reply, ttft = llm.stream_chat(transcript, on_sentence, on_token)
    
    total_time = (time.perf_counter() - llm_start) * 1000
    ttft_ms = ttft * 1000
    ttfa_ms = ((first_audio_time - llm_start) * 1000) if first_audio_time else 0
    
    print("\n" + "=" * 50)
    print("=== Pipeline Test Results ===")
    print(f"Total Sentences Produced: {len(sentences)}")
    print(f"ASR Latency:             {asr_time:.0f} ms")
    print(f"Time To First Token (TTFT): {ttft_ms:.0f} ms")
    print(f"Time To First Audio (TTFA): {ttfa_ms:.0f} ms")
    print(f"Total LLM + TTS Duration:   {total_time:.0f} ms")
    print("=" * 50)
    print("SUCCESS: End-to-end Speech-to-Speech test passed cleanly!")
    
    llm.close()

if __name__ == "__main__":
    main()
