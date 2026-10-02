#!/usr/bin/env python3
"""Lightweight Web Demo Server for Speech-to-Speech (sherpa-onnx + Edge0/LLM)."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# Ensure sherpa_onnx is in path
ROOT = Path(__file__).resolve().parent.parent
SHERPA_DIR = ROOT / "sherpa-onnx"
sys.path.insert(0, str(SHERPA_DIR / "sherpa-onnx/python"))
sys.path.insert(0, str(SHERPA_DIR / "scripts"))

from voice_chat import (
    create_recognizer,
    create_tts,
    recognize,
    LocalLlmClient,
    SentenceChunker,
    LLM_URL,
    LLM_PORT,
    ASR_DIR,
    TTS_DIR,
    KOKORO_DIR,
    TTS_SID,
    TTS_SPEED,
)
import sherpa_onnx

WEB_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("WEB_PORT", "7860"))
HOST = os.environ.get("WEB_HOST", "0.0.0.0")

# Global instances (initialized on startup)
asr_engine: sherpa_onnx.OnlineRecognizer | None = None
tts_engine: sherpa_onnx.OfflineTts | None = None
llm_client: LocalLlmClient | None = None
tts_gen_cfg: sherpa_onnx.GenerationConfig | None = None


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int16_data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        wf.writeframes(int16_data.tobytes())
    return buf.getvalue()


def wav_bytes_to_samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
        if width == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        return samples, sr


class WebDemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Suppress noisy access logs; print only important events
        if "/api/" in (args[0] if args else ""):
            print(f"[WebAPI] {self.command} {args[0]} - {args[1]}")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"

        if path == "/api/status":
            tts_name = "Kokoro (af_heart)" if (KOKORO_DIR / "model.onnx").is_file() else TTS_DIR.name
            self.send_json(
                {
                    "status": "ok",
                    "llm_url": llm_client.url if llm_client else LLM_URL,
                    "asr_model": ASR_DIR.name,
                    "tts_model": tts_name,
                }
            )
            return

        # Serve static files from WEB_DIR
        file_path = (WEB_DIR / path.lstrip("/")).resolve()
        if file_path.is_file() and str(file_path).startswith(str(WEB_DIR)):
            mime_type, _ = mimetypes.guess_type(str(file_path))
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/chat-voice":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "Empty audio payload")
                return

            wav_data = self.rfile.read(content_length)
            try:
                samples, sr = wav_bytes_to_samples(wav_data)
            except Exception as e:
                self.send_error(400, f"Invalid WAV audio: {e}")
                return

            self.handle_voice_chat(samples)

        elif path == "/api/chat-text":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            user_text = body.get("text", "").strip()
            if not user_text:
                self.send_error(400, "Empty text prompt")
                return

            self.handle_text_chat(user_text)

        else:
            self.send_error(404, "Endpoint not found")

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def handle_voice_chat(self, samples: np.ndarray) -> None:
        """Processes voice: ASR -> Streaming LLM -> Sentence Chunking -> TTS -> SSE Output."""
        global asr_engine, tts_engine, llm_client, tts_gen_cfg

        # 1. ASR
        t0 = time.perf_counter()
        transcript = recognize(asr_engine, samples)
        asr_time_ms = (time.perf_counter() - t0) * 1000

        self.stream_s2s_response(transcript, asr_time_ms)

    def handle_text_chat(self, user_text: str) -> None:
        """Processes text input directly into Streaming LLM -> TTS."""
        self.stream_s2s_response(user_text, asr_time_ms=0)

    def stream_s2s_response(self, transcript: str, asr_time_ms: float) -> None:
        global tts_engine, llm_client, tts_gen_cfg

        # Set up SSE headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def send_sse(event_data: dict) -> None:
            msg = f"data: {json.dumps(event_data)}\n\n".encode("utf-8")
            self.wfile.write(msg)
            self.wfile.flush()

        # Send initial transcript event
        send_sse({"type": "transcript", "text": transcript, "asr_ms": round(asr_time_ms)})

        if not transcript:
            send_sse({"type": "error", "message": "No speech detected"})
            send_sse({"type": "done", "metrics": {"asr_ms": round(asr_time_ms)}})
            return

        # 2. LLM Stream + TTS
        first_token_time = None
        first_audio_time = None
        llm_start = time.perf_counter()
        sentence_idx = 0

        def on_token(token: str) -> None:
            send_sse({"type": "token", "token": token})

        def on_sentence(sentence: str) -> None:
            nonlocal first_audio_time, sentence_idx
            sentence_idx += 1
            t_gen_start = time.perf_counter()
            audio = tts_engine.generate(sentence, tts_gen_cfg)
            gen_ms = (time.perf_counter() - t_gen_start) * 1000
            if first_audio_time is None:
                first_audio_time = time.perf_counter()

            # Encode audio to base64 WAV
            wav_bytes = samples_to_wav_bytes(np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
            b64_audio = base64.b64encode(wav_bytes).decode("ascii")

            send_sse(
                {
                    "type": "audio_sentence",
                    "index": sentence_idx,
                    "sentence": sentence,
                    "audio": b64_audio,
                    "gen_ms": round(gen_ms),
                }
            )

        try:
            reply, ttft_sec = llm_client.stream_chat(transcript, on_sentence, on_token)
            total_duration_ms = (time.perf_counter() - llm_start) * 1000
            ttft_ms = ttft_sec * 1000
            ttfa_ms = ((first_audio_time - llm_start) * 1000) if first_audio_time else 0

            send_sse(
                {
                    "type": "done",
                    "metrics": {
                        "asr_ms": round(asr_time_ms),
                        "ttft_ms": round(ttft_ms),
                        "ttfa_ms": round(ttfa_ms),
                        "total_ms": round(total_duration_ms),
                        "sentences": sentence_idx,
                    },
                }
            )
        except Exception as e:
            send_sse({"type": "error", "message": str(e)})


def main() -> None:
    global asr_engine, tts_engine, llm_client, tts_gen_cfg

    print("=" * 64)
    print("      LAED: Speech-to-Speech Interactive Web Studio        ")
    print("=" * 64)

    # Initialize models
    asr_engine = create_recognizer()
    tts_engine = create_tts()
    tts_gen_cfg = sherpa_onnx.GenerationConfig()
    tts_gen_cfg.speed = TTS_SPEED
    tts_gen_cfg.sid = TTS_SID

    llm_client = LocalLlmClient(LLM_URL)
    llm_client.start()

    server = ThreadingHTTPServer((HOST, PORT), WebDemoHandler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n🚀 Web Demo ready and running at: {url}")
    print(f"👉 Open {url} in your browser to start talking with Amy!\n" + "-" * 64)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Stopping web server...]")
    finally:
        server.server_close()
        if llm_client:
            llm_client.close()
        print("[Done] Web server exited.")


if __name__ == "__main__":
    main()
