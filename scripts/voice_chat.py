#!/usr/bin/env python3
"""Low-latency streaming voice chat: Microphone -> VAD -> ASR -> Edge0/LLM (SSE Stream) -> Sentence Chunking -> TTS -> Speaker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Generator

import numpy as np
import sherpa_onnx

try:
    import sounddevice as sd
except Exception:
    sd = None


ROOT = Path(__file__).resolve().parent.parent

def resolve_model_dir(env_name: str, rel_name: str) -> Path:
    env_path = os.environ.get(env_name)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for c in [
        ROOT / rel_name,
        ROOT / "models" / rel_name,
        ROOT / "sherpa-onnx" / rel_name,
    ]:
        if c.exists():
            return c
    return ROOT / rel_name

def resolve_default_asr_dir() -> Path:
    env_path = os.environ.get("ASR_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for name in [
        "sherpa-onnx-whisper-base.en",
        "sherpa-onnx-whisper-tiny.en",
        "sherpa-onnx-whisper-small.en",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    ]:
        for prefix in [ROOT, ROOT / "models", ROOT / "sherpa-onnx"]:
            candidate = prefix / name
            if candidate.is_dir():
                return candidate
    return ROOT / "models" / "sherpa-onnx-whisper-base.en"

ASR_DIR = resolve_default_asr_dir()
TTS_DIR = resolve_model_dir("TTS_DIR", "vits-piper-en_US-amy-low")
KOKORO_DIR = resolve_model_dir("KOKORO_DIR", "kokoro-en-v0_19")
def resolve_default_llm() -> Path:
    env_path = os.environ.get("LLM_MODEL")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for c in [
        ROOT / "models/edge0-8b.gguf",
        ROOT / "models/edge0-8b-instruct.gguf",
        ROOT / "models/qwen2.5-7b-instruct-q4_k_m.gguf",
        ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ]:
        if c.is_file():
            return c
    return ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

LLM_MODEL = resolve_default_llm()
VAD_MODEL = Path(os.environ.get("VAD_MODEL", ROOT / "models/silero_vad.onnx"))

SAMPLE_RATE = 16000
ASR_THREADS = int(os.environ.get("ASR_THREADS", "6"))
LLM_THREADS = int(os.environ.get("LLM_THREADS", "4"))
TTS_THREADS = int(os.environ.get("TTS_THREADS", "8"))
TTS_SID = int(os.environ.get("TTS_SID", "0"))  # 0: af_heart, 1: af_bella, 2: af_nicole, 3: af_sarah, 4: af_sky
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "96"))

def resolve_llm_url() -> str:
    env_url = os.environ.get("LLM_URL")
    if env_url:
        return env_url
    for port in (11434, 18080, 8000):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=0.3):
                return f"http://127.0.0.1:{port}"
        except Exception:
            pass
    return f"http://127.0.0.1:{os.environ.get('LLM_PORT', '18080')}"

LLM_PORT = int(os.environ.get("LLM_PORT", "18080"))
LLM_URL = resolve_llm_url()
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "Edge0-8B")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_SECONDS = float(os.environ.get("VAD_SILENCE_SECONDS", "0.5"))
VAD_MAX_SPEECH_SECONDS = float(os.environ.get("VAD_MAX_SPEECH_SECONDS", "20"))
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.15"))

SYSTEM_PROMPT = (
    "You are Sarah, an encouraging and patient native English teacher. "
    "Your sole mission is to help the student practice spoken English naturally. "
    "Always communicate strictly in natural, conversational English with smooth flowing sentences. "
    "Keep your spoken responses short, concise, and natural, exactly 1 to 2 complete sentences (maximum 30 words). "
    "Speak smoothly and avoid choppy fragments or unnecessary commas. "
    "If the student makes any grammatical or vocabulary errors, "
    "gently correct them in a friendly manner, then ask an engaging follow-up question. "
    "Never use emojis, markdown, asterisks, bullet points, or special characters, "
    "as your text is directly synthesized into real-time speech."
)

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e."
}


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))


class SentenceChunker:
    """Buffers incoming streaming tokens and yields ready-to-speak sentences without awkward pauses."""

    def __init__(self, min_chars: int = 35):
        self.buffer = ""
        self.min_chars = min_chars
        self.in_think = False
        self.first_chunk_sent = False

    def push(self, token: str) -> list[str]:
        self.buffer += token

        # Strip reasoning/thinking tags (e.g. <think>...</think>)
        if "<think>" in self.buffer:
            self.in_think = True
        if self.in_think:
            if "</think>" in self.buffer:
                _, self.buffer = self.buffer.split("</think>", 1)
                self.in_think = False
            else:
                return []

        sentences = []
        while True:
            # 1. Look for true sentence boundary: [.!?\n] followed by space or end
            match = re.search(r"([.!?\n]+)(\s+|$)", self.buffer)
            if match:
                end_idx = match.end()
                candidate = self.buffer[:end_idx].strip()

                # Ignore abbreviations like Dr., Mr., etc.
                words = candidate.split()
                last_word = words[-1].lower() if words else ""
                if last_word in ABBREVIATIONS:
                    break

                # Ignore decimal numbers like 3.14
                if re.search(r"\d+\.\d*$", candidate):
                    break

                # Only split if sentence is reasonably complete (at least 20 chars) or explicit newline
                # This prevents splitting on single words like "Yes." or "Hi." prematurely
                if len(candidate) >= 20 or match.group(1) == "\n" or len(words) >= 4:
                    sentences.append(candidate)
                    self.buffer = self.buffer[end_idx:].lstrip()
                    self.first_chunk_sent = True
                    continue
                else:
                    # Let the buffer accumulate more tokens for a fuller natural sentence
                    break

            # 2. If buffer has grown very long (> 75 chars) and has a clause break, split gently
            if len(self.buffer) >= 75:
                clause_match = re.search(r"([,;:]+)(\s+)", self.buffer)
                if clause_match:
                    end_idx = clause_match.end()
                    candidate = self.buffer[:end_idx].strip()
                    if len(candidate) >= 30:
                        sentences.append(candidate)
                        self.buffer = self.buffer[end_idx:].lstrip()
                        self.first_chunk_sent = True
                        continue

            break

        return sentences

    def flush(self) -> list[str]:
        rem = self.buffer.strip()
        self.buffer = ""
        return [rem] if rem else []


class LocalLlmClient:
    """Client for local LLM (Edge0 serve or llama-server) supporting SSE streaming."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.process: subprocess.Popen[bytes] | None = None
        self.log = None
        self.model_name = os.environ.get("LLM_MODEL_NAME", "Edge0-8B")
        self.sessions: dict[str, list[dict[str, str]]] = {}

    def get_session_history(self, session_id: str = "default") -> list[dict[str, str]]:
        if session_id not in self.sessions:
            # Prevent memory leaks: keep up to 100 active sessions
            if len(self.sessions) > 100:
                oldest = next(iter(self.sessions))
                del self.sessions[oldest]
            self.sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return self.sessions[session_id]

    @property
    def history(self) -> list[dict[str, str]]:
        return self.get_session_history("default")

    @history.setter
    def history(self, val: list[dict[str, str]]) -> None:
        self.sessions["default"] = val

    def _healthy(self) -> bool:
        for endpoint in (f"{self.url}/health", f"{self.url}/v1/models"):
            try:
                with urllib.request.urlopen(endpoint, timeout=0.5):
                    return True
            except (OSError, urllib.error.URLError):
                pass
        return False

    def start(self) -> None:
        if self._healthy():
            print(f"[LLM Engine] Connected to active server at {self.url}")
            try:
                with urllib.request.urlopen(f"{self.url}/v1/models", timeout=1.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    if models:
                        # Prioritize llama3.1:8b or any 8b/7b model over heavy 35b models
                        preferred = [m for m in models if "llama3.1" in m.lower() or "8b" in m.lower() or "7b" in m.lower()]
                        if preferred:
                            self.model_name = preferred[0]
                        elif self.model_name not in models:
                            self.model_name = models[0]
                        print(f"[LLM Engine] Auto-selected active model: {self.model_name}")
            except Exception:
                pass
            return

        binary = os.environ.get("LLM_BIN") or shutil.which("llama-server")
        if not binary:
            raise RuntimeError(
                f"LLM server not found at {self.url} and 'llama-server' binary is missing. "
                "Ensure direnv/nix develop is active or start 'edge0 serve' / 'llama-server' first."
            )

        print(f"[LLM Engine] Starting local llama-server on port {LLM_PORT} with model: {LLM_MODEL.name}...")
        log_path = Path(os.environ.get("TMPDIR", "/tmp")) / "sherpa-onnx-llm.log"
        self.log = log_path.open("wb")
        ngl = os.environ.get("LLM_NGL", "99")
        self.process = subprocess.Popen(
            [
                binary,
                "--model", str(LLM_MODEL),
                "--host", "127.0.0.1",
                "--port", str(LLM_PORT),
                "--ctx-size", "1024",
                "--threads", str(LLM_THREADS),
                "-ngl", ngl,
                "--parallel", "1",
                "--no-webui",
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

        for _ in range(120):
            if self._healthy():
                print(f"[LLM Engine] Server ready at {self.url}!")
                return
            if self.process.poll() is not None:
                break
            time.sleep(0.25)
        raise RuntimeError(f"Could not start LLM server. Check logs at {log_path}")

    def stream_chat(
        self,
        transcript: str,
        on_sentence: Callable[[str], None],
        on_token: Callable[[str], None],
        session_id: str = "default",
    ) -> tuple[str, float]:
        """Streams LLM tokens, chunks them into sentences, and returns (full_text, ttft_seconds)."""
        history = self.get_session_history(session_id)
        history.append({"role": "user", "content": transcript})
        # Keep sliding bounded window (system + last 6 messages)
        history = history[:1] + history[-6:]
        self.sessions[session_id] = history

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": history,
                "temperature": 0.7,
                "max_tokens": LLM_MAX_TOKENS,
                "stream": True,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        chunker = SentenceChunker(min_chars=35)
        full_reply: list[str] = []
        ttft: float | None = None
        start_time = time.perf_counter()

        with urllib.request.urlopen(request, timeout=30) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if not content:
                    continue

                if ttft is None:
                    ttft = time.perf_counter() - start_time

                full_reply.append(content)
                on_token(content)

                sentences = chunker.push(content)
                for sent in sentences:
                    on_sentence(sent)

        # Flush any remaining buffer text
        for sent in chunker.flush():
            on_sentence(sent)

        reply_str = "".join(full_reply).strip()
        if reply_str:
            self.sessions[session_id].append({"role": "assistant", "content": reply_str})

        return reply_str, ttft or 0.0

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()


class StreamingTtsPlayer:
    """Pipelined TTS synthesizer and audio player.
    Synthesizes sentence N+1 while sentence N is playing out the speaker.
    """

    def __init__(self, tts: sherpa_onnx.OfflineTts) -> None:
        self.tts = tts
        self.gen_cfg = sherpa_onnx.GenerationConfig()
        self.gen_cfg.speed = TTS_SPEED
        self.gen_cfg.sid = TTS_SID
        self.sample_rate = tts.sample_rate

        self.sentence_queue: queue.Queue[tuple[int, str | None]] = queue.Queue()
        self.active_turn_id = 0
        self._is_busy = False
        self._busy_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._turn_done_event = threading.Event()

        self.on_first_audio: Callable[[float], None] | None = None
        self._first_audio_fired = False

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def set_turn(self, turn_id: int, on_first_audio: Callable[[float], None] | None) -> None:
        self.active_turn_id = turn_id
        self.on_first_audio = on_first_audio
        self._first_audio_fired = False
        self._turn_done_event.clear()
        with self._busy_lock:
            self._is_busy = True

    def enqueue_sentence(self, sentence: str) -> None:
        self.sentence_queue.put((self.active_turn_id, sentence))

    def end_turn(self) -> None:
        """Signal that no more sentences will arrive for current turn."""
        self.sentence_queue.put((self.active_turn_id, None))

    def wait_until_done(self) -> None:
        """Wait until all sentences for the turn have finished playing."""
        self._turn_done_event.wait()

    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._is_busy

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                turn_id, sentence = self.sentence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Drop stale turns
            if turn_id != self.active_turn_id:
                self.sentence_queue.task_done()
                continue

            if sentence is None:
                # End of sentences for this turn
                with self._busy_lock:
                    self._is_busy = False
                self._turn_done_event.set()
                self.sentence_queue.task_done()
                continue

            # Synthesize sentence
            clean_text = sentence.strip()
            if not clean_text:
                self.sentence_queue.task_done()
                continue

            try:
                audio = self.tts.generate(clean_text, self.gen_cfg)
                if len(audio.samples) > 0 and turn_id == self.active_turn_id:
                    # Fire TTFA on first audio chunk hitting speaker
                    if not self._first_audio_fired and self.on_first_audio:
                        self.on_first_audio(time.perf_counter())
                        self._first_audio_fired = True

                    if sd is not None:
                        sd.play(samples, self.sample_rate)
                        sd.wait()
            except Exception as e:
                print(f"\n[TTS Error] Failed to play sentence '{clean_text}': {e}", file=sys.stderr)
            finally:
                self.sentence_queue.task_done()

    def stop(self) -> None:
        self._stop_event.set()
        if sd is not None:
            sd.stop()


def create_recognizer() -> sherpa_onnx.OnlineRecognizer | sherpa_onnx.OfflineRecognizer:
    provider = os.environ.get("SHERPA_PROVIDER", "cpu")
    num_threads = ASR_THREADS

    # 1. Check OpenAI Whisper models (Best for conversational English, accents, and punctuation)
    whisper_dirs = [
        ASR_DIR,
        ROOT / "models" / "sherpa-onnx-whisper-base.en",
        ROOT / "sherpa-onnx-whisper-base.en",
        ROOT / "models" / "sherpa-onnx-whisper-tiny.en",
        ROOT / "sherpa-onnx-whisper-tiny.en",
        ROOT / "models" / "sherpa-onnx-whisper-small.en",
        ROOT / "sherpa-onnx-whisper-small.en",
    ]
    for wdir in whisper_dirs:
        if not wdir.is_dir() or "whisper" not in wdir.name.lower():
            continue
        encoders = list(wdir.glob("*encoder*.onnx"))
        decoders = list(wdir.glob("*decoder*.onnx"))
        tokens = list(wdir.glob("*tokens*.txt"))
        if encoders and decoders and tokens:
            enc = next((e for e in encoders if "int8" in e.name), encoders[0])
            dec = next((d for d in decoders if "int8" in d.name), decoders[0])
            tok = tokens[0]
            print(f"[ASR Engine] Loading OpenAI Whisper ASR ({wdir.name})...")
            print(f"             Encoder: {enc.name} | Decoder: {dec.name}")
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(enc),
                decoder=str(dec),
                tokens=str(tok),
                language="en",
                task="transcribe",
                num_threads=num_threads,
                provider=provider,
            )
            recognizer._model_name = f"Whisper ({wdir.name})"
            return recognizer

    # 2. Check SenseVoice models
    sense_dirs = [
        ASR_DIR,
        ROOT / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        ROOT / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        ROOT / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
    ]
    for sdir in sense_dirs:
        if not sdir.is_dir() or "sense-voice" not in sdir.name.lower():
            continue
        models = [f for f in sdir.glob("*.onnx") if "encoder" not in f.name and "decoder" not in f.name]
        tokens = list(sdir.glob("tokens.txt"))
        if models and tokens:
            m = next((f for f in models if "int8" in f.name), models[0])
            print(f"[ASR Engine] Loading FunAudioLLM SenseVoice ASR ({sdir.name})...")
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(m),
                tokens=str(tokens[0]),
                use_itn=True,
                num_threads=num_threads,
                provider=provider,
            )
            recognizer._model_name = f"SenseVoice ({sdir.name})"
            return recognizer

    # 3. Fallback to Zipformer Transducer
    if (ASR_DIR / "tokens.txt").is_file():
        enc = ASR_DIR / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        dec = ASR_DIR / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
        jnr = ASR_DIR / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        if enc.is_file() and dec.is_file() and jnr.is_file():
            print(f"[ASR Engine] Loading streaming Zipformer ASR ({ASR_DIR.name})...")
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(ASR_DIR / "tokens.txt"),
                encoder=str(enc),
                decoder=str(dec),
                joiner=str(jnr),
                num_threads=num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                provider=provider,
            )
            recognizer._model_name = f"Zipformer ({ASR_DIR.name})"
            return recognizer

    raise FileNotFoundError(
        f"No recognized ASR model found in '{ASR_DIR}' or 'models/'.\n"
        "Please download Whisper Base ASR via:\n"
        "  cd models && wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2 && tar xvf sherpa-onnx-whisper-base.en.tar.bz2"
    )


def create_tts() -> sherpa_onnx.OfflineTts:
    if (KOKORO_DIR / "model.onnx").is_file():
        voice_names = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky"]
        vname = voice_names[TTS_SID] if TTS_SID < len(voice_names) else f"ID {TTS_SID}"
        print(f"[TTS Engine] Loading Kokoro TTS (hexgrad/Kokoro-82M, voice: {vname})...")
        kokoro_cfg = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(KOKORO_DIR / "model.onnx"),
            voices=str(KOKORO_DIR / "voices.bin"),
            tokens=str(KOKORO_DIR / "tokens.txt"),
            data_dir=str(KOKORO_DIR / "espeak-ng-data"),
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=kokoro_cfg,
                provider=os.environ.get("SHERPA_PROVIDER", "cpu"),
                num_threads=TTS_THREADS,
            ),
            max_num_sentences=1,
        )
        if not config.validate():
            raise RuntimeError("Invalid Kokoro TTS configuration")
        return sherpa_onnx.OfflineTts(config)

    print("[TTS Engine] Loading Piper Amy VITS TTS...")
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(TTS_DIR / "en_US-amy-low.onnx"),
                tokens=str(TTS_DIR / "tokens.txt"),
                data_dir=str(TTS_DIR / "espeak-ng-data"),
            ),
            provider="cpu",
            num_threads=TTS_THREADS,
        ),
        max_num_sentences=1,
    )
    if not config.validate():
        raise RuntimeError("Invalid TTS configuration")
    return sherpa_onnx.OfflineTts(config)


def create_vad() -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(VAD_MODEL)
    config.silero_vad.threshold = VAD_THRESHOLD
    config.silero_vad.min_silence_duration = VAD_SILENCE_SECONDS
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = VAD_MAX_SPEECH_SECONDS
    config.sample_rate = SAMPLE_RATE
    return (
        sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30),
        config.silero_vad.window_size,
    )


def recognize(recognizer: sherpa_onnx.OnlineRecognizer | sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    if isinstance(recognizer, sherpa_onnx.OfflineRecognizer):
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        text = re.sub(r"<\|.*?\|>", "", text).strip()
        return text
    else:
        stream = recognizer.create_stream()
        # Provide 0.2s leading silence context
        stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32))
        stream.accept_waveform(SAMPLE_RATE, samples)
        # Provide 0.3s trailing silence context to flush final tokens
        stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32))
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        return recognizer.get_result(stream).strip()


def microphone_device() -> int | str | None:
    value = os.environ.get("MIC_DEVICE") or os.environ.get("SHERPA_ONNX_MIC_DEVICE")
    if not value or value == "default":
        return None
    return int(value) if value.isdigit() else value


def main() -> int:
    print("=" * 64)
    print("      LAED: Streaming Speech-to-Speech (Edge0 + sherpa-onnx)")
    print("=" * 64)

    if not VAD_MODEL.is_file():
        raise FileNotFoundError(f"Missing Silero VAD model at {VAD_MODEL}. Run scripts/setup_models.sh first.")

    recognizer = create_recognizer()
    tts = create_tts()
    vad, window_size = create_vad()
    tts_player = StreamingTtsPlayer(tts)
    llm = LocalLlmClient(LLM_URL)
    llm.start()

    device = microphone_device()
    turn_counter = 0

    if sd is None:
        raise RuntimeError(
            "sounddevice library is required for local CLI mic/speaker audio. "
            "For cloud/headless servers, please run the Web Studio: python3 web/server.py"
        )

    print("\n[Status] System Ready! Speak into your microphone. (Ctrl+C to exit)\n" + "-" * 64)
    speaking = False

    try:
        with sd.InputStream(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=window_size,
            latency="low",
        ) as mic:
            while True:
                block, overflowed = mic.read(window_size)
                if overflowed:
                    continue

                # Echo Suppression: If assistant is actively speaking, drop mic audio
                if tts_player.is_busy():
                    continue

                vad.accept_waveform(block[:, 0])

                if vad.is_speech_detected() and not speaking:
                    speaking = True
                    print("\r[🎤 Listening...]", end="", flush=True)

                while not vad.empty():
                    samples = np.asarray(vad.front.samples, dtype=np.float32)
                    vad.pop()
                    speaking = False
                    turn_counter += 1

                    # Temporarily stop mic to prevent acoustic feedback
                    mic.stop()
                    turn_start_time = time.perf_counter()
                    ttfa_timestamp: float | None = None

                    def on_first_audio_callback(timestamp: float) -> None:
                        nonlocal ttfa_timestamp
                        ttfa_timestamp = timestamp

                    tts_player.set_turn(turn_counter, on_first_audio_callback)

                    try:
                        # 1. ASR Transcribe
                        asr_start = time.perf_counter()
                        transcript = recognize(recognizer, samples)
                        asr_time_ms = (time.perf_counter() - asr_start) * 1000

                        if not transcript:
                            print("\r[🎤 (No clear speech detected)]", flush=True)
                            tts_player.end_turn()
                            continue

                        print(f"\r[👤 You]: \"{transcript}\"  (ASR: {asr_time_ms:.0f}ms)")
                        print("[🤖 Amy]: ", end="", flush=True)

                        # 2. LLM Stream + Sentence Chunking
                        def on_token(token: str) -> None:
                            print(token, end="", flush=True)

                        def on_sentence(sentence: str) -> None:
                            tts_player.enqueue_sentence(sentence)

                        llm_start = time.perf_counter()
                        reply, ttft_sec = llm.stream_chat(transcript, on_sentence, on_token)
                        tts_player.end_turn()

                        # 3. Wait for all speech audio playback to finish
                        tts_player.wait_until_done()
                        turn_total_time = (time.perf_counter() - turn_start_time) * 1000
                        ttft_ms = ttft_sec * 1000
                        ttfa_ms = ((ttfa_timestamp - llm_start) * 1000) if ttfa_timestamp else 0

                        print(
                            f"\n  ↳ [Metrics] ASR: {asr_time_ms:.0f}ms | "
                            f"TTFT: {ttft_ms:.0f}ms | "
                            f"TTFA: {ttfa_ms:.0f}ms | "
                            f"Total: {turn_total_time:.0f}ms\n"
                        )

                    except Exception as error:
                        print(f"\n[Turn Error]: {error}", file=sys.stderr)
                        tts_player.end_turn()
                    finally:
                        # Clear residual audio, reset VAD state, and resume mic
                        vad.reset()
                        time.sleep(0.15)  # Guard delay for room reverberation
                        mic.start()
                        print("[🎤 Ready] Listening...", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n[Exiting] Stopping voice session...")
    finally:
        sd.stop()
        tts_player.stop()
        llm.close()
        print("[Done] Session closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
