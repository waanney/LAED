# LAED Voice — AI English Teacher (Edge0-8B × Kokoro-TTS)

Ultra-low latency conversational AI English Teacher Studio powered by:
- **LLM**: **Edge0-8B** (or OpenAI-compatible streaming server) with **Teacher Sarah** persona (clear spoken English, gentle mistake corrections, 1–2 sentence conversational pacing).
- **TTS**: **hexgrad/Kokoro-82M** (24kHz high-fidelity neural voice, `af_heart`).
- **ASR**: **sherpa-onnx** OpenAI Whisper Base / SenseVoice / Zipformer (high-accuracy conversational English & accent recognition).
- **Frontend**: Responsive Single-Page Application (SPA) with 3D audio-reactive orb, live subtitles, latency HUD, and remote backend connectivity (deployable on **GitHub Pages**).

---

## ⚡ Architecture

```text
[ Browser / GitHub Pages ]
  │
  ├─ 🎤 Microphone (16kHz PCM Web Audio)
  │      ↓  (POST /api/chat-voice)
[ Backend Server / Vast.ai GPU ]
  │
  ├─ 1. Whisper Base ASR (High-accuracy Speech → Text)
  ├─ 2. Edge0-8B / Llama 3.1 8B LLM (Prompt: English Teacher Sarah)
  │      ↓  (Streaming SSE Tokens)
  ├─ 3. Sentence Chunker ([.!?\n] boundary detection)
  ├─ 4. Kokoro-TTS Engine (Sentence N+1 synthesizes while N plays)
  │      ↓  (SSE Audio Chunks: base64 WAV 24kHz)
[ Browser AudioQueuePlayer ]
  │
  └─ 🔊 Continuous, seamless audio playback + 3D Orb Visualizer
```

---

## 🚀 Quick Start (Local)

### 1. Download Model Weights
Run the automated downloader to fetch Whisper Base ASR, Silero VAD, and Kokoro-TTS:
```bash
./scripts/setup_models.sh
```

### 2. Launch the Web Studio
```bash
./run_web.sh
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

---

## 🌐 Deploying to GitHub Pages + Vast.ai

### Frontend (GitHub Pages)
1. Push this repository to GitHub.
2. Go to **Repository Settings** → **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch** and choose `/web` folder (or root).
4. Your Web Studio will be live at `https://<username>.github.io/<repo-name>/`.

### Backend (Vast.ai GPU Instance)
1. Rent an instance on **Vast.ai** (Recommended: **1x RTX 3090 24GB**, Disk **60–80 GB**, PyTorch / CUDA image).
2. Clone repo and download models:
   ```bash
   git clone https://github.com/waanney/LAED.git
   cd LAED
   ./scripts/setup_models.sh
   ```
3. Start the Web Server:
   ```bash
   python3 web/server.py
   ```
4. Expose an HTTPS endpoint via Cloudflare Tunnel (required because GitHub Pages enforces HTTPS):
   ```bash
   cloudflared tunnel --url http://127.0.0.1:7860
   ```
5. In your GitHub Pages UI, click the **Settings (⚙️)** button in the top right, enter your Cloudflare Tunnel HTTPS URL, and click **Connect**.

---

## 👩‍🏫 Persona & Prompt Configuration

The model is configured with Teacher Sarah's persona:
> *"You are Sarah, an encouraging and patient native English teacher. Your sole mission is to help the student practice spoken English naturally. Always communicate strictly in natural, conversational English. Keep your spoken responses short and natural, exactly 1 to 2 sentences (maximum 35 words). If the student makes any grammatical, vocabulary, or phrasing errors, first gently correct them in a friendly manner, then ask a brief follow-up question."*

---

## 📂 Project Structure

```text
├── run_web.sh             # Launch Web Studio server
├── run_s2s.sh             # Launch CLI terminal speech-to-speech
├── scripts/
│   ├── setup_models.sh    # Automated ASR, VAD, and Kokoro model downloader
│   ├── voice_chat.py      # Core streaming S2S pipeline (Chunker, LLM SSE, Kokoro TTS)
│   └── test_pipeline.py   # Benchmark test script
├── web/
│   ├── index.html         # Modern SPA UI with 3D canvas visualizer & HUD
│   ├── style.css          # Glassmorphic dark theme with smooth micro-animations
│   ├── app.js             # Web Audio API recorder, SSE parser, and audio queue player
│   └── server.py          # Fast HTTP/SSE streaming backend
└── .gitignore
```
