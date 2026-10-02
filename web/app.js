/**
 * LAED Voice Studio — Web Audio & Speech-to-Speech Engine Client
 */

// DOM Elements
const canvas = document.getElementById('visualizer-canvas');
const ctx = canvas.getContext('2d');
const micButton = document.getElementById('mic-button');
const micRadar = document.getElementById('mic-radar');
const stateBadge = document.getElementById('state-badge');
const stateText = document.getElementById('state-text');
const liveUserSpeech = document.getElementById('live-user-speech');
const liveAiSpeech = document.getElementById('live-ai-speech');
const micHintText = document.getElementById('mic-hint-text');

// Metrics
const valAsr = document.getElementById('val-metric-asr');
const valTtft = document.getElementById('val-metric-ttft');
const valTtfa = document.getElementById('val-metric-ttfa');
const valTotal = document.getElementById('val-metric-total');

// Drawer & Chat
const btnToggleTranscript = document.getElementById('btn-toggle-transcript');
const btnCloseDrawer = document.getElementById('btn-close-drawer');
const chatDrawer = document.getElementById('chat-drawer');
const chatMessageList = document.getElementById('chat-message-list');
const chatEmptyState = document.getElementById('chat-empty-state');
const textChatForm = document.getElementById('text-chat-form');
const textInput = document.getElementById('text-input');

// Settings Modal
const btnOpenSettings = document.getElementById('btn-open-settings');
const btnCloseSettings = document.getElementById('btn-close-settings');
const modalSettings = document.getElementById('modal-settings');
const inputServerUrl = document.getElementById('input-server-url');
const btnTestConnection = document.getElementById('btn-test-connection');
const btnSaveSettings = document.getElementById('btn-save-settings');
const modalStatusDot = document.getElementById('modal-status-dot');
const modalStatusText = document.getElementById('modal-status-text');

function getBackendUrl() {
  const custom = localStorage.getItem('laed_backend_url');
  if (custom && custom.trim()) {
    return custom.trim().replace(/\/+$/, '');
  }
  if (window.location.hostname.endsWith('github.io')) {
    return '';
  }
  return window.location.origin;
}

// Audio Context & State
let audioCtx = null;
let micStream = null;
let mediaRecorder = null;
let audioChunks = [];
let analyser = null;
let micVolume = 0;
let isRecording = false;
let isSpacePressed = false;
let currentAppPhase = 'idle'; // 'idle' | 'listening' | 'thinking' | 'speaking'

// Playback Queue
class AudioQueuePlayer {
  constructor() {
    this.queue = [];
    this.isPlaying = false;
    this.currentAudio = null;
  }

  enqueue(base64Wav) {
    this.queue.push(base64Wav);
    if (!this.isPlaying) {
      this.playNext();
    }
  }

  playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      if (currentAppPhase === 'speaking') {
        setPhase('idle');
      }
      return;
    }

    this.isPlaying = true;
    setPhase('speaking');

    const base64Wav = this.queue.shift();
    const audioUrl = `data:audio/wav;base64,${base64Wav}`;
    this.currentAudio = new Audio(audioUrl);

    this.currentAudio.onended = () => {
      this.playNext();
    };

    this.currentAudio.onerror = (e) => {
      console.error('Audio playback error', e);
      this.playNext();
    };

    this.currentAudio.play().catch((err) => {
      console.warn('Playback autoplay blocked or failed', err);
      this.playNext();
    });
  }

  stopAll() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
    }
    this.queue = [];
    this.isPlaying = false;
  }
}

const audioQueue = new AudioQueuePlayer();

// Update Phase UI
function setPhase(phase, customLabel) {
  currentAppPhase = phase;
  stateBadge.className = `state-badge state-${phase}`;

  const labels = {
    idle: 'Practice English with Teacher Sarah',
    listening: 'Listening to You...',
    thinking: 'Edge0-8B is Thinking...',
    speaking: 'Teacher Sarah is Speaking...'
  };

  stateText.textContent = customLabel || labels[phase] || 'Ready';

  if (phase === 'listening') {
    micButton.classList.add('is-recording');
    micRadar.parentElement.classList.add('active');
  } else {
    micButton.classList.remove('is-recording');
    micRadar.parentElement.classList.remove('active');
  }
}

// --------------------------------------------------------------------------
// Audio Recording & Encoding (16kHz Mono WAV in Pure JS)
// --------------------------------------------------------------------------

async function initMicrophone() {
  if (micStream) return true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;

    const source = audioCtx.createMediaStreamSource(micStream);
    source.connect(analyser);

    return true;
  } catch (err) {
    alert('Microphone access denied or unavailable: ' + err.message);
    return false;
  }
}

function startRecording() {
  if (isRecording) return;
  initMicrophone().then((ok) => {
    if (!ok) return;

    audioQueue.stopAll();
    audioChunks = [];
    isRecording = true;
    setPhase('listening');

    liveUserSpeech.textContent = '';
    liveAiSpeech.textContent = 'Listening...';

    // Collect Float32 samples via ScriptProcessor for pristine 16kHz PCM
    const bufferSize = 4096;
    const recorderNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    const recordedSamples = [];

    recorderNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      recordedSamples.push(new Float32Array(inputData));
    };

    const source = audioCtx.createMediaStreamSource(micStream);
    source.connect(recorderNode);
    recorderNode.connect(audioCtx.destination);

    window._activeRecorder = {
      source,
      recorderNode,
      recordedSamples
    };
  });
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  setPhase('thinking');
  liveAiSpeech.textContent = 'Transcribing speech with Zipformer...';

  if (window._activeRecorder) {
    const { source, recorderNode, recordedSamples } = window._activeRecorder;
    source.disconnect();
    recorderNode.disconnect();
    window._activeRecorder = null;

    // Resample/Merge to 16kHz Float32
    const totalLen = recordedSamples.reduce((acc, b) => acc + b.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of recordedSamples) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    // Downsample if browser provided higher sample rate (e.g. 48kHz or 44.1kHz)
    const targetSampleRate = 16000;
    const finalSamples = resampleAudio(merged, audioCtx.sampleRate, targetSampleRate);
    const wavBlob = encodeWAV(finalSamples, targetSampleRate);

    sendVoiceQuery(wavBlob);
  }
}

function resampleAudio(audioBuffer, fromSampleRate, toSampleRate) {
  if (fromSampleRate === toSampleRate) return audioBuffer;
  const ratio = fromSampleRate / toSampleRate;
  const newLength = Math.round(audioBuffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const originalIndex = i * ratio;
    const index1 = Math.floor(originalIndex);
    const index2 = Math.min(index1 + 1, audioBuffer.length - 1);
    const fraction = originalIndex - index1;
    result[i] = audioBuffer[index1] * (1 - fraction) + audioBuffer[index2] * fraction;
  }
  return result;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

// --------------------------------------------------------------------------
// SSE Stream Communication with Backend
// --------------------------------------------------------------------------

async function sendVoiceQuery(wavBlob) {
  const backend = getBackendUrl();
  if (!backend && window.location.hostname.endsWith('github.io')) {
    openSettingsModal('Please connect your Vast.ai or Tunnel URL first!');
    liveAiSpeech.textContent = 'Please configure your backend server URL in Settings.';
    setPhase('idle');
    return;
  }

  try {
    const response = await fetch(`${backend}/api/chat-voice`, {
      method: 'POST',
      body: wavBlob,
      headers: { 'Content-Type': 'audio/wav' }
    });

    handleSseStream(response);
  } catch (err) {
    console.error('API Error', err);
    liveAiSpeech.textContent = `Error: ${err.message}. If using GitHub Pages, ensure backend has HTTPS (Cloudflare Tunnel).`;
    setPhase('idle');
  }
}

async function sendTextQuery(text) {
  const backend = getBackendUrl();
  if (!backend && window.location.hostname.endsWith('github.io')) {
    openSettingsModal('Please connect your Vast.ai or Tunnel URL first!');
    return;
  }

  audioQueue.stopAll();
  setPhase('thinking');
  liveUserSpeech.textContent = `"${text}"`;
  liveAiSpeech.textContent = '';
  appendChatMessage('user', text);

  try {
    const response = await fetch(`${backend}/api/chat-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: jsonStringify({ text })
    });

    handleSseStream(response);
  } catch (err) {
    console.error('API Error', err);
    liveAiSpeech.textContent = `Error: ${err.message}`;
    setPhase('idle');
  }
}

async function handleSseStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let fullAiText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // Keep incomplete tail

    for (const chunk of lines) {
      const line = chunk.trim();
      if (!line.startsWith('data: ')) continue;
      const dataStr = line.slice(6);

      try {
        const ev = JSON.parse(dataStr);

        if (ev.type === 'transcript') {
          liveUserSpeech.textContent = `"${ev.text}"`;
          valAsr.innerHTML = `${ev.asr_ms} <span class="unit">ms</span>`;
          appendChatMessage('user', ev.text, `${ev.asr_ms}ms`);
          liveAiSpeech.textContent = '';
        } else if (ev.type === 'token') {
          fullAiText += ev.token;
          liveAiSpeech.textContent = fullAiText;
        } else if (ev.type === 'audio_sentence') {
          // Play synthesized audio sentence immediately!
          audioQueue.enqueue(ev.audio);
        } else if (ev.type === 'done') {
          if (ev.metrics) {
            valTtft.innerHTML = `${ev.metrics.ttft_ms} <span class="unit">ms</span>`;
            valTtfa.innerHTML = `${ev.metrics.ttfa_ms} <span class="unit">ms</span>`;
            valTotal.innerHTML = `${(ev.metrics.total_ms / 1000).toFixed(1)} <span class="unit">s</span>`;
          }
          appendChatMessage('ai', fullAiText, `TTFA: ${ev.metrics.ttfa_ms}ms`);
        }
      } catch (err) {
        console.warn('Malformed SSE line:', dataStr);
      }
    }
  }
}

function appendChatMessage(role, text, tag) {
  if (!text.trim()) return;
  chatEmptyState.style.display = 'none';

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.innerHTML = `
    <div>${escapeHtml(text)}</div>
    ${tag ? `<div class="bubble-meta">${tag}</div>` : ''}
  `;
  chatMessageList.appendChild(bubble);
  chatMessageList.scrollTop = chatMessageList.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (m) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[m]));
}

function jsonStringify(obj) {
  return JSON.stringify(obj);
}

// --------------------------------------------------------------------------
// Canvas 3D Audio-Reactive Orb Visualizer
// --------------------------------------------------------------------------

let angle = 0;
function drawVisualizer() {
  requestAnimationFrame(drawVisualizer);

  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  // Read mic frequency if listening
  let freqIntensity = 0;
  if (analyser && isRecording) {
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
    const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
    freqIntensity = avg / 128;
  }

  angle += currentAppPhase === 'thinking' ? 0.05 : 0.015;

  // Base Orb Colors based on phase
  let color1 = 'rgba(6, 182, 212, '; // Cyan
  let color2 = 'rgba(139, 92, 246, '; // Purple

  if (currentAppPhase === 'listening') {
    color1 = 'rgba(244, 63, 94, '; // Rose
    color2 = 'rgba(234, 88, 12, '; // Orange
  } else if (currentAppPhase === 'speaking') {
    color1 = 'rgba(16, 185, 129, '; // Emerald
    color2 = 'rgba(6, 182, 212, '; // Cyan
  } else if (currentAppPhase === 'thinking') {
    color1 = 'rgba(139, 92, 246, '; // Purple
    color2 = 'rgba(236, 72, 153, '; // Pink
  }

  const baseRadius = 85 + (isRecording ? freqIntensity * 40 : Math.sin(angle * 2) * 5);

  // Outer Aura
  const gradient = ctx.createRadialGradient(cx, cy, baseRadius * 0.4, cx, cy, baseRadius * 1.8);
  gradient.addColorStop(0, color1 + '0.8)');
  gradient.addColorStop(0.5, color2 + '0.35)');
  gradient.addColorStop(1, 'transparent');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * 1.8, 0, Math.PI * 2);
  ctx.fill();

  // Orbital Frequency Ring
  const points = 64;
  ctx.beginPath();
  for (let i = 0; i <= points; i++) {
    const a = (i / points) * Math.PI * 2 + angle;
    const wave = Math.sin(a * 6 + angle * 3) * (currentAppPhase === 'speaking' ? 18 : 6);
    const r = baseRadius + wave + (isRecording ? freqIntensity * 25 : 0);
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = color1 + '0.9)';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Inner Core Orb
  const coreGrad = ctx.createRadialGradient(cx - 20, cy - 20, 10, cx, cy, baseRadius * 0.9);
  coreGrad.addColorStop(0, '#ffffff');
  coreGrad.addColorStop(0.3, color1 + '1)');
  coreGrad.addColorStop(1, color2 + '0.8)');

  ctx.fillStyle = coreGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * 0.85, 0, Math.PI * 2);
  ctx.fill();
}

drawVisualizer();

// --------------------------------------------------------------------------
// Event Listeners & Keyboard Triggers
// --------------------------------------------------------------------------

// Click Mic Button to Toggle
micButton.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// Spacebar Hold-to-Talk
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !isSpacePressed && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    isSpacePressed = true;
    startRecording();
  }
});

window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && isSpacePressed) {
    e.preventDefault();
    isSpacePressed = false;
    stopRecording();
  }
});

// Toggle Chat Drawer
btnToggleTranscript.addEventListener('click', () => {
  chatDrawer.classList.toggle('open');
});

btnCloseDrawer.addEventListener('click', () => {
  chatDrawer.classList.remove('open');
});

// Text Chat Form
textChatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const val = textInput.value.trim();
  if (!val) return;
  textInput.value = '';
  sendTextQuery(val);
});

// Settings Modal Logic
function openSettingsModal(hint) {
  modalSettings.classList.add('open');
  inputServerUrl.value = localStorage.getItem('laed_backend_url') || (window.location.hostname.endsWith('github.io') ? '' : window.location.origin);
  if (hint) {
    modalStatusDot.className = 'status-indicator-dot error';
    modalStatusText.textContent = hint;
  }
}

btnOpenSettings.addEventListener('click', () => openSettingsModal());
btnCloseSettings.addEventListener('click', () => modalSettings.classList.remove('open'));
modalSettings.addEventListener('click', (e) => {
  if (e.target === modalSettings) modalSettings.classList.remove('open');
});

btnTestConnection.addEventListener('click', async () => {
  const url = inputServerUrl.value.trim().replace(/\/+$/, '');
  if (!url) {
    modalStatusDot.className = 'status-indicator-dot error';
    modalStatusText.textContent = 'Please enter a URL';
    return;
  }
  modalStatusText.textContent = 'Testing connection...';
  modalStatusDot.className = 'status-indicator-dot';

  try {
    const res = await fetch(`${url}/api/status`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    modalStatusDot.className = 'status-indicator-dot connected';
    modalStatusText.textContent = `Connected! Model: ${data.tts_model || 'Ready'}`;
  } catch (err) {
    modalStatusDot.className = 'status-indicator-dot error';
    modalStatusText.textContent = `Failed: ${err.message}. Ensure HTTPS if on GitHub Pages.`;
  }
});

btnSaveSettings.addEventListener('click', async () => {
  const url = inputServerUrl.value.trim().replace(/\/+$/, '');
  localStorage.setItem('laed_backend_url', url);
  modalSettings.classList.remove('open');
  checkBackendHealth();
});

async function checkBackendHealth() {
  const backend = getBackendUrl();
  if (!backend) return;
  try {
    const res = await fetch(`${backend}/api/status`);
    if (res.ok) {
      const data = await res.json();
      if (data.tts_model) {
        const pillTts = document.getElementById('pill-tts');
        if (pillTts) pillTts.innerHTML = `<span class="indicator-dot"></span> TTS: ${data.tts_model}`;
      }
    }
  } catch (_) {}
}

// Auto-check health on load
checkBackendHealth();

