use crate::audio::{decode_pcm16_le, pcm16_to_wav, resample_mono, ASR_SAMPLE_RATE_HZ};
use crate::engines::{EngineBundle, SYSTEM_PROMPT};
use crate::memory::ConversationMemory;
use anyhow::{bail, Result};
use flutter_rust_bridge::{frb, StreamSink};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;

#[derive(Clone)]
pub struct SpeechConfig {
    pub model_root: String,
    pub demo_mode: bool,
    pub context_tokens: u32,
    pub max_output_tokens: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    Idle,
    Recording,
    Transcribing,
    Generating,
    Speaking,
    Error,
}

#[derive(Clone)]
pub struct PipelineEvent {
    pub turn_id: i64,
    pub phase: PipelinePhase,
    pub transcript: Option<String>,
    pub response: Option<String>,
    pub audio_wav: Option<Vec<u8>>,
    pub error: Option<String>,
}

#[frb(opaque)]
pub struct SpeechCore {
    config: SpeechConfig,
    next_turn_id: AtomicI64,
    active_turn_id: AtomicI64,
    memory: Mutex<ConversationMemory>,
    engines: Mutex<EngineBundle>,
}

impl SpeechCore {
    pub fn create(config: SpeechConfig) -> Result<Self> {
        if config.context_tokens == 0 || config.context_tokens > 8_192 {
            bail!("context_tokens nằm ngoài giới hạn an toàn");
        }
        if config.max_output_tokens == 0 || config.max_output_tokens > 512 {
            bail!("max_output_tokens nằm ngoài giới hạn an toàn");
        }
        let engines = if config.demo_mode {
            EngineBundle::demo()
        } else {
            EngineBundle::native(&config.model_root)?
        };
        Ok(Self {
            config,
            next_turn_id: AtomicI64::new(0),
            active_turn_id: AtomicI64::new(0),
            memory: Mutex::new(ConversationMemory::new(6, 5_000)),
            engines: Mutex::new(engines),
        })
    }

    pub fn is_demo_mode(&self) -> bool {
        self.config.demo_mode
    }

    pub fn begin_turn(&self) -> i64 {
        let turn_id = self.next_turn_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.active_turn_id.store(turn_id, Ordering::SeqCst);
        turn_id
    }

    pub fn cancel_turn(&self) {
        let invalid_id = self.next_turn_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.active_turn_id.store(invalid_id, Ordering::SeqCst);
    }

    pub fn clear_session(&self) {
        self.cancel_turn();
        self.memory.lock().expect("memory mutex poisoned").clear();
    }

    /// flutter_rust_bridge exposes the StreamSink parameter as a Dart Stream<PipelineEvent>.
    pub fn process_turn(
        &self,
        turn_id: i64,
        pcm16_le: Vec<u8>,
        input_sample_rate_hz: u32,
        sink: StreamSink<PipelineEvent>,
    ) -> Result<()> {
        self.ensure_current(turn_id)?;
        let is_cancelled = || self.active_turn_id.load(Ordering::SeqCst) != turn_id;
        let input = decode_pcm16_le(&pcm16_le, input_sample_rate_hz)?;
        let asr_audio = resample_mono(&input, input_sample_rate_hz, ASR_SAMPLE_RATE_HZ);

        self.emit(&sink, turn_id, PipelinePhase::Transcribing, None, None, None, None)?;
        let transcript = self
            .engines
            .lock()
            .expect("engine mutex poisoned")
            .asr
            .transcribe(&asr_audio, "vi", &is_cancelled)?
            .trim()
            .to_owned();
        self.ensure_current(turn_id)?;
        if transcript.is_empty() {
            bail!("Mình chưa nghe rõ. Bạn thử nói lại nhé.");
        }
        self.emit(
            &sink,
            turn_id,
            PipelinePhase::Generating,
            Some(transcript.clone()),
            None,
            None,
            None,
        )?;

        let history = self.memory.lock().expect("memory mutex poisoned").snapshot();
        let mut response = self
            .engines
            .lock()
            .expect("engine mutex poisoned")
            .llm
            .generate(
                SYSTEM_PROMPT,
                &history,
                &transcript,
                self.config.max_output_tokens,
                &is_cancelled,
            )?
            .trim()
            .to_owned();
        self.ensure_current(turn_id)?;
        if response.len() > 600 {
            let boundary = response
                .char_indices()
                .map(|(index, _)| index)
                .take_while(|index| *index <= 600)
                .last()
                .unwrap_or(0);
            response.truncate(boundary);
        }
        if response.is_empty() {
            bail!("Mô hình không tạo được câu trả lời.");
        }
        self.emit(
            &sink,
            turn_id,
            PipelinePhase::Speaking,
            Some(transcript.clone()),
            Some(response.clone()),
            None,
            None,
        )?;

        let speech = self
            .engines
            .lock()
            .expect("engine mutex poisoned")
            .tts
            .synthesize(&response, "vi", &is_cancelled)?;
        self.ensure_current(turn_id)?;
        let wav = pcm16_to_wav(&speech.samples, speech.sample_rate_hz);
        self.memory
            .lock()
            .expect("memory mutex poisoned")
            .add_exchange(transcript.clone(), response.clone());
        self.emit(
            &sink,
            turn_id,
            PipelinePhase::Idle,
            Some(transcript),
            Some(response),
            Some(wav),
            None,
        )?;
        Ok(())
    }

    fn ensure_current(&self, turn_id: i64) -> Result<()> {
        if self.active_turn_id.load(Ordering::SeqCst) != turn_id {
            bail!("Lượt nói đã bị hủy");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        &self,
        sink: &StreamSink<PipelineEvent>,
        turn_id: i64,
        phase: PipelinePhase,
        transcript: Option<String>,
        response: Option<String>,
        audio_wav: Option<Vec<u8>>,
        error: Option<String>,
    ) -> Result<()> {
        self.ensure_current(turn_id)?;
        sink.add(PipelineEvent { turn_id, phase, transcript, response, audio_wav, error })
            .map_err(|_| anyhow::anyhow!("Dart event stream đã đóng"))
    }
}
