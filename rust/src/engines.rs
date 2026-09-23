use crate::memory::{ChatMessage, ChatRole};
use anyhow::{bail, Result};
use std::f32::consts::TAU;
use std::path::Path;

pub const SYSTEM_PROMPT: &str = "Bạn là trợ lý hội thoại trên điện thoại. Trả lời bằng ngôn ngữ của người dùng, đúng trọng tâm, tự nhiên và chỉ từ một đến ba câu ngắn. Không hiển thị suy luận nội bộ.";

pub struct SynthesizedAudio {
    pub samples: Vec<i16>,
    pub sample_rate_hz: u32,
}

pub trait AsrEngine: Send {
    fn transcribe(
        &mut self,
        pcm16_mono_16khz: &[i16],
        language: &str,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String>;
}

pub trait LlmEngine: Send {
    fn generate(
        &mut self,
        system_prompt: &str,
        history: &[ChatMessage],
        user_text: &str,
        max_output_tokens: u32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String>;
}

pub trait TtsEngine: Send {
    fn synthesize(
        &mut self,
        text: &str,
        language: &str,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<SynthesizedAudio>;
}

pub struct EngineBundle {
    pub asr: Box<dyn AsrEngine>,
    pub llm: Box<dyn LlmEngine>,
    pub tts: Box<dyn TtsEngine>,
}

impl EngineBundle {
    pub fn demo() -> Self {
        Self {
            asr: Box::new(DemoAsr),
            llm: Box::new(DemoLlm),
            tts: Box::new(DemoTts),
        }
    }

    pub fn native(model_root: &str) -> Result<Self> {
        let whisper = Path::new(model_root).join("whisper/ggml-tiny.bin");
        let llama = Path::new(model_root).join("qwen3/model.gguf");
        let tts = Path::new(model_root).join("tts-vais1000");
        if !whisper.is_file() || !llama.is_file() || !tts.is_dir() {
            bail!("Thiếu model trong thư mục ứng dụng: {model_root}");
        }
        bail!("Native model adapters chưa được link; xem docs/MODEL_BACKENDS.md")
    }

}

struct DemoAsr;
impl AsrEngine for DemoAsr {
    fn transcribe(&mut self, _pcm: &[i16], _language: &str, _is_cancelled: &dyn Fn() -> bool) -> Result<String> {
        Ok("Xin chào, hãy giới thiệu ngắn về bạn.".into())
    }
}

struct DemoLlm;
impl LlmEngine for DemoLlm {
    fn generate(
        &mut self,
        _system: &str,
        history: &[ChatMessage],
        _user: &str,
        _max: u32,
        _is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        // Exercise the real history representation in demo mode as a wiring check.
        let _history_characters: usize = history
            .iter()
            .map(|message| message.content.len() + matches!(message.role, ChatRole::Assistant) as usize)
            .sum();
        Ok("Mình là trợ lý giọng nói chạy cục bộ. Hiện ứng dụng đang ở chế độ demo pipeline.".into())
    }
}

struct DemoTts;
impl TtsEngine for DemoTts {
    fn synthesize(&mut self, text: &str, _language: &str, is_cancelled: &dyn Fn() -> bool) -> Result<SynthesizedAudio> {
        let sample_rate_hz = 22_050;
        let seconds = (text.chars().count() as f32 / 24.0).clamp(0.5, 3.0);
        let sample_count = (sample_rate_hz as f32 * seconds) as usize;
        let samples: Vec<i16> = (0..sample_count)
            .map(|index| {
                let fade = (index as f32 / (sample_rate_hz as f32 / 50.0)).min(1.0);
                ((TAU * 440.0 * index as f32 / sample_rate_hz as f32).sin() * fade * 2_400.0) as i16
            })
            .take_while(|_| !is_cancelled())
            .collect();
        if samples.len() != sample_count {
            bail!("Lượt nói đã bị hủy");
        }
        Ok(SynthesizedAudio { samples, sample_rate_hz })
    }
}
