use anyhow::{bail, Result};

pub const ASR_SAMPLE_RATE_HZ: u32 = 16_000;
pub const MAX_RECORDING_SECONDS: usize = 15;
pub const MIN_RECORDING_MS: usize = 250;

pub fn decode_pcm16_le(bytes: &[u8], sample_rate_hz: u32) -> Result<Vec<i16>> {
    if sample_rate_hz == 0 {
        bail!("Sample rate must be positive");
    }
    if bytes.len() % 2 != 0 {
        bail!("PCM16 input contains an incomplete sample");
    }
    let samples: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let duration_ms = samples.len().saturating_mul(1_000) / sample_rate_hz as usize;
    if duration_ms < MIN_RECORDING_MS {
        bail!("Đoạn thu quá ngắn. Hãy giữ nút và nói lại.");
    }
    if samples.len() > sample_rate_hz as usize * MAX_RECORDING_SECONDS {
        bail!("Đoạn thu vượt quá 15 giây.");
    }
    Ok(samples)
}

/// Linear resampling is sufficient for the first integration pass. Replace with a band-limited
/// resampler if device measurements show meaningful ASR degradation.
pub fn resample_mono(input: &[i16], input_rate: u32, output_rate: u32) -> Vec<i16> {
    if input.is_empty() || input_rate == output_rate {
        return input.to_vec();
    }
    let output_len = input.len().saturating_mul(output_rate as usize) / input_rate as usize;
    (0..output_len)
        .map(|out_index| {
            let source = out_index as f64 * input_rate as f64 / output_rate as f64;
            let left = source.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = source - left as f64;
            (input[left] as f64 * (1.0 - fraction) + input[right] as f64 * fraction) as i16
        })
        .collect()
}

pub fn pcm16_to_wav(samples: &[i16], sample_rate_hz: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate_hz.to_le_bytes());
    wav.extend_from_slice(&(sample_rate_hz * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_has_expected_header_and_size() {
        let wav = pcm16_to_wav(&[0, 1, -1], 22_050);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 50);
    }

    #[test]
    fn rejects_odd_pcm_byte_count() {
        assert!(decode_pcm16_le(&[1], 16_000).is_err());
    }
}

