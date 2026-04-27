use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use asr_features::{
    FeatureMatrix, extract_w2v_bert_features_from_samples, w2v_bert_frontend_config,
};
use rand::Rng;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

#[derive(Clone, Debug)]
pub struct AudioDecodeConfig {
    pub fallback_sample_rate: u32,
    pub skip_decode_errors: bool,
}

impl Default for AudioDecodeConfig {
    fn default() -> Self {
        Self {
            fallback_sample_rate: 16_000,
            skip_decode_errors: true,
        }
    }
}

pub struct AudioFeatures {
    pub features: FeatureMatrix,
    pub sample_rate: u32,
    pub sample_count: usize,
    pub decode_elapsed: Duration,
    pub feature_elapsed: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WaveformAugmentConfig {
    pub gain_min_db: Option<f32>,
    pub gain_max_db: Option<f32>,
    pub noise_std: f32,
}

impl WaveformAugmentConfig {
    pub fn is_enabled(&self) -> bool {
        self.gain_min_db.is_some() || self.gain_max_db.is_some() || self.noise_std > 0.0
    }
}

impl AudioFeatures {
    pub fn duration_seconds(&self) -> f64 {
        self.sample_count as f64 / self.sample_rate as f64
    }
}

pub fn audio_file_to_w2v_bert_features(path: impl AsRef<Path>) -> Result<AudioFeatures> {
    audio_file_to_w2v_bert_features_with_config(
        path,
        &AudioDecodeConfig::default(),
        &w2v_bert_frontend_config(None, None, None, None, None, None),
    )
}

pub fn audio_file_to_w2v_bert_features_with_config(
    path: impl AsRef<Path>,
    decode_config: &AudioDecodeConfig,
    frontend_config: &asr_features::W2vBertFrontendConfig,
) -> Result<AudioFeatures> {
    let decode_start = Instant::now();
    let (samples, sample_rate) = decode_audio_file_to_mono_f32(path.as_ref(), decode_config)?;
    samples_to_w2v_bert_features(samples, sample_rate, decode_start, frontend_config)
}

pub fn audio_file_to_w2v_bert_features_with_augmentation(
    path: impl AsRef<Path>,
    decode_config: &AudioDecodeConfig,
    frontend_config: &asr_features::W2vBertFrontendConfig,
    augment: WaveformAugmentConfig,
) -> Result<AudioFeatures> {
    let decode_start = Instant::now();
    let (mut samples, sample_rate) = decode_audio_file_to_mono_f32(path.as_ref(), decode_config)?;
    apply_waveform_augmentation(&mut samples, augment);
    samples_to_w2v_bert_features(samples, sample_rate, decode_start, frontend_config)
}

pub fn audio_bytes_to_w2v_bert_features(
    audio_bytes: impl Into<Vec<u8>>,
    format_hint: Option<&str>,
) -> Result<AudioFeatures> {
    audio_bytes_to_w2v_bert_features_with_config(
        audio_bytes,
        format_hint,
        &AudioDecodeConfig::default(),
        &w2v_bert_frontend_config(None, None, None, None, None, None),
    )
}

pub fn audio_bytes_to_w2v_bert_features_with_config(
    audio_bytes: impl Into<Vec<u8>>,
    format_hint: Option<&str>,
    decode_config: &AudioDecodeConfig,
    frontend_config: &asr_features::W2vBertFrontendConfig,
) -> Result<AudioFeatures> {
    let decode_start = Instant::now();
    let (samples, sample_rate) =
        decode_audio_bytes_to_mono_f32(audio_bytes.into(), format_hint, decode_config)?;
    samples_to_w2v_bert_features(samples, sample_rate, decode_start, frontend_config)
}

pub fn audio_bytes_to_w2v_bert_features_with_augmentation(
    audio_bytes: impl Into<Vec<u8>>,
    format_hint: Option<&str>,
    decode_config: &AudioDecodeConfig,
    frontend_config: &asr_features::W2vBertFrontendConfig,
    augment: WaveformAugmentConfig,
) -> Result<AudioFeatures> {
    let decode_start = Instant::now();
    let (mut samples, sample_rate) =
        decode_audio_bytes_to_mono_f32(audio_bytes.into(), format_hint, decode_config)?;
    apply_waveform_augmentation(&mut samples, augment);
    samples_to_w2v_bert_features(samples, sample_rate, decode_start, frontend_config)
}

fn samples_to_w2v_bert_features(
    samples: Vec<f32>,
    sample_rate: u32,
    decode_start: Instant,
    frontend_config: &asr_features::W2vBertFrontendConfig,
) -> Result<AudioFeatures> {
    let decode_elapsed = decode_start.elapsed();
    let sample_count = samples.len();

    let feature_start = Instant::now();
    let features = extract_w2v_bert_features_from_samples(&samples, sample_rate, frontend_config)
        .context("failed to extract W2V-BERT features")?;

    Ok(AudioFeatures {
        features,
        sample_rate,
        sample_count,
        decode_elapsed,
        feature_elapsed: feature_start.elapsed(),
    })
}

fn apply_waveform_augmentation(samples: &mut [f32], config: WaveformAugmentConfig) {
    if !config.is_enabled() || samples.is_empty() {
        return;
    }
    let mut rng = rand::rng();
    let min_gain = config.gain_min_db.unwrap_or(0.0);
    let max_gain = config.gain_max_db.unwrap_or(min_gain);
    let gain_db = if (max_gain - min_gain).abs() > f32::EPSILON {
        rng.random_range(min_gain..=max_gain)
    } else {
        min_gain
    };
    let gain = 10.0_f32.powf(gain_db / 20.0);
    for sample in samples.iter_mut() {
        let noise = if config.noise_std > 0.0 {
            rng.random_range(-config.noise_std..=config.noise_std)
        } else {
            0.0
        };
        *sample = (*sample * gain + noise).clamp(-1.0, 1.0);
    }
}

fn decode_audio_file_to_mono_f32(
    path: &Path,
    config: &AudioDecodeConfig,
) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path)
        .with_context(|| format!("failed to open audio file {}", path.display()))?;
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    decode_audio_source_to_mono_f32(
        Box::new(file),
        hint,
        config,
        format!("audio file {}", path.display()),
    )
}

fn decode_audio_bytes_to_mono_f32(
    audio_bytes: Vec<u8>,
    format_hint: Option<&str>,
    config: &AudioDecodeConfig,
) -> Result<(Vec<f32>, u32)> {
    if audio_bytes.is_empty() {
        bail!("audio byte buffer is empty");
    }

    let mut hint = Hint::new();
    if let Some(format_hint) = format_hint {
        hint.with_extension(format_hint);
    }

    decode_audio_source_to_mono_f32(
        Box::new(Cursor::new(audio_bytes)),
        hint,
        config,
        "audio byte buffer",
    )
}

fn decode_audio_source_to_mono_f32(
    source: Box<dyn MediaSource>,
    hint: Hint,
    config: &AudioDecodeConfig,
    source_label: impl AsRef<str>,
) -> Result<(Vec<f32>, u32)> {
    let source_label = source_label.as_ref();
    let media_source = MediaSourceStream::new(source, MediaSourceStreamOptions::default());

    let probed = get_probe()
        .format(
            &hint,
            media_source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("failed to probe {source_label}"))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("audio container has no default track"))?;
    if track.codec_params.codec == CODEC_TYPE_NULL {
        bail!("unsupported null audio codec");
    }

    let track_id = track.id;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("failed to create audio decoder")?;

    let mut mono_samples = Vec::new();
    let mut sample_rate = track
        .codec_params
        .sample_rate
        .unwrap_or(config.fallback_sample_rate);

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("decoder reset is not supported for this audio stream");
            }
            Err(error) => return Err(error).context("failed to read audio packet"),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) if config.skip_decode_errors => continue,
            Err(error) => return Err(error).context("failed to decode audio packet"),
        };

        append_mono_samples(decoded, &mut mono_samples, &mut sample_rate);
    }

    if mono_samples.is_empty() {
        bail!("decoded audio stream is empty");
    }

    Ok((mono_samples, sample_rate))
}

fn append_mono_samples(decoded: AudioBufferRef<'_>, output: &mut Vec<f32>, sample_rate: &mut u32) {
    let spec = *decoded.spec();
    *sample_rate = spec.rate;
    let channels = spec.channels.count().max(1);
    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    sample_buffer.copy_interleaved_ref(decoded);

    for frame in sample_buffer.samples().chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        output.push(sum / channels as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_wav_bytes() {
        let samples = [0_i16, 16_384, -16_384, 0];
        let wav = mono_pcm_wav_bytes(16_000, &samples);

        let (decoded, sample_rate) =
            decode_audio_bytes_to_mono_f32(wav, Some("wav"), &AudioDecodeConfig::default())
                .expect("WAV bytes should decode");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(decoded.len(), samples.len());
        assert!((decoded[1] - 0.5).abs() < 0.001);
        assert!((decoded[2] + 0.5).abs() < 0.001);
    }

    #[test]
    fn rejects_empty_audio_bytes() {
        let error =
            decode_audio_bytes_to_mono_f32(Vec::new(), Some("wav"), &AudioDecodeConfig::default())
                .expect_err("empty bytes should fail");

        assert!(error.to_string().contains("audio byte buffer is empty"));
    }

    fn mono_pcm_wav_bytes(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        bytes
    }
}
