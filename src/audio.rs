use std::fs::File;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use asr_features::{
    FeatureMatrix, extract_w2v_bert_features_from_samples, w2v_bert_frontend_config,
};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
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

fn decode_audio_file_to_mono_f32(
    path: &Path,
    config: &AudioDecodeConfig,
) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path)
        .with_context(|| format!("failed to open audio file {}", path.display()))?;
    let media_source = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    let probed = get_probe()
        .format(
            &hint,
            media_source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("failed to probe audio file {}", path.display()))?;
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
